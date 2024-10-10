import torch
import torch.nn.functional as F
import numpy as np 


def initialize_params(args):

    params = dict({})
    DEVICE = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
    params['device'] = DEVICE

    params['nanometers'] = 1E-9
    params['degrees']    = np.pi / 180

    params['upsample'] = 1
    params['normalize_psf'] = args.normalize_psf

    params['magnification'] = args.mag
    params['sensor_pixel'] = 5E-6
    params['b_sqrt'] = args.b_sqrt

    params['f'] = 2.5E-3
    params['v'] = 1.125*params['f']

    lambda_base = [606.0, 511.0, 462.0]
    params['lambda_base'] = lambda_base

    params['arraySize'] = 9     # 3*3 metalens array
    params['batchSize'] = np.size(lambda_base) * params['arraySize']
    num_pixels = 1429           # corresponding to 0.5mm aperture
    params['pixels_aperture'] = num_pixels
    pixelsX = num_pixels
    pixelsY = num_pixels
    params['pixelsX'] = pixelsX
    params['pixelsY'] = pixelsY

    params['wavelength_nominal'] = 452E-9  
    params['wavelength_hyperboloid'] = 511E-9
    params['pitch'] = 350E-9
    params['Lx'] = 1 * params['pitch']
    params['Ly'] = params['Lx']
    dx = params['Lx'] # grid resolution along x
    dy = params['Ly'] # grid resolution along x
    xa = np.linspace(0, pixelsX - 1, pixelsX) * dx 
    xa = xa - np.mean(xa) # center x axis at zero
    ya = np.linspace(0, pixelsY - 1, pixelsY) * dy 
    ya = ya - np.mean(ya) # center y axis at zero
    [y_mesh, x_mesh] = np.meshgrid(ya, xa, indexing='ij')       # 注意坐标系
    params['x_mesh'] = torch.tensor(x_mesh, device=params['device'])
    params['y_mesh'] = torch.tensor(y_mesh, device=params['device'])

    # Wavelengths
    lam0 = params['nanometers'] * torch.tensor(np.tile(lambda_base, params['arraySize']), dtype=torch.float32, device=params['device'])  # [606,511,462,606,511,462,...]
    lam0 = lam0.unsqueeze(1).unsqueeze(2)
    lam0 = lam0.repeat(1, pixelsX, pixelsY)
    params['lam0'] = lam0   # (27,1429,1429)

    # Field angle 
    theta_base = args.theta_base
    phi_base = args.phi_base

    theta_base_tensor = torch.tensor(np.repeat(theta_base, 3), dtype=torch.float32, device=params['device'])
    theta = params['degrees'] * theta_base_tensor
    theta = theta.unsqueeze(1).unsqueeze(2)  
    theta = theta.repeat(1, pixelsX, pixelsY)  
    params['theta'] = theta

    phi_tensor = torch.tensor(np.repeat(phi_base, 3), dtype=torch.float32, device=params['device'])
    phi = params['degrees'] * phi_tensor
    phi = phi.unsqueeze(1).unsqueeze(2)  
    phi = phi.repeat(1, pixelsX, pixelsY) 
    params['phi'] = phi

    # Propagation parameters
    params['propagator'] = make_propagator(params, params['v'])
    params['input'] = define_input_fields(params)

    # Metasurface proxy phase model
    params['phase_to_structure_coeffs'] = [-0.1484, 0.6809, 0.2923]
    params['structure_to_phase_coeffs'] = [6.051, -0.02033, 2.26, 1.371E-5, -0.002947, 0.797]

    params['phase_type'] = args.phase_type
    params['cubic_alpha'] = args.alpha
    params['s1'] = args.s1
    params['s2'] = args.s2
    params['lb'] = args.lb
    params['ub'] = args.ub
    params['norm_weight'] = args.norm_weight
    params['spatial_weight'] = args.spatial_weight
    params['loss_mode'] = args.loss_mode

    return params


@torch.no_grad()
def define_input_fields(params):
    # Define the cartesian cross section
    input_fields = torch.zeros(size=params['x_mesh'].shape, device=params['device'])
    n = input_fields.size(1)
    light = torch.tensor([[0, 1, 0]], device=params['device'])
    light = light.T * light
    input_fields[n//2-1:n//2+2, n//2-1:n//2+2] = light
    return input_fields.unsqueeze(0)


def duty_cycle_from_phase(phase, params):
    phase = phase / (2 * np.pi)
    p = params['phase_to_structure_coeffs']
    return p[0] * phase ** 2 + p[1] * phase + p[2]


def phase_from_duty_and_lambda(duty, params):
    p = params['structure_to_phase_coeffs']
    # lam = params['lam0'] / params['nanometers']
    # phase = p[0] + p[1]*lam + p[2]*duty + p[3]*lam**2 + p[4]*lam*duty + p[5]*duty**2

    lam = params['lam0'] / params['nanometers']
    duty = duty.repeat(3,1,1)
    phase = p[0] + p[1]*lam + p[2]*duty + p[3]*lam**2 + p[4]*lam*duty + p[5]*duty**2

    return phase * 2 * np.pi


def metasurface_phase_generator(fs, params):
    x_mesh = params['x_mesh'].unsqueeze(0)
    y_mesh = params['y_mesh'].unsqueeze(0)
    fs_tensor = fs.unsqueeze(1).unsqueeze(2)
    fs_tensor = fs_tensor.repeat(1, x_mesh.size(1), x_mesh.size(2))

    # Design for nominal wavelength
    if (params['phase_type'] == 'hyperboloid') or (params['phase_type'] == 'hyperboloid_learn'):
        phase_def = 2 * np.pi / params['wavelength_hyperboloid'] * (fs_tensor - torch.sqrt(x_mesh**2 + y_mesh**2 + fs_tensor**2))
    elif (params['phase_type'] == 'cubic') or (params['phase_type'] == 'cubic_learn'): 
        phase_def = 2 * np.pi / params['wavelength_nominal'] * (fs_tensor - torch.sqrt(x_mesh**2 + y_mesh**2 + fs_tensor**2)) \
            + params['cubic_alpha'] / (params['pixels_aperture'] * params['Lx'] / 2.0)**3 * (x_mesh**3 + y_mesh**3)
    elif params['phase_type'] == 'log_asphere':
        r_phase = torch.sqrt(x_mesh ** 2 + y_mesh ** 2)
        R = params['pixels_aperture'] * params['Lx'] / 2.0
        quo = (params['s2'] - params['s1']) / R**2
        quo_large = params['s1'] + quo * r_phase**2
        term1 = np.pi / params['wavelength_nominal'] / quo
        term2 = torch.log(2 * quo * (torch.sqrt(r_phase**2 + quo_large**2) + quo_large) + 1) - np.log(4*quo*params['s1'] + 1)
        phase_def = (-term1 * term2).repeat(9,1,1)
    elif params['phase_type'] == 'shifted_axicon':
        pass

    phase_def = phase_def % (2 * np.pi)       # NC使用tf.math.floormod，E2EMLF使用torch.fmod
    duty = duty_cycle_from_phase(phase_def, params)
    phase_def = phase_from_duty_and_lambda(duty, params)
    mask = ((x_mesh ** 2 + y_mesh ** 2) < (params['pixels_aperture'] * params['Lx'] / 2.0) ** 2)
    phase_def = phase_def * mask
    return phase_def


def define_metasurface(fs, params):
    phase_def = metasurface_phase_generator(fs, params)
    # phase_def = phase_def.to(torch.complex64)
    amp = (params['x_mesh'] ** 2 + params['y_mesh'] ** 2) < (params['pixels_aperture'] * params['Lx'] / 2.0) ** 2
    I = 1.0 / torch.sum(amp)
    E_amp = torch.sqrt(I)
    return amp * E_amp * torch.exp(1j * phase_def).to(torch.complex64)


@torch.no_grad()
def make_propagator(params, distance):

    batchSize = params['batchSize']
    pixelsX = params['pixelsX']

    # Propagator definition
    k = 2 * np.pi / params['lam0'][:, 0, 0]
    k = k.unsqueeze(1).unsqueeze(2)
    samp = params['upsample'] * pixelsX
    k = torch.tile(k, (1, 2 * samp - 1, 2 * samp - 1)).type(torch.complex64)     # (27,2857,2857)
    k_xlist_pos = 2 * np.pi * torch.linspace(0, 1 / (2 *  params['Lx'] / params['upsample']), samp, device=params['device'])
    front = k_xlist_pos[-(samp - 1):]
    front = -torch.flip(front, [0])
    k_xlist = torch.cat((front, k_xlist_pos), dim=0)      
    k_x = torch.kron(k_xlist, torch.ones((2 * samp - 1, 1), device=params['device']))
    k_x = k_x.unsqueeze(0)
    k_y = k_x.transpose(1, 2)
    k_x = k_x.repeat(batchSize, 1, 1)       # (27,2857,28557)
    k_y = k_y.repeat(batchSize, 1, 1)
    k_z_arg = torch.square(k) - (torch.square(k_x) + torch.square(k_y))
    k_z = torch.sqrt(k_z_arg)           # 里面为有什么有虚数，只有462波长时才不为虚数，去看NC的值会不会这样(答：也会)

    theta = params['theta'][:, 0, 0]
    theta = theta.unsqueeze(1).unsqueeze(2) 
    x0 = torch.tan(theta) * params['f']
    x0 = x0.repeat(1, 2 * samp - 1, 2 * samp - 1)  
    # x0 = x0.to(torch.complex64) 

    phi = params['phi'][:, 0, 0]
    phi = phi.unsqueeze(1).unsqueeze(2)
    y0 = torch.tan(phi) * params['f']
    y0 = y0.repeat(1, 2 * samp - 1, 2 * samp - 1)  
    # y0 = y0.to(torch.complex64)

    # propagator_arg = 1j * (k_z * params['v'] + k_x * x0 + k_y * y0)     # TODO 后面这两项是导致乱跑的原因
    # propagator_arg = 1j * (k_z * params['f'])       # Airy斑更小
   
    # propagator_arg = 1j * (k_z * distance)       # TODO: 使用这个看起来正确了很多
    propagator_arg = 1j * (k_z * distance + k_x * x0 + k_y * y0)
    propagator = torch.exp(propagator_arg)

    return propagator


def propTF(u1, L, wavelength, z, params):
    B, N, N = u1.shape
    dx = L / (N-1)
    k = 2 * np.pi / wavelength

    # TF
    # fx = torch.linspace(-1/(2*dx), 1/(2*dx), N)
    # FX, FY = torch.meshgrid(fx, fx)
    # H = torch.exp(-1j * torch.pi * wavelength * z * (FX**2 + FY**2))
    # H = torch.fft.fftshift(H)
  
    # IR
    x = torch.linspace(-L/2, L/2, N, device=params['device'])
    X, Y = torch.meshgrid(x,x)
    h = 1 / (1j * wavelength * z) * torch.exp(1j * k / (2*z) * (X**2+Y**2))
    H = torch.fft.fft2(torch.fft.fftshift(h)) * dx**2

    u1 = torch.fft.fft2(torch.fft.fftshift(u1))
    u2 = H * u1
    u2 = torch.fft.ifftshift(torch.fft.ifft2(u2))
    return u2


@torch.no_grad()
def propagate_first(field, distance, params):
    B, n, n = field.shape
    field = F.pad(field, ((n-1)//2, n-1-(n-1)//2, (n-1)//2, n-1-(n-1)//2)) 
    L = params['pitch'] * (2*n-2)
    wavelength = params['lam0'][:,0:1,0:1]
    theta = params['theta'][:, 0:1, 0:1]
    phi = params['phi'][:, 0:1, 0:1]
    z = distance
    out = propTF(field, L, wavelength, z, params)
    out = out[:, (n-1)//2:-(n-1)//2, (n-1)//2:-(n-1)//2]
    phase_def = 2 * torch.pi / wavelength * (torch.sin(theta) * params['x_mesh'] + torch.sin(phi) * params['y_mesh'])
    return out * torch.exp(1j * phase_def).to(torch.complex64)


def propagate_second(field, params):
    # Field has dimensions of (batchSize, pixelsX, pixelsY)
    # Each element corresponds to the zero order planewave component on the output
    propagator = params['propagator']

    # Zero pad `field` to be a stack of 2n-1 x 2n-1 matrices
    _, _, n = field.shape
    n = n * params['upsample']
    field_real = field.real
    field_imag = field.imag
    field_real = F.interpolate(field_real.unsqueeze(0), size=(n,n)).squeeze(0)
    field_imag = F.interpolate(field_imag.unsqueeze(0), size=(n,n)).squeeze(0)
    field = torch.view_as_complex(torch.stack([field_real, field_imag], dim=-1))
    field = F.pad(field, ((n-1)//2, n-1-(n-1)//2, (n-1)//2, n-1-(n-1)//2))    

    # field_freq = torch.fft.fftshift(torch.fft.fft2(field))
    # field_filtered = torch.fft.ifftshift(field_freq * propagator)
    # out = torch.fft.ifft2(field_filtered)

    propagator = torch.fft.fftshift(propagator)
    field_freq = torch.fft.fft2(torch.fft.fftshift(field))
    out = torch.fft.ifftshift(torch.fft.ifft2(field_freq * propagator))

    # Crop back down to n x n matrices
    out = out[:, (n-1)//2:-(n-1)//2, (n-1)//2:-(n-1)//2]    # (27,1429,1429)
    return out


def compute_intensity_at_sensor(metasurface_func, distance, params):
    # first propagate: input field -> metasurface
    field_ahead_meta = propagate_first(params['input'], distance, params)

    # sacle for upsampling
    # _, _, n = metasurface_func.shape
    # field_real = field_ahead_meta.real
    # field_imag = field_ahead_meta.imag
    # field_real = F.interpolate(field_real.unsqueeze(0), size=(n,n)).squeeze(0)
    # field_imag = F.interpolate(field_imag.unsqueeze(0), size=(n,n)).squeeze(0)
    # field_ahead_meta = torch.view_as_complex(torch.stack([field_real, field_imag], dim=-1))

    # second propagate: metasurface -> sensor
    coherent_psf = propagate_second(field_ahead_meta * metasurface_func, params)
    return torch.abs(coherent_psf) ** 2


def calculate_psf(intensity, params):
    aperture = params['pixels_aperture']
    sensor_pixel = params['sensor_pixel']
    magnification = params['magnification']
    period = params['Lx']

    # Determine PSF shape after optical magnification
    mag_width = int(np.round(aperture * period * magnification / sensor_pixel)) # mag8.1--810
    mag_intensity = torch.nn.functional.interpolate(intensity.unsqueeze(0), \
                        size=(mag_width, mag_width), mode='bilinear', align_corners=False).squeeze(0)

    # Maintain same energy as before optical magnification
    denom = torch.sum(mag_intensity, dim=[1, 2], keepdim=True) 
    mag_intensity = mag_intensity * torch.sum(intensity, dim=[1, 2], keepdim=True) / denom

    # Crop to sensor dimensions
    sensor_psf = mag_intensity
    sensor_psf = torch.clamp(sensor_psf, 0.0, 1.0)
  
    if params['normalize_psf']:
        sensor_psf_sum = torch.sum(sensor_psf, dim=(1,2), keepdim=True)
        sensor_psf = sensor_psf / sensor_psf_sum

    return sensor_psf


def test_phase(fs, params):
    x_mesh = torch.tensor(params['x_mesh']).unsqueeze(0)
    y_mesh = torch.tensor(params['y_mesh']).unsqueeze(0)
    fs_tensor = fs.unsqueeze(1).unsqueeze(2).repeat(3,1,1)
    phase_def = -2 * np.pi / params['lam0'] *  ((x_mesh ** 2 + y_mesh ** 2) / (2 * fs_tensor))
    phase_def = phase_def % (2 * np.pi)
    mask = ((x_mesh ** 2 + y_mesh ** 2) < (params['pixels_aperture'] * params['Lx'] / 2.0) ** 2)
    return torch.exp(1j * phase_def) * mask


## TODO: rotate_psfs
def get_psfs(fs, distance, params):
    metasurface_func = define_metasurface(fs, params)
    # metasurface_func = test_phase(fs, params)
    intensity = compute_intensity_at_sensor(metasurface_func, distance, params)
    psf = calculate_psf(intensity, params)
    return psf














