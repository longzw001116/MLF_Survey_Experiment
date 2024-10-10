import argparse

def parse_args():
    def str2bool(v):
        assert(v == 'True' or v == 'False')
        return v.lower() in ('true')
    
    parser = argparse.ArgumentParser(description='Parameter settings for end-to-end optimization of metalens array')
    parser.add_argument('--device', type=int, default=0, help='Index of gpu')
    parser.add_argument('--exp_name', type=str, default='exp0')

    # Data loading arguments
    parser.add_argument('--train_dir'  , type=str, default='./data/train', help='Directory of training input images')
    parser.add_argument('--eval_dir'   , type=str, default='./data/val', help='Directory of evaling input images')
    parser.add_argument('--test_dir'   , type=str, default='./data/test', help='Directory of testing input images')
    parser.add_argument('--train_batch_size', type=int, default=4)

    # Saving and logging arguments
    parser.add_argument('--save_dir'   , type=str, default='./save', help='Directory for saving ckpts and TensorBoard file')
    parser.add_argument('--save_freq'  , type=int, default=5, help='Interval to save model')
    parser.add_argument('--log_freq'   , type=int, default=1, help='Interval to write to TensorBoard')
    parser.add_argument('--log_dir'   , type=str, default='./log', help='***')
    # parser.add_argument('--ckpt_dir'   , type=str, default='None', help='Restoring from a checkpoint')
    parser.add_argument('--ckpt_dir'   , type=str, default='./save/exp1/parameters.pt', help='Restoring from a checkpoint')

    # Loss arguments
    parser.add_argument('--loss_mode'          , type=str, default='L1')
    parser.add_argument('--norm_weight'   , type=float, default=1.0)
    parser.add_argument('--spatial_weight'   , type=float, default=0.1)

    # Training arguments
    parser.add_argument('--epochs'     , type=int, default=100, help='Total number of optimization cycles')

    # Convolution arguments
    # parser.add_argument('--do_taper'     , type=str2bool, default=True, help='Activate edge tapering')
    parser.add_argument('--normalize_psf', type=str2bool, default=True, help='True to normalize PSF')
    parser.add_argument('--theta_base'   , type=str, default = '-10.0,0.0,10.0,-10.0,0.0,10.0,-10.0,0.0,10.0', help='Field angles')
    parser.add_argument('--phi_base'   , type=str, default = '10.0,10.0,10.0,0.0,0.0,0.0,-10.0,-10.0,-10.0', help='Field angles')

    parser.add_argument('--lb', type=float, default=19.40E-3, help='lower bound of depth range')
    parser.add_argument('--ub', type=float, default=26.78E-3, help='upper bound of depth range')

    # Metasurface arguments
    # parser.add_argument('--s1'               , type=float, default=2.25e-3, help='s1 parameter for log-asphere/saxicon')
    parser.add_argument('--s1'               , type=float, default=19.40e-3, help='s1 parameter for log-asphere/saxicon')
    # parser.add_argument('--s2'               , type=float, default=3.5e-3, help='s2 parameter for log-asphere/saxicon')
    parser.add_argument('--s2'               , type=float, default=26.78e-3, help='s2 parameter for log-asphere/saxicon')
    parser.add_argument('--phase_type'       , type=str  , default='hyperboloid', help='Type of phase profile')
    # parser.add_argument('--alpha'            , type=float, default=270.176968209, help='Alpha value for cubic (set to 86*pi)')
    parser.add_argument('--alpha'            , type=float, default=172.78759594, help='Alpha value for cubic (set to 55*pi)')

    # Sensor arguments
    parser.add_argument('--b_sqrt'   , type=float, default=0.05, help='Gaussian noise standard deviation')
    parser.add_argument('--mag'      , type=float, default=8.1, help='Relay system magnification factor (slightly less than 10x)')

    # Optimization arguments
    parser.add_argument('--phase_iters', type=int, default=30, help='Number of meta-optic optimization iterations per cycle')
    parser.add_argument('--phase_lr'   , type=float, default=5e-3, help='Meta-optic learning rate')
    parser.add_argument('--nn_iters'    , type=int, default=10, help='Number of deconvolution optimization iterations per cycle')
    parser.add_argument('--nn_lr'       , type=float, default=1e-4, help='Deconvolution learning rate')

    args = parser.parse_args()
    args.theta_base = [float(w) for w in args.theta_base.split(',')]
    args.phi_base = [float(w) for w in args.phi_base.split(',')]
    print(args)
    return args    