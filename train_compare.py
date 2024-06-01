import torch
import lpips
import os
import numpy as np
from PIL import Image
from args import parse_args
from conv_deconv import conv_fn, wiener, cal_mse, cal_psnr, cal_ssim, cal_lpips, sensor_noise
from utils import initialize_params, get_psfs
from dataset import MyDataset
from torch.utils.data import DataLoader

"""
    用于测试传统方法/对比方法的性能
"""


def save_img(tensor, save_dir, idx):
    for i in range(tensor.size(0)):
        img_tensor = tensor[i]
        img_array = img_tensor.permute(1,2,0).detach().cpu().numpy()
        img = Image.fromarray((img_array * 255).astype(np.uint8))
        dir = os.path.join(save_dir, str(idx)+str(i)+".jpg")
        img.save(dir)


def train(args):
    params = initialize_params(args)  
    # params['phase_type'] = 'cubic'
    params['phase_type'] = 'log_asphere'

    if (params['phase_type'] == 'hyperboloid' or params['phase_type'] == 'cubic' or params['phase_type'] == 'log_asphere'):
        fs = torch.tensor([2.5E-3 * 511 / 452]*9, device=params['device'])  
        # fs = torch.tensor([2.5E-3]*9, device=params['device'])  
    elif (params['phase_type'] == 'hyperbolid_learn' or params['phase_type'] == 'cubic_learn'):
        fs = torch.tensor([2.5E-3]*9, device=params['device'])
        fs = torch.nn.Parameter(fs)
    # distance = 22.25E-3
    
    test_depth = torch.load('test_depth.pt').to(params['device'])                       
    test_dir = './data/test'
    test_data = MyDataset(test_dir, 810)
    test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

    mse_sum = 0
    psnr_sum = 0
    ssim_sum = 0
    lpips_sum = 0
    lpips_fn = lpips.LPIPS(net = 'vgg').to(params['device'])
    deconv_dir = os.path.join('./results', 'log_asphere_wiener_our')       # 注意修改这个值
    for idx, img in enumerate(test_loader):
        # img: (b,3,810,810)
        print(idx)
        psf = get_psfs(fs, test_depth[idx], params)                     # (27,810,810)
        psf = psf.reshape(9,3,psf.size(-2),psf.size(-1))                # (9,3,810,810)
        img = img.to(params['device'])
        blur = conv_fn(img, psf)                # (4,9,3,810,810)
        blur = sensor_noise(blur, params['b_sqrt'])
        deconv_img = wiener(blur, psf)          # (4,9,3,810,810)
        # deconv_img = blur        # ONLY for hyperboloid with no deconvolution algorithm
        # save_img(deconv_img[:,0,:,:,:], deconv_dir, idx)
        mse_sum += cal_mse(img, deconv_img[:,0,:,:,:]) * img.size(0)
        psnr_sum += cal_psnr(img, deconv_img[:,0,:,:,:]) * img.size(0)
        ssim_sum += cal_ssim(img, deconv_img[:,0,:,:,:]) * img.size(0)
        lpips_sum += cal_lpips(img, deconv_img[:,0,:,:,:], lpips_fn) * img.size(0)
    mse_mean = mse_sum / len(test_data)
    psnr_mean = psnr_sum / len(test_data)
    ssim_mean = ssim_sum / len(test_data)
    lpips_mean = lpips_sum / len(test_data)
    torch.set_printoptions(precision=8)
    print('mse: ', mse_mean)
    print('psnr: ', psnr_mean)
    print('ssim: ', ssim_mean)
    print('lpips: ', lpips_mean)


def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()