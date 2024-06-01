# Convolution and Fourier operations

import torch
import cv2
import torch.nn as nn
from torch.fft import fft2, ifft2
from torch.nn.functional import interpolate
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np


def psf2otf(psf):
    psf = torch.fft.fftshift(psf)
    otf = torch.fft.fft2(psf)
    return otf


def conv_fn(image, psf):
    """
    psf: (9,3,810,810)
    image: (b,3,810,810)
    blur: (b,9,3,810,810)
    """
    otf = psf2otf(psf).unsqueeze(0)                 # (1,9,3,810,810)
    image = image.unsqueeze(1)                      # (b,1,3,810,810)
    blur = ifft2(fft2(image) * otf)                 # (b,9,3,810,810)
    return torch.abs(blur)


def sensor_noise(input, std_gaussian=1E-5):
    gauss = torch.randn_like(input) * std_gaussian
    output = input + gauss
    return output


def cal_mse(img1, img2):
    mse_loss = torch.mean((img1 - img2) ** 2)
    return mse_loss


def cal_psnr(img1, img2, max_val=1.0):
    psnr_sum = 0
    for i in range(img1.size(0)):
        mse_value = cal_mse(img1[i,:,:,:], img2[i,:,:,:])
        psnr_sum += 20 * torch.log10(max_val / torch.sqrt(mse_value+1E-7))
    return psnr_sum / img1.size(0)


def cal_ssim(img1, img2):
    batch_size = img1.size(0)
    ssim_sum = 0
    for i in range(batch_size):
        img1_np = np.array(img1[i].detach().cpu())
        img2_np = np.array(img2[i].detach().cpu())
        ssim_value= compare_ssim(img1_np, img2_np, data_range=1, channel_axis=0)
        ssim_sum += ssim_value
    return ssim_sum / batch_size


def cal_lpips(img1, img2, lpips_fn):
    with torch.no_grad():
        lpips_value = torch.mean(lpips_fn(img1, img2))
    return lpips_value


def spatial_loss(deconv_img, gt_img, params):
    if params['loss_mode'] == 'L1': metric = torch.abs
    if params['loss_mode'] == 'L2': metric = torch.square

    def spatial_gradient(x):
        dh = x[:, :, :, :-1] - x[:, :, :, 1:]
        dv = x[:, :, :-1, :] - x[:, :, 1:, :]
        diag_down = x[:, :, 1:, 1:] - x[:, :, :-1, :-1]
        diag_up = x[:, :, :-1, 1:] - x[:, :, 1:, :-1]
        return [dh, dv, diag_down, diag_up]
    
    deconv_img_gradient_list = spatial_gradient(deconv_img)
    gt_img_gradient_list = spatial_gradient(gt_img)
    total_loss = 0
    for i in range(4):
        total_loss += torch.mean(metric(deconv_img_gradient_list[i] - gt_img_gradient_list[i]))
    total_loss = total_loss / 4
    return total_loss


def norm_loss(deconv_img, gt_img, params):
    if params['loss_mode'] == 'L1': metric = torch.abs
    if params['loss_mode'] == 'L2': metric = torch.square

    return torch.mean(metric(deconv_img - gt_img))


def loss_fn(deconv_img, gt_img, params):
    """
    L1 + L_grad
    """
    norm_loss_value = norm_loss(deconv_img, gt_img, params)
    spatial_loss_value = spatial_loss(deconv_img, gt_img, params)
    return params['norm_weight'] * norm_loss_value + params['spatial_weight'] * spatial_loss_value
 

def wiener(blur, psf, snr=300):
    """
    blur: (4,9,3,810,810)
    psf:  (9,3,810,810)
    out:  (4,9,3,810,810)
    """
    # TODO: do edge taper
    otf = psf2otf(psf)  
    wiener_filter = torch.conj(otf) / ((torch.abs(otf))**2 + 1/snr) 
    out = torch.abs(ifft2(wiener_filter * fft2(blur)))
    return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation=nn.LeakyReLU(), apply_instnorm=True, padding='same'):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.activation = activation
        self.apply_instnorm = apply_instnorm
        if apply_instnorm:
            self.instnorm = nn.InstanceNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        if self.apply_instnorm:
            x = self.instnorm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ConvTranspBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation=nn.LeakyReLU(), apply_instnorm=True):
        super(ConvTranspBlock, self).__init__()
        self.conv_transp = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.activation = activation
        self.apply_instnorm = apply_instnorm
    
    def forward(self, x):
        x = self.conv_transp(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class feat_extract(nn.Module):
    def __init__(self, input_ch):
        super(feat_extract, self).__init__()
        self.LReLU = nn.LeakyReLU()
        
        self.down_l0_1 = ConvBlock(input_ch, 15, 7, 1, self.LReLU, apply_instnorm=False)
        self.down_l0_2 = ConvBlock(15, 15, 7, 1, self.LReLU, apply_instnorm=False)
        
        self.down_l1_1 = ConvBlock(15, 30, 5, 2, self.LReLU, apply_instnorm=False, padding=2)
        self.down_l1_2 = ConvBlock(30, 30, 3, 1, self.LReLU, apply_instnorm=False)
        self.down_l1_3 = ConvBlock(30, 30, 3, 1, self.LReLU, apply_instnorm=False)
        
        self.down_l2_1 = ConvBlock(30, 60, 5, 2, self.LReLU, apply_instnorm=False, padding=2)
        self.down_l2_2 = ConvBlock(60, 60, 3, 1, self.LReLU, apply_instnorm=False)
        self.down_l2_3 = ConvBlock(60, 60, 3, 1, self.LReLU, apply_instnorm=False)
        
        # 4x
        self.conv_l2_k0 = ConvBlock(60, 60, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_l2_k1 = ConvBlock(60, 60, 3, 1, self.LReLU, apply_instnorm=False)    
        self.conv_l2_k2 = ConvBlock(120, 60, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_l2_k3 = ConvBlock(60, 60, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_l2_k4 = ConvBlock(60, 60, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_l2_k5 = ConvBlock(60, 60, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_transp1 = ConvTranspBlock(60, 30, 2, 2, self.LReLU, apply_instnorm=False)
        
        # 2x
        self.conv_l1_k0 = ConvBlock(30, 30, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_l1_k1 = ConvBlock(30, 30, 3, 1, self.LReLU, apply_instnorm=False)       
        self.conv_l1_k2 = ConvBlock(60, 30, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_l1_k3 = ConvBlock(30, 30, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_l1_k4 = ConvBlock(30, 30, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_l1_k5 = ConvBlock(30, 30, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_l1_k6 = ConvBlock(60, 30, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_l1_k7 = ConvBlock(30, 30, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_transp2 = ConvTranspBlock(30, 15, 2, 2, self.LReLU, apply_instnorm=False)
        
        # 1x
        self.conv_l0_k0 = ConvBlock(15, 15, 5, 1, self.LReLU, apply_instnorm=False)
        self.conv_l0_k1 = ConvBlock(15, 15, 5, 1, self.LReLU, apply_instnorm=False)
        self.conv_l0_k2 = ConvBlock(30, 15, 5, 1, self.LReLU, apply_instnorm=False)
        self.conv_l0_k3 = ConvBlock(15, 15, 5, 1, self.LReLU, apply_instnorm=False)
        self.conv_l0_k4 = ConvBlock(15, 15, 5, 1, self.LReLU, apply_instnorm=False)
        self.conv_l0_k5 = ConvBlock(15, 15, 5, 1, self.LReLU, apply_instnorm=False)
        self.conv_l0_k6 = ConvBlock(30, 15, 5, 1, self.LReLU, apply_instnorm=False)
        self.conv_l0_k7 = ConvBlock(15, 15, 5, 1, self.LReLU, apply_instnorm=False)
        self.conv_l0_k8 = ConvBlock(15, 15, 5, 1, self.LReLU, apply_instnorm=False)

        
        # DE 4x
        self.decoder_l2_k0 = ConvBlock(120, 120, 3, 1, self.LReLU, apply_instnorm=False)
        self.decoder_l2_k1 = ConvBlock(120, 120, 3, 1, self.LReLU, apply_instnorm=False)
        self.decoder_l2_k2 = ConvBlock(240, 120, 3, 1, self.LReLU, apply_instnorm=False)
        self.decoder_l2_k3 = ConvBlock(120, 120, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_transp_de_4x = ConvTranspBlock(120, 60, 2, 2, self.LReLU, apply_instnorm=False)

        # DE 2x
        self.decoder_l1_k0 = ConvBlock(60, 60, 5, 1, self.LReLU, apply_instnorm=False)
        self.decoder_l1_k1 = ConvBlock(60, 60, 5, 1, self.LReLU, apply_instnorm=False)
        self.decoder_l1_k2 = ConvBlock(120, 60, 5, 1, self.LReLU, apply_instnorm=False)
        self.decoder_l1_k3 = ConvBlock(60, 60, 5, 1, self.LReLU, apply_instnorm=False)
        self.conv_transp_de_2x = ConvTranspBlock(60, 30, 2, 2, self.LReLU, apply_instnorm=False)
        self.down_2x = ConvBlock(60, 60, 3, 2, self.LReLU, apply_instnorm=False, padding=2)

        # DE 1x
        self.decoder_l0_k0 = ConvBlock(15, 30, 3, 1, self.LReLU, apply_instnorm=False)
        self.decoder_l0_k1 = ConvBlock(30, 30, 3, 1, self.LReLU, apply_instnorm=False)
        self.decoder_l0_k2 = ConvBlock(60, 30, 3, 1, self.LReLU, apply_instnorm=False)
        self.decoder_l0_k3 = ConvBlock(30, 30, 3, 1, self.LReLU, apply_instnorm=False)
        self.decoder_l0_k4 = ConvBlock(30, 3, 3, 1, self.LReLU, apply_instnorm=False)
        self.down_1x = ConvBlock(30, 30, 5, 2, self.LReLU, apply_instnorm=False, padding=2)

    def forward(self, img, psf):
        down_l0 = self.down_l0_1(img)
        down_l0 = self.down_l0_2(down_l0)
        
        down_l1 = self.down_l1_1(down_l0)
        down_l1 = self.down_l1_2(down_l1)
        down_l1 = self.down_l1_3(down_l1)
        
        down_l2 = self.down_l2_1(down_l1)
        down_l2 = self.down_l2_2(down_l2)
        down_l2 = self.down_l2_3(down_l2)
        
        # 4x
        conv_l2_0 = self.conv_l2_k0(down_l2)
        conv_l2_1 = self.conv_l2_k1(conv_l2_0)  
        conv_l2_2 = self.conv_l2_k2(torch.cat([down_l2, conv_l2_1], dim=1))
        conv_l2_3 = self.conv_l2_k3(conv_l2_2)
        conv_l2_4 = self.conv_l2_k4(conv_l2_3)
        out_4x = self.conv_l2_k5(conv_l2_4)
        up1 = self.conv_transp1(out_4x)

        # 2x
        conv_l1_0 = self.conv_l1_k0(down_l1)
        conv_l1_1 = self.conv_l1_k1(conv_l1_0)
        conv_l1_2 = self.conv_l1_k2(torch.cat([down_l1, conv_l1_1], dim=1))
        conv_l1_3 = self.conv_l1_k3(conv_l1_2)
        conv_l1_4 = self.conv_l1_k4(conv_l1_3)
        conv_l1_5 = self.conv_l1_k5(conv_l1_4)
        if (up1.size(-1) != conv_l1_5.size(-1)):
            up1 = interpolate(up1, size=(conv_l1_5.size(-2), conv_l1_5.size(-1)))
        conv_l1_6 = self.conv_l1_k6(torch.cat([up1, conv_l1_5], dim=1))
        out_2x = self.conv_l1_k7(conv_l1_6)
        up2 = self.conv_transp2(conv_l1_5)
        
        # 1x
        conv_l0_0 = self.conv_l0_k0(down_l0)
        conv_l0_1 = self.conv_l0_k1(conv_l0_0)
        conv_l0_2 = self.conv_l0_k2(torch.cat([down_l0, conv_l0_1], dim=1))
        conv_l0_3 = self.conv_l0_k3(conv_l0_2)
        conv_l0_4 = self.conv_l0_k4(conv_l0_3)
        conv_l0_5 = self.conv_l0_k5(conv_l0_4)
        if (up2.size(-1) != conv_l0_5.size(-1)):
            up2 = interpolate(up2, size=(conv_l0_5.size(-2), conv_l0_5.size(-1)))
        conv_l0_6 = self.conv_l0_k6(torch.cat([up2, conv_l0_5], dim=1))
        conv_l0_7 = self.conv_l0_k7(conv_l0_6)
        out_1x = self.conv_l0_k8(conv_l0_7)
        
        psf_1x = torch.mean(psf.repeat(1,5,1,1), dim=0, keepdim=True)
        psf_1x = psf_1x / torch.sum(psf_1x, dim=(-2,-1), keepdim=True)
        out_1x = wiener(out_1x, psf_1x)

        psf_2x = interpolate(psf, size=(out_2x.size(-2), out_2x.size(-1)))
        psf_2x = torch.mean(psf_2x.repeat(1,10,1,1), dim=0, keepdim=True)
        psf_2x = psf_2x / torch.sum(psf_2x, dim=(-2,-1), keepdim=True)
        out_2x = wiener(out_2x, psf_2x)

        psf_4x = interpolate(psf, size=(out_4x.size(-2), out_4x.size(-1)))
        psf_4x = torch.mean(psf_4x.repeat(1,20,1,1), dim=0, keepdim=True)
        psf_4x = psf_4x / torch.sum(psf_4x, dim=(-2,-1), keepdim=True)
        out_4x = wiener(out_4x, psf_4x)

        # DE 1x
        decoder_l0_0 = self.decoder_l0_k0(out_1x)
        decoder_l0_1 = self.decoder_l0_k1(decoder_l0_0)
        decoder_down_1x = self.down_1x(decoder_l0_0)

        # DE 2x
        if (out_2x.size(-1) != decoder_down_1x.size(-1)):
            decoder_down_1x = interpolate(decoder_down_1x, size=(out_2x.size(-2), out_2x.size(-1)))
        decoder_l1_0 = self.decoder_l1_k0(torch.cat([out_2x, decoder_down_1x], dim=1))
        decoder_l1_1 = self.decoder_l1_k1(decoder_l1_0)
        decoder_down_2x = self.down_2x(decoder_l1_1)

        # DE 4x
        if (out_4x.size(-1) != decoder_down_2x.size(-1)):
            decoder_down_2x = interpolate(decoder_down_2x, size=(out_4x.size(-2), out_4x.size(-1)))
        decoder_l2_0 = self.decoder_l2_k0(torch.cat([out_4x, decoder_down_2x], dim=1))
        decoder_l2_1 = self.decoder_l2_k1(decoder_l2_0)
        decoder_l2_2 = self.decoder_l2_k2(torch.cat([decoder_l2_1, out_4x, decoder_down_2x], dim=1))
        decoder_l2_3 = self.decoder_l2_k3(decoder_l2_2)
        decoder_up_4x = self.conv_transp_de_4x(decoder_l2_3)

        # DE 2x
        if (decoder_up_4x.size(-1) != decoder_l1_1.size(-1)):
            decoder_up_4x = interpolate(decoder_up_4x, size=(decoder_l1_1.size(-1), decoder_l1_1.size(-1)))
        decoder_l1_2 = self.decoder_l1_k2(torch.cat([decoder_l1_1, decoder_up_4x], dim=1))
        decoder_l1_3 = self.decoder_l1_k3(decoder_l1_2)
        decoder_up_2x = self.conv_transp_de_2x(decoder_l1_3)

        # DE 1x
        if (decoder_l0_1.size(-1) != decoder_up_2x.size(-1)):
            decoder_up_2x = interpolate(decoder_up_2x, size=(decoder_l0_1.size(-1), decoder_l0_1.size(-1)))
        decoder_l0_2 = self.decoder_l0_k2(torch.cat([decoder_l0_1, decoder_up_2x], dim=1))
        decoder_l0_3 = self.decoder_l0_k3(decoder_l0_2)
        out = self.decoder_l0_k4(decoder_l0_3)

        return out
