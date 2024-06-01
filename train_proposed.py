import torch
import os
from args import parse_args
from utils import initialize_params, get_psfs
from dataset import MyDataset
from torch.utils.data import DataLoader
from conv_deconv import conv_fn, loss_fn, sensor_noise, feat_extract, wiener
from torch.utils.tensorboard import SummaryWriter

"""
只有FE的网络
"""

def train(args):
    params = initialize_params(args)
    params['phase_type'] = 'hyperboloid_learn'
    parameters_to_save = {}

    if (params['phase_type'] == 'hyperboloid' or params['phase_type'] == 'cubic' or params['phase_type'] == 'log_asphere'):
        fs = torch.tensor([2.5E-3 * 511 / 452]*9, device=params['device'])  
    elif (params['phase_type'] == 'hyperboloid_learn' or params['phase_type'] == 'cubic_learn'):
        fs = torch.tensor([2.5E-3]*9, device=params['device'])
        fs = torch.nn.Parameter(fs)
        phase_optimizer = torch.optim.Adam([fs], lr=args.phase_lr)
        phase_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(phase_optimizer, mode='min', factor=0.1, patience=5)
        parameters_to_save.update({'phase_params': fs, 'phase_optimizer': phase_optimizer.state_dict()})

    trainset = MyDataset(args.train_dir,810)
    evalset = MyDataset(args.eval_dir, 810)
    trainloader = DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True)
    evalloader = DataLoader(evalset, batch_size=args.train_batch_size, shuffle=False)

    nn = feat_extract(27).to(params['device'])
    nn_optimizer = torch.optim.Adam(nn.parameters(), lr=args.nn_lr)
    nn_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(nn_optimizer, mode='min', factor=0.1, patience=5)
    parameters_to_save.update({'nn_params': nn.state_dict(), 'nn_optimizer': nn_optimizer.state_dict()})

    if not os.path.exists(os.path.join(args.save_dir, args.exp_name)):
        os.makedirs(os.path.join(args.save_dir, args.exp_name))

    writer = SummaryWriter(os.path.join(args.log_dir, args.exp_name))
    train_stage_flag = 'optimize_nn'
    train_stage_iter = 0
    log_iter = 0
    best_eval_loss = 1E6

    for i in range(args.epochs):
        # Train
        loss_val_epoch = 0
        for j, img in enumerate(trainloader):
            train_stage_iter += 1
            log_iter += 1
            rand_depth = torch.rand(1, device=params['device']) * (params['ub'] - params['lb']) + params['lb']
            psf = get_psfs(fs, rand_depth, params)                  # (27,810,810)
            psf = psf.reshape(9, 3, psf.size(-2),psf.size(-1))      # (9,3,810,810)
            img = img.to(params['device'])
            blur = conv_fn(img, psf)                                # (b,9,3,810,810)
            blur = sensor_noise(blur, params['b_sqrt'])             # (b,9,3,810,810)
            deconv_result = wiener(blur, psf)                       # (b,9,3,810,810)
            deconv_result = deconv_result.reshape(blur.size(0), -1, blur.size(-2), blur.size(-1))     # (b,27,810,810)
            deconv_result = nn(deconv_result)                                                # (b,3,810,810)
            loss_val = loss_fn(deconv_result, img, params)
            loss_val_epoch += loss_val * img.size(0)
            loss_val.backward()
            writer.add_scalar("train loss vs iters", loss_val, log_iter)
            print("epoch:", i, " ---- ", "iters(/{}): ".format(len(trainloader)), j, " ---- ", "loss_val: ", loss_val.item())
            
            if (train_stage_flag == 'optimize_nn'):
                nn_optimizer.step()
                nn_optimizer.zero_grad()
                if (train_stage_iter % args.nn_iters == 0):
                    train_stage_flag = 'optimize_phase'
                    train_stage_iter = 0
            elif (train_stage_flag == 'optimize_phase'):
                phase_optimizer.step()
                phase_optimizer.zero_grad()
                if (train_stage_iter % args.phase_iters == 0):
                    train_stage_flag = 'optimize_nn'
                    train_stage_iter = 0

        loss_val_epoch /= len(trainset)
        nn_scheduler.step(loss_val_epoch)
        phase_scheduler.step(loss_val_epoch)
        writer.add_scalar('train loss vs epochs', loss_val_epoch, i)
        writer.add_scalar('phase learning rate', phase_optimizer.param_groups[0]['lr'], i)
        writer.add_scalar('nn learning rate', nn_optimizer.param_groups[0]['lr'], i)
        
        # Eval
        loss_val_epoch = 0
        if (i % args.log_freq == 0):
            for j, img in enumerate(evalloader):
                with torch.no_grad():
                    linear_depth = ((j+1) / len(evalloader)) * (params['ub'] - params['lb']) + params['lb']
                    psf = get_psfs(fs, linear_depth, params)                  # (27,810,810)
                    psf = psf.reshape(9, 3, psf.size(-2),psf.size(-1))      # (9,3,810,810)
                    img = img.to(params['device'])
                    blur = conv_fn(img, psf)                                # (b,9,3,810,810)
                    blur = sensor_noise(blur, params['b_sqrt'])             # (b,9,3,810,810)
                    deconv_result = wiener(blur, psf)                       # (b,9,3,810,810)
                    deconv_result = deconv_result.reshape(blur.size(0), -1, blur.size(-2), blur.size(-1))     # (b,27,810,810)
                    deconv_result = nn(deconv_result)                                                # (b,3,810,810)
                    loss_val = loss_fn(deconv_result, img, params)
                    loss_val_epoch += loss_val * img.size(0)
            loss_val_epoch /= len(evalset)
            writer.add_scalar('eval loss vs epochs', loss_val_epoch, i)
            if (loss_val_epoch < best_eval_loss):
                pt_save_path = os.path.join(args.save_dir, args.exp_name, 'paramters_best.pt'.format(i))
                torch.save(parameters_to_save, pt_save_path)
                best_eval_loss = loss_val_epoch

        if (i % args.save_freq == 0):
            pt_save_path = os.path.join(args.save_dir, args.exp_name, 'parameters.pt'.format(i))                    
            torch.save(parameters_to_save, pt_save_path)
        


def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()