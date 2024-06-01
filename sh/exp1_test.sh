#! /bin/bash

python train_compare_v2.py \
--exp_name          exp1_test \
--device            0 \
--train_dir         ./data/train \
--eval_dir          ./data/val \
--test_dir          ./data/test \
--ckpt_dir          ./save/exp1/parameters.pt \
--train_batch_size  10 \
--save_dir          ./save \
--save_freq         5 \
--log_freq          1 \
--loss_mode         L1 \
--norm_weight       1.0 \
--spatial_weight    0.1 \
--epochs            100 \
--normalize_psf     True \
--lb                19.40E-3 \
--ub                26.78E-3 \
--phase_type        hyperboloid_learn \
--b_sqrt            0.05 \
--mag               8.1 \
--phase_iters       10 \
--phase_lr          5E-3 \
--nn_iters          50 \
--nn_lr             1E-4