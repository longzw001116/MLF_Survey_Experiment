# MLF_Survey_Experiment
The code for experiment part of our metasurface light field survey (Laser &amp; Optoelectronics Progress) (To be relased)


## Requirements
This code has been tested with Python 3.9.17 using Pytorch 2.2.1 running on Linux with an Nvidia A40 with 48GB RAM.

Install the following library packages to run this code:
```
Pytorch
Torchvision
lpips
Numpy
PIL
```


## Results
<center>

| Design | MSE | PSNR(dB) | SSIM | LPIPS |
| --------- | ---- | ---- | ---- | ---- |
| Hyperboloid_1 | 1.49e-2 | 18.89 | 0.52 | 0.59 |
| Hyperboloid_2 | 1.74e-2 | 18.01 | 0.19 | 0.64 |
| Hyperboloid_3 | 9.83e-3 | 20.54 | 0.39 | 0.65 |
| Cubic_86 $\pi$ | 1.01e-2 | 20.60 | 0.43 | 0.60 |
| Cubic_55 $\pi$  | 9.77e-3 | 20.71 | 0.40 | 0.61 |
| Log_asphere_1 | 9.92e-3 | 20.26 | 0.27 | 0.63 |
| Log_asphere_2 | 1.82e-3 | 17.86 | 0.44 | 0.64 |
| **E2EMLF** | **7.22e-3** | **22.36** | **0.65** | **0.52** |
| | | | |

</center>


+ See the following picture for our end-to-end design MLF (E2EMLF) with other methods:

![Result](/images/result.png)


+ Further imaging quality improvement is required. See defects in the zoom-in result as follows:

<center>

![Zoom_in_result](/images/zoom_in_result.png)

</center>


## Desciption
1. train_compare.py
    - Test for the comparision methods (hyperboloid, cubic, log-asphere, etc)

2. train_compare_v2.py
    - Test for the learned phase profile and 

3. train_proposed.py
    - Train with conv_deconv (only feature extraction module) (final choice)

4. train_proposed_v2.py
    - Train with conv_deconv_v2 (FE + feature propagation + decoder)

5. utils.py
    - Use D-FFT 【IR method】 for the first diffraction
    - D-FFT 【TF method】 and Angular Spectrum Method is undersampled in our setting, check Table 5.1 in the [Computational Fourier Optics](https://www.spiedigitallibrary.org/ebooks/TT/Computational-Fourier-Optics-A-MATLAB-Tutorial/eISBN-9780819482051/10.1117/3.858456#_=_)
    - Use ASM for the second diffraction

6. test_depth.pt
    - Depth arrange for test dataset, uniformaly sampled from DoF

7. other files (dataset.py, conv_deconv.py, conv_deconv_v2.py, note.txt, test.ipynb): obvious or unimportant

8. ./sh/: some training/testing scipts when tuning
    + exp1.sh: train with wiener + 【FE】
    + exp2_v2.sh: train with wiener + 【FE_FP_DE】
    + exp3_v2.sh: train with 【FE_FP_DE】, no deconvolution before NN
    + exp1_test.sh: test using train_compare_v2


## Data
Downloading data [INIRIA Holidays dataset](https://lear.inrialpes.fr/~jegou/data.php.html#holidays)

Total: 1491 images

Split: Train:Eval:Test $\approx$ 8:1:1 (1193:149:149)

Organize data as follow:
```
Project
|--data
|  |--train
|  |--val
|  |--test
```


## TODOs:
* [ ] Unify the pipeline
* [ ] Code for do-taper for Wiener
* [ ] Try advanced deconv+NN
* [ ] Try High degree-of-freedom phase profile 
<!-- * [ ] Account for oblique incident light -->
<!-- * [ ] Add Poision noise -->




