GaussianKernel
====
![](http://yunopon.sakura.ne.jp/sblo_files/wordroid/image/demo01_01.png)

![](http://yunopon.sakura.ne.jp/sblo_files/wordroid/image/demo01_02.png)

Overview


## Requirement
Keras

## Demo

## Usage
    oup_gk1 = GaussianKernel(num_lm, 2, kernel_gamma='auto', name='gkernel1')(oup)

* num_landmark:  
number of landmark
* num_features:  
depth of landmark  
equal to inputs.shape\[1]  
* kernel_gamma:  
kernel parameter  
if 'auto', use 1/(2 * d_mean**2)  
d is distance between samples and landmark  
d_mean is mean of d  

## Author
[darecophoenixx](https://github.com/darecophoenixx)
