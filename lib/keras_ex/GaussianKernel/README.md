GaussianKernel
====
![](http://yunopon.sakura.ne.jp/sblo_files/wordroid/image/demo01_01.png)

![](http://yunopon.sakura.ne.jp/sblo_files/wordroid/image/demo01_02-a3333.png)

Overview


## Requirement
Keras

## Demo
[demo/demo01.ipynb](demo/demo01.ipynb)  
simple stacked example  
GaussianKernel x2 -> Dense

demo02  
simple example  
GaussianKernel -> Dense

demo03  
fixed landmark example  
GaussianKernel(trainable=False) -> GaussianKernel(trainable=True) -> Dense

## Usage
    oup_gk1 = GaussianKernel(num_landmark=20, num_features=5, kernel_gamma='auto')(inp)

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
