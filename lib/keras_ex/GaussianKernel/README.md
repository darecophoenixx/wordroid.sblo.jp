GaussianKernel layer
====
![](http://yunopon.sakura.ne.jp/sblo_files/wordroid/image/demo01_01.png)

![](http://yunopon.sakura.ne.jp/sblo_files/wordroid/image/demo03_02.png)

Do you need a strong classifier machine with neural network? Try this.

(see \_\_init\_\_.py)

## Requirement
Keras

## Demo
### [demo/demo01.ipynb](demo/demo01.ipynb)  
simple stacked example  
GaussianKernel x2 -> Dense  
moon data

### [demo/demo01_1.ipynb](demo/demo01_1.ipynb)  
simple stacked example  
moon data  
GaussianKernel x2 -> Dense  
* GaussianKernel #1  
landmarks = 10
* GaussianKernel #2  
landmarks = 5  
fixed gamma

### [demo/demo02.ipynb](demo/demo02.ipynb)  
simple example  
GaussianKernel -> Dense  
moon data

### [demo/demo03.ipynb](demo/demo03.ipynb)  
fixed landmark example  
GaussianKernel(trainable=False) -> GaussianKernel(trainable=True) -> Dense  
trainable=False means landmarks fixed  
moon data

### demo04
GaussianKernel2 example  
![](http://yunopon.sakura.ne.jp/sblo_files/wordroid/image/demo04_01.png)

### demo_digit_01.ipynb
scikit-learn digits dataset example


## Usage
### GaussianKernel
    oup_gk1 = GaussianKernel(num_landmark=20, num_feature=5, kernel_gamma='auto')(inp)

* num_landmark:  
number of landmark
* num_feature:  
depth of landmark  
equal to inputs.shape\[1]  
* kernel_gamma:  
kernel parameter  
exp(-kernel_gamma * (x-landmark)\**2)  
if 'auto', use 1/(2 * d_mean\**2)  
d is distance between samples and landmark  
d_mean is mean of d  

### GaussianKernel2
    oup_gk1 = GaussianKernel2(landmarks)(inp)

this layer uses fixed landmark  
estimate kernel_gamma  
see demo04

## Licence
Copyright (c) 2018 Norio Tamada  
Released under the MIT license  
https://github.com/darecophoenixx/wordroid.sblo.jp/blob/master/lib/keras_ex/GaussianKernel/LICENSE.md


## Author
[darecophoenixx](https://github.com/darecophoenixx)
