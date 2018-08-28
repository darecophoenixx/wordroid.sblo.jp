GaussianKernel layer
====
![](http://yunopon.sakura.ne.jp/sblo_files/wordroid/image/demo01_01.png)

![](http://yunopon.sakura.ne.jp/sblo_files/wordroid/image/demo03_02.png)

Do you need a strong classifier machine with neural network? Try this.  
This layer is implemented using Gaussian kernel.  
(see \_\_init\_\_.py)

## Requirement
Keras

## Demo
### [demo01](demo/demo01.ipynb)
simple example  
GaussianKernel (fixed gamma) -> Dense  
moon data

<img src="http://yunopon.sakura.ne.jp/sblo_files/wordroid/image/demo101_01.png" width="320px">

### [demo01_2](demo/demo01_2.ipynb)
simple example  
GaussianKernel (fixed gamma) -> Dense  
moon data (fat)

<img src="http://yunopon.sakura.ne.jp/sblo_files/wordroid/image/demo101_2.png" width="320px">

### [demo02](demo/demo02.ipynb)
simple example  
GaussianKernel (kernel_gamma='auto') -> Dense  
moon data

### [demo03](demo/demo03.ipynb)
GaussianKernel2 example  
<img src="http://yunopon.sakura.ne.jp/sblo_files/wordroid/image/demo103_01.png" width="320px">

### [demo04](demo/demo04.ipynb)
big kernel_gamma \(small Sigma\) example  
<img src="http://yunopon.sakura.ne.jp/sblo_files/wordroid/image/demo104_01.png" width="320px">

### [demo05](demo/demo05.ipynb)
GaussianKernel3 example  

### [demo_digit_01](demo/demo_digit_01.ipynb)
scikit-learn digits dataset example  
```python
num_lm = 100
GaussianKernel(num_lm, 64, kernel_gamma=1./(2.*64*0.1), weights=[init_wgt], name='gkernel1')
Dense(10, activation='softmax')
```
<img src="http://yunopon.sakura.ne.jp/sblo_files/wordroid/image/demo_digit_101.png" width="320px"> <img src="http://yunopon.sakura.ne.jp/sblo_files/wordroid/image/demo_digit_101_2.png" width="320px">


### [demo_digit_02](demo/demo_digit_02.ipynb)
scikit-learn digits dataset example  
using only 2 factors  
```python
num_lm = 100
GaussianKernel(num_lm, 64, kernel_gamma=1./(2.*64*0.1), weights=[init_wgt], name='gkernel1')
Dense(2)
Dense(10, activation='softmax')
```
<img src="http://yunopon.sakura.ne.jp/sblo_files/wordroid/image/demo_digit_102.png" width="320px">

### [demo_digit_03](demo/demo_digit_03.ipynb)
scikit-learn digits dataset example  
add conv layers  
```python
num_lm = 100

Conv2D(32, (2, 2), activation="relu")
Conv2D(32, (2, 2), activation="sigmoid")
MaxPooling2D(pool_size=(2,2))
GaussianKernel(num_lm, 288, kernel_gamma=1./(2.*288*0.1), name='gkernel1')
Dense(10, activation='softmax')
```
<img src="http://yunopon.sakura.ne.jp/sblo_files/wordroid/image/demo_digit_103_2.png" width="320px"> <img src="http://yunopon.sakura.ne.jp/sblo_files/wordroid/image/demo_digit_103.png" width="320px">

### [demo_digit_AE_01](demo/demo_digit_AE_01.ipynb)
scikit-learn digits dataset Auto-Encoder example  
Projection on high dimensional space(64 -> 200)
```python
num_lm = 200
GaussianKernel3(num_lm, 64, name='gkernel1', weights=[init_wgt, np.log(np.array([1./(2.*64*0.1)]))])
Dense(64, activation='sigmoid')
```
<img src="http://yunopon.sakura.ne.jp/sblo_files/wordroid/image/demo_digit_AE_01.png" width="320px">

## Usage
### GaussianKernel
```python
GaussianKernel(num_landmark=20, num_feature=5, kernel_gamma='auto')
```

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
this layer uses fixed landmarks  
train kernel_gamma  
see demo03
```python
GaussianKernel2(landmarks)
```

### GaussianKernel3
this layer train both landmarks and kernel_gamma  
see demo05
```python
GaussianKernel3(num_landmark, num_feature)
```
* num_landmark:  
number of landmark
* num_feature:  
depth of landmark  
equal to inputs.shape\[1]  


## Licence
Copyright (c) 2018 Norio Tamada  
Released under the MIT license  
https://github.com/darecophoenixx/wordroid.sblo.jp/blob/master/lib/keras_ex/gkernel/LICENSE.md


## Author
[darecophoenixx](https://github.com/darecophoenixx)
