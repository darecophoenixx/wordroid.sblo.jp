no negative sampling  

NN_word = 200  
num_features = 3

(2019.6.3) use fit (not fit_generator)  
(2019.6.16) lr_scheduler  


```python
%matplotlib inline
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
```

    Using TensorFlow backend.
    /home/admin/miniconda3/envs/da03/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:516: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint8 = np.dtype([("qint8", np.int8, 1)])
    /home/admin/miniconda3/envs/da03/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:517: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
    /home/admin/miniconda3/envs/da03/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:518: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint16 = np.dtype([("qint16", np.int16, 1)])
    /home/admin/miniconda3/envs/da03/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:519: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
    /home/admin/miniconda3/envs/da03/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:520: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint32 = np.dtype([("qint32", np.int32, 1)])
    /home/admin/miniconda3/envs/da03/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:525: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      np_resource = np.dtype([("resource", np.ubyte, 1)])
    /home/admin/miniconda3/envs/da03/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:541: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint8 = np.dtype([("qint8", np.int8, 1)])
    /home/admin/miniconda3/envs/da03/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:542: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
    /home/admin/miniconda3/envs/da03/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:543: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint16 = np.dtype([("qint16", np.int16, 1)])
    /home/admin/miniconda3/envs/da03/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:544: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
    /home/admin/miniconda3/envs/da03/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:545: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint32 = np.dtype([("qint32", np.int32, 1)])
    /home/admin/miniconda3/envs/da03/lib/python3.6/site-packages/tensorboard/compat/tensorflow_stub/dtypes.py:550: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      np_resource = np.dtype([("resource", np.ubyte, 1)])



```python
import sys
sys.path.append('/home/admin/github/wordroid.sblo.jp/lib')
from feature_eng import lowcols
```


```python
import os.path
import sys
import re
import itertools
import csv
import datetime
import pickle
import random
from collections import defaultdict, Counter
import gc

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import seaborn as sns
import pandas as pd
import numpy as np
import scipy
import gensim
from sklearn.metrics import f1_score, classification_report, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
import gensim
from keras.preprocessing.sequence import skipgrams
import tensorflow as tf
```


```python
def hexbin(x, y, color, **kwargs):
    cmap = sns.light_palette(color, as_cmap=True)
    plt.hexbin(x, y, cmap=cmap, **kwargs)
def scatter(x, y, color, **kwargs):
    plt.scatter(x, y, marker='.')
```

### create sample data


```python
NN_word = 200
NN_sentence = 10000
NN_SEG = 7
```


```python
product_list = [ee+1 for ee in range(NN_word)]
user_list = [ee+1 for ee in range(NN_sentence)]
```


```python
a, _ = divmod(len(user_list), NN_SEG)
a
cls_user = [int(user_id / (a+1)) for user_id in range(1, 1+len(user_list))]
```


```python
a, _ = divmod(len(product_list), NN_SEG)
print(a)

cls_prod = [int(prod_id / (a+1)) for prod_id in range(1, 1+len(product_list))]
```

    28



```python
random.seed(0)

X_list = []

for ii in range(len(user_list)):
    cls = cls_user[ii]
    product_group = np.array(product_list)[np.array(cls_prod) == cls]
    nword = random.randint(5, 20)
    prods = random.sample(product_group.tolist(), nword)
    irow = np.zeros((1,NN_word))
    irow[0,np.array(prods)-1] = 1
    X_list.append(irow)

X = np.concatenate(X_list)
print(X.shape)
X
```

    (10000, 200)





    array([[0., 1., 1., ..., 0., 0., 0.],
           [0., 0., 1., ..., 0., 0., 0.],
           [1., 1., 1., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 1.],
           [0., 0., 0., ..., 1., 1., 0.],
           [0., 0., 0., ..., 1., 1., 1.]])




```python
X_df = pd.DataFrame(X, dtype=int)
X_df.index = ['r'+ee.astype('str') for ee in (np.arange(X_df.shape[0])+1)]
X_df.columns = ['c'+ee.astype('str') for ee in np.arange(X_df.shape[1])+1]
print(X_df.shape)
X_df.head()
```

    (10000, 200)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>c1</th>
      <th>c2</th>
      <th>c3</th>
      <th>c4</th>
      <th>c5</th>
      <th>c6</th>
      <th>c7</th>
      <th>c8</th>
      <th>c9</th>
      <th>c10</th>
      <th>...</th>
      <th>c191</th>
      <th>c192</th>
      <th>c193</th>
      <th>c194</th>
      <th>c195</th>
      <th>c196</th>
      <th>c197</th>
      <th>c198</th>
      <th>c199</th>
      <th>c200</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>r1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>r2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>r3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>r4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>r5</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 200 columns</p>
</div>




```python
X_df.values.shape
```




    (10000, 200)




```python
plt.figure(figsize=(15, 15))
plt.imshow(X_df.values.T)
```




    <matplotlib.image.AxesImage at 0x7f16a28d3c18>




![png](output_13_1.png)



```python

```


```python
wd2v = lowcols.WD2vec(X_df)
wd2v
```




    <feature_eng.lowcols.WD2vec at 0x7f16a0ccd320>




```python
num_features = 3

models = wd2v.make_model(num_user=X_df.shape[0], num_product=NN_word, num_features=num_features)
models['model_gk1'].summary()
models['model_user'].summary()
print('\n\n##################### model >>>')
model = models['model']
model.summary()
```

    WARNING:tensorflow:From /home/admin/miniconda3/envs/da03/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:517: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    WARNING:tensorflow:From /home/admin/miniconda3/envs/da03/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:74: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
    
    WARNING:tensorflow:From /home/admin/miniconda3/envs/da03/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:4138: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.
    
    WARNING:tensorflow:From /home/admin/miniconda3/envs/da03/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:174: The name tf.get_default_session is deprecated. Please use tf.compat.v1.get_default_session instead.
    
    WARNING:tensorflow:From /home/admin/miniconda3/envs/da03/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:181: The name tf.ConfigProto is deprecated. Please use tf.compat.v1.ConfigProto instead.
    
    WARNING:tensorflow:From /home/admin/miniconda3/envs/da03/lib/python3.6/site-packages/keras/optimizers.py:790: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.
    
    WARNING:tensorflow:From /home/admin/miniconda3/envs/da03/lib/python3.6/site-packages/tensorflow/python/ops/nn_impl.py:180: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_user (InputLayer)      (None, 1)                 0         
    _________________________________________________________________
    user_embedding (Embedding)   (None, 1, 3)              30000     
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 3)                 0         
    _________________________________________________________________
    gkernel1 (GaussianKernel3)   (None, 200)               601       
    =================================================================
    Total params: 30,601
    Trainable params: 30,601
    Non-trainable params: 0
    _________________________________________________________________
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_user (InputLayer)      (None, 1)                 0         
    _________________________________________________________________
    user_embedding (Embedding)   (None, 1, 3)              30000     
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 3)                 0         
    =================================================================
    Total params: 30,000
    Trainable params: 30,000
    Non-trainable params: 0
    _________________________________________________________________
    
    
    ##################### model >>>
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_user (InputLayer)      (None, 1)                 0         
    _________________________________________________________________
    user_embedding (Embedding)   (None, 1, 3)              30000     
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 3)                 0         
    _________________________________________________________________
    gkernel1 (GaussianKernel3)   (None, 200)               601       
    =================================================================
    Total params: 30,601
    Trainable params: 30,601
    Non-trainable params: 0
    _________________________________________________________________



```python
wd2v.models['model'].get_layer('user_embedding').get_weights()[0].shape
```




    (10000, 3)




```python
wgt_user = wd2v.get_wgt_byrow()
# wgt_user = model.get_layer('user_embedding').get_weights()[0]
print(wgt_user.shape)
df = pd.DataFrame(wgt_user[:,:5])
sns.set_context('paper')
g = sns.PairGrid(df, size=3.5)
g.map_diag(plt.hist, edgecolor="w")
g.map_lower(scatter)
g.map_upper(hexbin)
```

    (10000, 3)


    /home/admin/miniconda3/envs/da03/lib/python3.6/site-packages/seaborn/axisgrid.py:1241: UserWarning: The `size` paramter has been renamed to `height`; please update your code.
      warnings.warn(UserWarning(msg))





    <seaborn.axisgrid.PairGrid at 0x7f16a104c710>




![png](output_18_3.png)



```python
wgt_lm = wd2v.get_wgt_bycol()
print(wgt_lm.shape)
df = pd.DataFrame(wgt_lm[:,:5])
sns.set_context('paper')
g = sns.PairGrid(df, size=3.5)
g.map_diag(plt.hist, edgecolor="w")
g.map_lower(scatter)
g.map_upper(hexbin)
```

    (200, 3)





    <seaborn.axisgrid.PairGrid at 0x7f16a02be7b8>




![png](output_19_2.png)



```python
import keras
keras.__version__
```




    '2.2.4'



### 1


```python
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau

# def lr_schedule(epoch, lr, epochs=50, lr0=0.01, base=8):
#     b = 1 / np.log((epochs-1+np.exp(1)))
#     a = 1 / np.log((epoch+np.exp(1))) / (1-b) - b/(1-b)
#     lr = a*(1-1/base)*lr0 + lr0/base
#     print('Learning rate: ', lr)
#     return lr

# lr_scheduler = LearningRateScheduler(lr_schedule)
# callbacks = [lr_scheduler]

# hst = wd2v.train(epochs=50, batch_size=32, verbose=2,
#            callbacks=callbacks)
```


```python
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau

# def lr_schedule(epoch):
#     def reduce(epoch, lr):
#         if divmod(epoch,4)[1] == 3:
#             lr *= (1/8)
#         elif divmod(epoch,4)[1] == 2:
#             lr *= (1/4)
#         elif divmod(epoch,4)[1] == 1:
#             lr *= (1/2)
#         elif divmod(epoch,4)[1] == 0:
#             pass
#         return lr
    
#     lr0 = 0.01
#     epoch1 = 8
#     epoch2 = 8
#     epoch3 = 8
#     epoch4 = 8
    
#     if epoch<epoch1:
#         lr = lr0
#         #lr = reduce(epoch, lr)
#     elif epoch<epoch1+epoch2:
#         lr = lr0/2
#         #lr = reduce(epoch, lr)
#     elif epoch<epoch1+epoch2+epoch3:
#         lr = lr0/4
#         #lr = reduce(epoch, lr)
#     elif epoch<epoch1+epoch2+epoch3+epoch4:
#         lr = lr0/8
#         #lr = reduce(epoch, lr)
#     else:
#         lr = lr0/16
    
#     print('Learning rate: ', lr)
#     return lr

# lr_scheduler = LearningRateScheduler(lr_schedule)
# callbacks = [lr_scheduler]

# hst = wd2v.train(epochs=32, batch_size=32, verbose=2,
#            callbacks=callbacks)
```


```python
hst = wd2v.train(epochs=50, batch_size=32, verbose=2)
```

    Epoch 1/50
    Learning rate:  0.01
     - 4s - loss: 0.3269 - acc: 0.9148
    Epoch 2/50
    Learning rate:  0.007204268570481045
     - 2s - loss: 0.2523 - acc: 0.9376
    Epoch 3/50
    Learning rate:  0.005834135746073483
     - 2s - loss: 0.2378 - acc: 0.9377
    Epoch 4/50
    Learning rate:  0.0050013254202696785
     - 2s - loss: 0.2289 - acc: 0.9377
    Epoch 5/50
    Learning rate:  0.004432620519199698
     - 2s - loss: 0.2193 - acc: 0.9377
    Epoch 6/50
    Learning rate:  0.004014837552604439
     - 2s - loss: 0.2078 - acc: 0.9377
    Epoch 7/50
    Learning rate:  0.00369216769989574
     - 2s - loss: 0.1936 - acc: 0.9377
    Epoch 8/50
    Learning rate:  0.003433714942906377
     - 2s - loss: 0.1760 - acc: 0.9377
    Epoch 9/50
    Learning rate:  0.003220895793898085
     - 2s - loss: 0.1553 - acc: 0.9383
    Epoch 10/50
    Learning rate:  0.0030418116587033316
     - 2s - loss: 0.1358 - acc: 0.9403
    Epoch 11/50
    Learning rate:  0.0028884628098809063
     - 2s - loss: 0.1183 - acc: 0.9433
    Epoch 12/50
    Learning rate:  0.0027552555781387256
     - 2s - loss: 0.1037 - acc: 0.9462
    Epoch 13/50
    Learning rate:  0.0026381505830680055
     - 2s - loss: 0.0956 - acc: 0.9480
    Epoch 14/50
    Learning rate:  0.002534151184934775
     - 2s - loss: 0.0918 - acc: 0.9492
    Epoch 15/50
    Learning rate:  0.0024409828962551417
     - 2s - loss: 0.0901 - acc: 0.9499
    Epoch 16/50
    Learning rate:  0.0023568851052123392
     - 2s - loss: 0.0891 - acc: 0.9507
    Epoch 17/50
    Learning rate:  0.0022804715278789056
     - 2s - loss: 0.0883 - acc: 0.9512
    Epoch 18/50
    Learning rate:  0.002210634178153207
     - 2s - loss: 0.0876 - acc: 0.9517
    Epoch 19/50
    Learning rate:  0.0021464757216000914
     - 2s - loss: 0.0871 - acc: 0.9521
    Epoch 20/50
    Learning rate:  0.002087260830499979
     - 2s - loss: 0.0866 - acc: 0.9525
    Epoch 21/50
    Learning rate:  0.0020323805553448624
     - 2s - loss: 0.0862 - acc: 0.9527
    Epoch 22/50
    Learning rate:  0.0019813257979569548
     - 2s - loss: 0.0860 - acc: 0.9528
    Epoch 23/50
    Learning rate:  0.0019336672671061345
     - 2s - loss: 0.0858 - acc: 0.9531
    Epoch 24/50
    Learning rate:  0.0018890401285365371
     - 2s - loss: 0.0856 - acc: 0.9531
    Epoch 25/50
    Learning rate:  0.0018471321061274935
     - 2s - loss: 0.0854 - acc: 0.9533
    Epoch 26/50
    Learning rate:  0.0018076741552439314
     - 2s - loss: 0.0853 - acc: 0.9533
    Epoch 27/50
    Learning rate:  0.0017704330774042942
     - 2s - loss: 0.0851 - acc: 0.9535
    Epoch 28/50
    Learning rate:  0.001735205617116534
     - 2s - loss: 0.0850 - acc: 0.9535
    Epoch 29/50
    Learning rate:  0.001701813702416052
     - 2s - loss: 0.0849 - acc: 0.9536
    Epoch 30/50
    Learning rate:  0.001670100576643398
     - 2s - loss: 0.0848 - acc: 0.9536
    Epoch 31/50
    Learning rate:  0.0016399276310837452
     - 2s - loss: 0.0847 - acc: 0.9537
    Epoch 32/50
    Learning rate:  0.001611171793445072
     - 2s - loss: 0.0846 - acc: 0.9537
    Epoch 33/50
    Learning rate:  0.0015837233606549829
     - 2s - loss: 0.0846 - acc: 0.9538
    Epoch 34/50
    Learning rate:  0.0015574841894612367
     - 2s - loss: 0.0845 - acc: 0.9538
    Epoch 35/50
    Learning rate:  0.0015323661771647283
     - 2s - loss: 0.0844 - acc: 0.9539
    Epoch 36/50
    Learning rate:  0.001508289979143041
     - 2s - loss: 0.0844 - acc: 0.9538
    Epoch 37/50
    Learning rate:  0.0014851839208119217
     - 2s - loss: 0.0843 - acc: 0.9541
    Epoch 38/50
    Learning rate:  0.0014629830701671676
     - 2s - loss: 0.0842 - acc: 0.9540
    Epoch 39/50
    Learning rate:  0.0014416284436659722
     - 2s - loss: 0.0842 - acc: 0.9541
    Epoch 40/50
    Learning rate:  0.0014210663233967624
     - 2s - loss: 0.0841 - acc: 0.9540
    Epoch 41/50
    Learning rate:  0.0014012476675849387
     - 2s - loss: 0.0841 - acc: 0.9542
    Epoch 42/50
    Learning rate:  0.0013821275997388009
     - 2s - loss: 0.0840 - acc: 0.9541
    Epoch 43/50
    Learning rate:  0.001363664964343722
     - 2s - loss: 0.0840 - acc: 0.9542
    Epoch 44/50
    Learning rate:  0.0013458219391061501
     - 2s - loss: 0.0839 - acc: 0.9542
    Epoch 45/50
    Learning rate:  0.0013285636954414256
     - 2s - loss: 0.0839 - acc: 0.9543
    Epoch 46/50
    Learning rate:  0.0013118581002746184
     - 2s - loss: 0.0839 - acc: 0.9542
    Epoch 47/50
    Learning rate:  0.0012956754533465648
     - 2s - loss: 0.0838 - acc: 0.9544
    Epoch 48/50
    Learning rate:  0.0012799882551385894
     - 2s - loss: 0.0838 - acc: 0.9543
    Epoch 49/50
    Learning rate:  0.0012647710012886498
     - 2s - loss: 0.0837 - acc: 0.9545
    Epoch 50/50
    Learning rate:  0.00125
     - 2s - loss: 0.0837 - acc: 0.9543



```python
hst_history = hst.history
```


```python
fig, ax = plt.subplots(1, 3, figsize=(20,5))
ax[0].set_title('loss')
ax[0].plot(list(range(len(hst_history["loss"]))), hst_history["loss"], label="Train loss")
ax[1].set_title('acc')
ax[1].plot(list(range(len(hst_history["loss"]))), hst_history["acc"], label="accuracy")
ax[2].set_title('learning rate')
ax[2].plot(list(range(len(hst_history["loss"]))), hst_history["lr"], label="learning rate")
ax[0].legend()
ax[1].legend()
ax[2].legend()
```




    <matplotlib.legend.Legend at 0x7f1698143be0>




![png](output_26_1.png)



```python
wgt_prod = wd2v.get_wgt_bycol()
print(wgt_prod.shape)
df = pd.DataFrame(wgt_prod[:,:5])
sns.set_context('paper')
g = sns.PairGrid(df, size=3.5)
g.map_diag(plt.hist, edgecolor="w")
g.map_lower(scatter)
g.map_upper(hexbin)
```

    (200, 3)


    /home/admin/miniconda3/envs/da03/lib/python3.6/site-packages/seaborn/axisgrid.py:1241: UserWarning: The `size` paramter has been renamed to `height`; please update your code.
      warnings.warn(UserWarning(msg))





    <seaborn.axisgrid.PairGrid at 0x7f16981244a8>




![png](output_27_3.png)



```python
wgt_prod = wd2v.get_wgt_bycol()
print(wgt_prod.shape)
df = pd.DataFrame(wgt_prod[:,:5])
df['cls'] = ['c'+str(ii) for ii in cls_prod]
sns.pairplot(df, markers='o', hue='cls', size=3.5, diag_kind='hist')
```

    (200, 3)


    /home/admin/miniconda3/envs/da03/lib/python3.6/site-packages/seaborn/axisgrid.py:2065: UserWarning: The `size` parameter has been renamed to `height`; pleaes update your code.
      warnings.warn(msg, UserWarning)





    <seaborn.axisgrid.PairGrid at 0x7f16743bf320>




![png](output_28_3.png)



```python
wgt_user = wd2v.get_wgt_byrow()
print(wgt_user.shape)
df = pd.DataFrame(wgt_user[:,:5])
sns.set_context('paper')
g = sns.PairGrid(df, size=3.5)
g.map_diag(plt.hist, edgecolor="w")
g.map_lower(scatter)
g.map_upper(hexbin)
```

    (10000, 3)





    <seaborn.axisgrid.PairGrid at 0x7f166779e860>




![png](output_29_2.png)



```python
wgt_user = wd2v.get_wgt_byrow()
print(wgt_user.shape)
df = pd.DataFrame(wgt_user[:,:5])
df['cls'] = ['c'+str(ii) for ii in cls_user]
sns.pairplot(df, markers='o', hue='cls', size=3.5, diag_kind='hist')
```

    (10000, 3)





    <seaborn.axisgrid.PairGrid at 0x7f16a0391ba8>




![png](output_30_2.png)



```python
'''show row side and col side at the same time'''
df1 = pd.DataFrame(wgt_prod)
df1['cls'] = ['c'+str(ii) for ii in cls_prod]
df2 = pd.DataFrame(wgt_user)
df2['cls'] = ['r'+str(ii) for ii in cls_user]
df = pd.concat([df2, df1])
df.head()

sns.pairplot(df, markers=['.']*7+['s']*7, hue='cls', size=3.5, diag_kind='hist')
```




    <seaborn.axisgrid.PairGrid at 0x7f16668aeb38>




![png](output_31_1.png)



```python
'''PCA'''
from sklearn import decomposition
pca = decomposition.PCA(n_components=2)
pca.fit(np.vstack([wgt_prod, wgt_user]))
```




    PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
        svd_solver='auto', tol=0.0, whiten=False)




```python
f = pca.transform(np.vstack([wgt_prod, wgt_user]))
f.shape
```




    (10200, 2)




```python
df = pd.DataFrame(f)
df['cls'] = ['c'+str(ii) for ii in cls_prod] + ['c'+str(ii) for ii in cls_user]
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>cls</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.721878</td>
      <td>0.361685</td>
      <td>c0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.842911</td>
      <td>0.206010</td>
      <td>c0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.725766</td>
      <td>0.354397</td>
      <td>c0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.764267</td>
      <td>0.235077</td>
      <td>c0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.786172</td>
      <td>0.348091</td>
      <td>c0</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.pairplot(df, markers='o', hue='cls', size=3.5, diag_kind='hist')
```

    /home/admin/miniconda3/envs/da03/lib/python3.6/site-packages/seaborn/axisgrid.py:2065: UserWarning: The `size` parameter has been renamed to `height`; pleaes update your code.
      warnings.warn(msg, UserWarning)





    <seaborn.axisgrid.PairGrid at 0x7f1665b5e3c8>




![png](output_35_2.png)



```python
'''t-SNE'''
from sklearn import manifold
wgt = np.vstack([wgt_prod, wgt_user])
print(wgt.shape)
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
X_tsne = tsne.fit_transform(wgt)
```

    (10200, 3)



```python
df = pd.DataFrame(X_tsne)
df['cls'] = ['c'+str(ii) for ii in cls_prod] + ['c'+str(ii) for ii in cls_user]
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>cls</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>54.600273</td>
      <td>37.230598</td>
      <td>c0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>77.810532</td>
      <td>21.399820</td>
      <td>c0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>83.906212</td>
      <td>12.056941</td>
      <td>c0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>77.232224</td>
      <td>23.684269</td>
      <td>c0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>83.977257</td>
      <td>14.840405</td>
      <td>c0</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.pairplot(df, markers='o', hue='cls', size=3.5, diag_kind='hist')
```

    /home/admin/miniconda3/envs/da03/lib/python3.6/site-packages/seaborn/axisgrid.py:2065: UserWarning: The `size` parameter has been renamed to `height`; pleaes update your code.
      warnings.warn(msg, UserWarning)





    <seaborn.axisgrid.PairGrid at 0x7f166571ebe0>




![png](output_38_2.png)



```python

```


```python

```


```python

```
