

```python
%matplotlib inline
import os, sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.metrics import f1_score, classification_report, confusion_matrix, make_scorer
from sklearn.preprocessing import OneHotEncoder
import keras
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD
from keras import regularizers
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from keras import backend as K
import tensorflow as tf
```

    Using TensorFlow backend.



```python
sys.path.append('/home/admin/github/wordroid.sblo.jp/lib')
from keras_ex.gkernel import GaussianKernel, GaussianKernel2, GaussianKernel3
from keras_ex.gkernel.sklearn import (
    RBFClassifier, RBFRegressor,
    make_model_gkernel3,
    make_model_gkernel2,
    make_model_gkernel1,
    make_model_out,
    make_model
)
```


```python
iris = datasets.load_iris()
X = iris.data.astype(np.float32)
Y = iris.target
N = Y.size
Y2 = keras.utils.to_categorical(Y, num_classes=3)

index = np.arange(N)
xtrain = X[index[index % 2 != 0],:]
ytrain = Y2[index[index % 2 != 0]]
xtest = X[index[index % 2 == 0],:]
yans = Y2[index[index % 2 == 0]]
```


```python
import warnings
warnings.filterwarnings('ignore')
```


```python

```


```python
clf1 = RBFClassifier()
hst = clf1.fit(xtrain, ytrain, epochs=10, verbose=0)
```


```python
clf2 = RBFClassifier()
hst = clf2.fit(xtrain, ytrain, epochs=10, verbose=0)
```


```python
'''
error occur
because clf2 cleared session of clf1
'''
clf1.predict(xtrain)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-7-7c8000622cb2> in <module>()
          3 because clf2 cleared session of clf1
          4 '''
    ----> 5 clf1.predict(xtrain)
    

    ~/miniconda3/envs/da02/lib/python3.6/site-packages/keras/wrappers/scikit_learn.py in predict(self, x, **kwargs)
        227         kwargs = self.filter_sk_params(Sequential.predict_classes, kwargs)
        228 
    --> 229         proba = self.model.predict(x, **kwargs)
        230         if proba.shape[-1] > 1:
        231             classes = proba.argmax(axis=-1)


    ~/miniconda3/envs/da02/lib/python3.6/site-packages/keras/engine/training.py in predict(self, x, batch_size, verbose, steps)
       1162         else:
       1163             ins = x
    -> 1164         self._make_predict_function()
       1165         f = self.predict_function
       1166         return training_arrays.predict_loop(self, f, ins,


    ~/miniconda3/envs/da02/lib/python3.6/site-packages/keras/engine/training.py in _make_predict_function(self)
        552                                                updates=self.state_updates,
        553                                                name='predict_function',
    --> 554                                                **kwargs)
        555 
        556     def _uses_dynamic_learning_phase(self):


    ~/miniconda3/envs/da02/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py in function(inputs, outputs, updates, **kwargs)
       2742                 msg = 'Invalid argument "%s" passed to K.function with TensorFlow backend' % key
       2743                 raise ValueError(msg)
    -> 2744     return Function(inputs, outputs, updates=updates, **kwargs)
       2745 
       2746 


    ~/miniconda3/envs/da02/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py in __init__(self, inputs, outputs, updates, name, **session_kwargs)
       2544         self.inputs = list(inputs)
       2545         self.outputs = list(outputs)
    -> 2546         with tf.control_dependencies(self.outputs):
       2547             updates_ops = []
       2548             for update in updates:


    ~/miniconda3/envs/da02/lib/python3.6/site-packages/tensorflow/python/framework/ops.py in control_dependencies(control_inputs)
       5424     return NullContextmanager()
       5425   else:
    -> 5426     return get_default_graph().control_dependencies(control_inputs)
       5427 
       5428 


    ~/miniconda3/envs/da02/lib/python3.6/site-packages/tensorflow/python/framework/ops.py in control_dependencies(self, control_inputs)
       4865           (hasattr(c, "_handle") and hasattr(c, "op"))):
       4866         c = c.op
    -> 4867       c = self.as_graph_element(c)
       4868       if isinstance(c, Tensor):
       4869         c = c.op


    ~/miniconda3/envs/da02/lib/python3.6/site-packages/tensorflow/python/framework/ops.py in as_graph_element(self, obj, allow_tensor, allow_operation)
       3794 
       3795     with self._lock:
    -> 3796       return self._as_graph_element_locked(obj, allow_tensor, allow_operation)
       3797 
       3798   def _as_graph_element_locked(self, obj, allow_tensor, allow_operation):


    ~/miniconda3/envs/da02/lib/python3.6/site-packages/tensorflow/python/framework/ops.py in _as_graph_element_locked(self, obj, allow_tensor, allow_operation)
       3873       # Actually obj is just the object it's referring to.
       3874       if obj.graph is not self:
    -> 3875         raise ValueError("Tensor %s is not an element of this graph." % obj)
       3876       return obj
       3877     elif isinstance(obj, Operation) and allow_operation:


    ValueError: Tensor Tensor("model_out/dense_1/Softmax:0", shape=(?, 3), dtype=float32) is not an element of this graph.



```python
clf2.predict(xtrain)
```




    array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 2, 1, 1, 0, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1,
           1, 1, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2])



## set session_clear=False, error never occur


```python
clf3 = RBFClassifier()
hst = clf3.fit(xtrain, ytrain, epochs=10, verbose=0)
```


```python
clf4 = RBFClassifier(session_clear=False)
hst = clf4.fit(xtrain, ytrain, epochs=10, verbose=0)
```


```python
'''
no error occure
'''
clf3.predict(xtrain)
```




    array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
           0, 0, 0, 2, 1, 1, 0, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1,
           1, 1, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2])




```python
clf4.predict(xtrain)
```




    array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 2, 1, 1, 0, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1,
           1, 1, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2])




```python

```


```python

```


```python

```
