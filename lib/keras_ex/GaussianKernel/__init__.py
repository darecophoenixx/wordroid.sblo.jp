'''
Copyright (c) 2018 Norio Tamada
Released under the MIT license
https://github.com/darecophoenixx/wordroid.sblo.jp/blob/master/lib/keras_ex/GaussianKernel/LICENSE.md
'''
from collections import deque

import numpy as np
from keras import initializers
from keras.engine.topology import Layer
from keras import backend as K
from tensorflow.python.framework.errors_impl import InvalidArgumentError

class GaussianKernel(Layer):
    
    def __init__(self, num_landmark, num_feature,
                 kernel_initializer='glorot_uniform',
                 kernel_gamma='auto',
                 **kwargs):
        '''
        num_landmark:
            number of landmark
            that was number of output features
        num_feature:
            depth of landmark
            equal to inputs.shape[1]
        kernel_gamma:
            kernel parameter
            if 'auto', use 1/(2 * d_mean**2)
            d is distance between samples and landmark
            d_mean is mean of d
        '''
        super(GaussianKernel, self).__init__(**kwargs)
        
        self.output_dim = num_landmark
        self.num_feature = num_feature
        self.kernel_initializer = initializers.get(kernel_initializer)
        
        # for loop
        self.indx = K.arange(self.output_dim)
        
        # kernel parameter
        self.kernel_gamma= kernel_gamma

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.output_dim, self.num_feature),
                                      initializer=self.kernel_initializer)
        super(GaussianKernel, self).build(input_shape)  # Be sure to call this somewhere!
    
    def call(self, x, training=None):
        return self.gauss(x, self.kernel, self.kernel_gamma)
    
    def gauss(self, x, K_tf, gamma):
        def fn(ii):
            lm = K.gather(K_tf, ii)
            return K.sum(K.square(x - lm), axis=1)
        d2 = K.map_fn(fn, self.indx, dtype='float32')
        d2 = K.transpose(d2)
        if gamma == 'auto':
            '''
            gamma is calculated by each batch
            '''
            d = K.sqrt(d2)
            d_mean = K.mean(d)
            gamma = 1. / (2. * d_mean**2)
        return K.exp(-gamma * d2)


class GaussianKernel2(Layer):
    
    def __init__(self, landmarks, **kwargs):
        '''
        landmarks:
            fixed landmarks using
        '''
        super(GaussianKernel2, self).__init__(**kwargs)
        self.landmarks = landmarks.astype(np.float32)
        self.num_landmark, self.num_feature = landmarks.shape
        self.output_dim = self.num_landmark
        
        # for loop
        self.indx = K.arange(self.output_dim)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        
        self.gamma_elm = self.add_weight(name='gamma_elm',
                                      shape=(1, ),
                                      initializer=initializers.random_uniform(-2, -1))
        super(GaussianKernel2, self).build(input_shape)  # Be sure to call this somewhere!
    
    def call(self, x, training=None):
        return self.gauss(x, self.landmarks, K.exp(self.gamma_elm), training=training)
    
    def gauss(self, x, K_tf, gamma, training=None):
        def fn(ii):
            lm = K.gather(K_tf, ii)
            return K.sum(K.square(x - lm), axis=1)
        d2 = K.map_fn(fn, self.indx, dtype='float32')
        d2 = K.transpose(d2)
        
        return K.exp(-gamma * d2)


class GaussianKernel3(Layer):
    
    def __init__(self, num_landmark, num_feature,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        '''
        num_landmark:
            number of landmark
            that was number of output features
        num_feature:
            depth of landmark
            equal to inputs.shape[1]
        '''
        super(GaussianKernel3, self).__init__(**kwargs)
        
        self.output_dim = num_landmark
        self.num_feature = num_feature
        self.kernel_initializer = initializers.get(kernel_initializer)
        
        # for loop
        self.indx = K.arange(self.output_dim)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.output_dim, self.num_feature),
                                      initializer=self.kernel_initializer)
        self.gamma_elm = self.add_weight(name='gamma_elm',
                                      shape=(1, ),
                                      initializer=initializers.random_uniform(-2, -1))
        super(GaussianKernel3, self).build(input_shape)  # Be sure to call this somewhere!
    
    def call(self, x, training=None):
        return self.gauss(x, self.kernel, K.exp(self.gamma_elm))
    
    def gauss(self, x, K_tf, gamma):
        def fn(ii):
            lm = K.gather(K_tf, ii)
            return K.sum(K.square(x - lm), axis=1)
        d2 = K.map_fn(fn, self.indx, dtype='float32')
        d2 = K.transpose(d2)
        
        return K.exp(-gamma * d2)

