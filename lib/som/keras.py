'''
Copyright (c) 2019 Norio Tamada
Released under the MIT license
https://github.com/darecophoenixx/wordroid.sblo.jp/blob/master/lib/keras_ex/gkernel/LICENSE.md
'''

import sys
import numpy as np
from tqdm import tqdm

from keras import initializers, constraints
from keras.layers import Layer
from keras.layers import Lambda
from keras.layers import Input
from keras import backend as K
from keras.models import Model, Sequential
from keras.optimizers import RMSprop, Adam, SGD

class SOM(Layer):
    
    def __init__(self, map_shape, lm_init,
                 qd,
                 r=1.0,
                 kernel_initializer='glorot_uniform',
                 kernel_constraint=None,
                 **kwargs):
        '''
        map_shape:
            shape of som_map
            typicaly (40, 50) or (3, 4) etc.
        lm_init:
            initial landmarks
            shape = (2000, num_feature) or (12, num_feature)
        r:
            gamma = 1 / (2 * r**2)
        '''
        kwargs['weights'] = [lm_init]
        self.lm_init = lm_init
        super(SOM, self).__init__(**kwargs)
        self.map_shape = map_shape
        self.qd = qd
        self.output_dim = lm_init.shape
        self.num_feature = lm_init.shape[1]
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_constraint = constraints.get(kernel_constraint)

        # kernel parameter
        self.r = r
        self.gamma = 1 / (2 * r**2)
        self._calc_qd()

    def _calc_qd(self):
        qd = self.qd
        self.qd_calced = K.cast_to_floatx(np.exp(-self.gamma * qd))
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],) +  self.output_dim
    
    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        
        self.landmarks = self.add_weight(name='landmarks',
                                         shape=self.output_dim,
                                         initializer=self.kernel_initializer,
                                         constraint=self.kernel_constraint)
        #print('self.landmarks.shape: ', self.landmarks.shape)
        super(SOM, self).build(input_shape)  # Be sure to call this somewhere!
    
    def call(self, x, training=None):
        return self.calc_delta(x, self.landmarks, self.gamma)
    
    def calc_delta(self, x, landmarks, gamma):
        '''
        landmarks : shape = (1200, num_feature)
        qd_calced : shape = (1200, 1200)
        x : shape = (None, num_feature)
        '''
        #print('landmarks.shape', landmarks.shape)
        
        xlm = K.dot(x, K.transpose(landmarks)) # (None, num_feature) . (num_feature, 1200) -> (None, 1200)
        #print('xlm: ', K.int_shape(xlm))
        x2 = K.sum(K.square(x), axis=1) # (None,)
        #print('x2: ', K.int_shape(x2))
        x2 = K.reshape(x2, (-1,1)) # (None, 1)
        #print('x2: ', K.int_shape(x2))
        x2 = K.repeat_elements(x2, self.output_dim[0], axis=1) # (None, 1200)
        #print('x2: ', K.int_shape(x2))
        lm2 = K.sum(K.square(landmarks), axis=1) # (1200,)
        #print('lm2: ', K.int_shape(lm2))
        
        d = x2 + lm2 - 2*xlm # (None, 1200)
        #print('d: ', K.int_shape(d))
        min_idx = K.argmin(d) # (None,)
        #print('min_idx: ', K.int_shape(min_idx))
        h = K.gather(self.qd_calced, min_idx) # (None, 1200)
        #print('h: ', K.int_shape(h))
        h = K.repeat(h, self.num_feature) # (None, 1200)
        h = K.permute_dimensions(h, (0,2,1))
        #print('h: ', K.int_shape(h))
        x_expand = K.repeat(x, self.output_dim[0]) # (None, 1200, num_feature)
        #print('x_expand: ', K.int_shape(x_expand))
        delta0 = h * (x_expand - landmarks) # (None, 1200, num_feature) - (1200, num_feature) -> (None, 1200, num_feature)
        #print('delta0: ', K.int_shape(delta0))
        #ret = K.exp(-gamma * d)
        return delta0
    
    def get_config(self):
        config = {
            'map_shape': self.map_shape,
            'lm_init': self.lm_init,
            'r': self.r,
            'qd': self.qd,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
        }
        base_config = super(SOM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class CalcDistance(Layer):
    
    def __init__(self, lm_init,
                 kernel_initializer='glorot_uniform',
                 kernel_constraint=None,
                 **kwargs):
        '''
        map_shape:
            shape of som_map
            typicaly (40, 50) or (3, 4) etc.
        lm_init:
            initial landmarks
            shape = (2000, num_feature) or (12, num_feature)
        '''
        kwargs['weights'] = [lm_init]
        self.lm_init = lm_init
        super(CalcDistance, self).__init__(**kwargs)
        self.output_dim = lm_init.shape[:1]
        self.num_feature = lm_init.shape[1]
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_constraint = constraints.get(kernel_constraint)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) +  self.output_dim
    
    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        
        self.landmarks = self.add_weight(name='landmarks',
                                         shape=self.lm_init.shape,
                                         initializer=self.kernel_initializer,
                                         constraint=self.kernel_constraint)
        super(CalcDistance, self).build(input_shape)  # Be sure to call this somewhere!
    
    def call(self, x, training=None):
        return self.calc_d2(x, self.landmarks)
    
    def calc_d2(self, x, landmarks):
        '''
        landmarks : shape = (1200, num_feature)
        qd : shape = (1200, 1200)
        x : shape = (None, num_feature)
        '''
        xlm = K.dot(x, K.transpose(landmarks)) # (None, num_feature) . (num_feature, 1200) -> (None, 1200)
        #print('xlm: ', K.int_shape(xlm))
        x2 = K.sum(K.square(x), axis=1) # (None,)
        #print('x2: ', K.int_shape(x2))
        x2 = K.reshape(x2, (-1,1)) # (None, 1)
        #print('x2: ', K.int_shape(x2))
        x2 = K.repeat_elements(x2, self.output_dim[0], axis=1) # (None, 1200)
        #print('x2: ', K.int_shape(x2))
        lm2 = K.sum(K.square(landmarks), axis=1) # (1200,)
        #print('lm2: ', K.int_shape(lm2))
        
        d2 = x2 + lm2 - 2*xlm # (None, 1200)
        return d2
    
    def get_config(self):
        config = {
            'lm_init': self.lm_init,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
        }
        base_config = super(CalcDistance, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class CalcDelta(Layer):
    
    def __init__(self, num_feature, qd,
                 r=1.0,
                 kernel_initializer=None,
                 kernel_constraint=None,
                 **kwargs):
        '''
        map_shape:
            shape of som_map
            typicaly (40, 50) or (3, 4) etc.
        r:
            gamma = 1 / (2 * r**2)
        '''
        super(CalcDelta, self).__init__(**kwargs)
        self.qd = qd
        self.map_shape = map_shape
        self.output_dim = (qd.shape[0], num_feature)
        self.num_feature = num_feature
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_constraint = constraints.get(kernel_constraint)

        # kernel parameter
        self.r = r
        self.gamma = 1 / (2 * r**2)
        
        self.qd_calced = np.exp(-gamma * self.qd)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) +  self.output_dim
    
    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        
        super(CalcDelta, self).build(input_shape)  # Be sure to call this somewhere!
    
    def call(self, x, training=None):
        return self.calc_delta(x)
    
    def calc_delta(self, X):
        '''
        d2 : distance^2 shape = (None, 1200)
        qd : shape = (1200, 1200)
        x : shape = (None, num_feature)
        '''
        x, d2 = X
        min_idx = K.argmin(d2) # (None,)
        #print('min_idx: ', K.int_shape(min_idx))
        h = K.gather(self.qd_calced, min_idx) # (None, 1200)
        #print('h: ', K.int_shape(h))
        h = K.repeat(h, self.num_feature) # (None, 1200)
        h = K.permute_dimensions(h, (0,2,1))
        #print('h: ', K.int_shape(h))
        x_expand = K.repeat(x, self.output_dim[0]) # (None, 1200, num_feature)
        #print('x_expand: ', K.int_shape(x_expand))
        delta0 = h * (x_expand - landmarks) # (None, 1200, num_feature) - (1200, num_feature) -> (None, 1200, num_feature)
        #print('delta0: ', K.int_shape(delta0))
        #ret = K.exp(-gamma * d)
        return delta0
    
    def get_config(self):
        config = {
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
        }
        base_config = super(CalcDelta, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



from sklearn.cluster import KMeans

class sksom_keras(object):
    
    def __init__(self, kshape, init_K,
                 form=None,
                 rand_stat=0,
                 r=None, epochs=500,
                 early_stopping=(5, 1.0e-7),
                 batch_size=1024,
                 loss='mae', optimizer=SGD(learning_rate=100.0),
                 verbose=0):
        '''
        predict:
            use KMeans
        '''
        if early_stopping:
            if len(early_stopping) != 2:
                raise Exception('lenght of early_stopping must be 2...{}'.format(early_stopping))
        self.early_stopping = early_stopping
        self.kshape = kshape
        self.init_K = init_K
        if r is None:
            self.r = (min(self.kshape)/6.0, min(self.kshape)/30.0)
        else:
            self.r = r
        self.epochs = epochs
        self.verbose = verbose
        self.optimizer = optimizer
        self.loss = loss
        self.batch_size = batch_size
        
        self.landmarks_ = self.init_K.copy()
        self.kmeans = self._kmeans()
        self.labels_ = self.kmeans.labels_
        
        self._calc_qd()
    
    def _calc_qd(self):
        qd = [np.array((ii,jj)) for ii in range(self.kshape[0]) for jj in range(self.kshape[1])]
        qd = np.array([np.square(p1 - p2).sum() for p1 in qd for p2 in qd]).reshape(np.array(self.kshape).prod(), np.array(self.kshape).prod())
        self.qd = qd
    
    def _kmeans(self):
        kmeans = KMeans(n_clusters=self.kshape[0]*self.kshape[1], n_init=1, max_iter=1)
        kmeans.labels_ = np.arange(self.kshape[0]*self.kshape[1])
        kmeans.cluster_centers_ = self.landmarks_
        kmeans._n_threads = None
        return kmeans
    
    def _make_keras_model(self, r, LM):
        inp = Input(shape=(self.init_K.shape[1],), name='inp')
        l_som = SOM(map_shape=self.kshape, lm_init=LM, r=r, qd=self.qd, name='som')
        oup = l_som(inp)

        model = Model(inp, oup, name='model')
        self.model = model
        return model
    
    def fit(self, x, nstep_r_reduce=100,
            batch_size=None, epochs=500, verbose=None, shuffle=True,
            r=None,
            optimizer=None, loss=None):
        if optimizer is None:
            optimizer = self.optimizer
        if loss is None:
            loss = self.loss
        if epochs is None:
            epochs = self.epochs
        if batch_size is None:
            batch_size = self.batch_size
        if verbose is None:
            verbose = self.verbose
        if r is None:
            r = self.r
        
        n_split, _ = divmod(epochs, nstep_r_reduce)
        if isinstance(r, tuple):
            t = np.arange(epochs)
            l_epochs = [len(ee) for ee in np.array_split(t, n_split)][::-1]
            l_r = [r[0] - (r[0]-r[1])/np.sqrt(len(l_epochs)-1)*np.sqrt(ii) for ii in range(len(l_epochs))]
            sche = [ee for ee in zip(l_epochs, l_r)]
        else:
            sche = ((epochs, r),)
        
        self.hst = {}
        num_feature = self.init_K.shape[1]
        #LM = self.landmarks_
        for ii, (i_epochs, i_r) in enumerate(sche):
            print(ii+1, ' / ', len(sche))
#            print('=====> r: ', i_r, ' / epochs: ', i_epochs)
#            self._make_keras_model(i_r, LM)
#            self.model.compile(loss=loss, optimizer=optimizer)
#            res = self.model.fit(x, np.zeros((x.shape[0],1,num_feature)),
#                                 batch_size=batch_size, epochs=i_epochs, verbose=verbose,
#                                 shuffle=shuffle)
#            for k, v in res.history.items():
#                hst.setdefault(k, [])
#                hst[k] = hst[k] + v
#            self.gamma = 1.0 / (2.0 * i_r**2)
#            self.landmarks_ = LM = self.model.get_layer('som').get_weights()[0]
            self._fit(i_epochs, i_r, x,
                      batch_size=batch_size, verbose=verbose, shuffle=shuffle,
                      optimizer=optimizer, loss=loss)
        self.kmeans.cluster_centers_[:,:] = self.landmarks_
        return self.hst
    
    def _fit(self, i_epochs, i_r,
             x,
             batch_size=None, verbose=None, shuffle=True,
             optimizer=None, loss=None):
        self.gamma = 1.0 / (2.0 * i_r**2)
        num_feature = self.init_K.shape[1]
        print('=====> r: ', i_r, ' / epochs: ', i_epochs)
        self._make_keras_model(i_r, self.landmarks_)
        self.model.compile(loss=loss, optimizer=optimizer)
        hst = self.model.fit(x, np.zeros((x.shape[0],1,1)),
                             batch_size=batch_size, epochs=i_epochs, verbose=verbose,
                             shuffle=shuffle)
        for k, v in hst.history.items():
            self.hst.setdefault(k, [])
            self.hst[k] = self.hst[k] + v
        self.landmarks_ = LM = self.model.get_layer('som').get_weights()[0]
    
    def predict(self, X):
        return self.kmeans.predict(X.astype(float))
    
    def predict_proba(self, X):
        p_list = []
        for ii in self.labels_:
            d = np.square(X - self.landmarks_[ii]).sum(axis=1)
            p_list.append(np.exp(-self.gamma * d))
        return np.vstack(p_list).T



class sksom_keras2(sksom_keras):
    
    def __init__(self, kshape, init_K,
                 form=None,
                 rand_stat=0,
                 r=None, epochs=500,
                 early_stopping=(5, 1.0e-7),
                 batch_size=1024,
                 loss='mae', optimizer=SGD(learning_rate=100.0),
                 verbose=0):
        super(sksom_keras2, self).__init__(
            kshape=kshape, init_K=init_K, form=form, rand_stat=rand_stat,
            r=r,
            epochs=epochs, early_stopping=early_stopping, batch_size=batch_size,
            verbose=verbose,
            loss=loss, optimizer=optimizer
            )
        self.num_feature = self.init_K.shape[1]
    
    def _make_keras_model(self, r, LM):
        inp = Input(shape=(self.init_K.shape[1],), name='inp')
        l_som = SOM(map_shape=self.kshape, lm_init=LM, r=r, qd=self.qd, name='som')
        oup = l_som(inp)
        model = Model(inp, oup, name='model')
        
        calc_d = CalcDistance(lm_init=LM, name='calc_d')
        oup_d = calc_d(inp)
        oup_min_d = Lambda(lambda d: K.min(d, axis=1), name='min_d')(oup_d)
        model_min_d = Model(inp, oup_min_d, name='model_min_d')
        
        self.model = model
        self.models = {
            'model': model,
            'model_min_d': model_min_d,
        }
        return model
    
    def _fit(self, i_epochs, i_r,
             x,
             batch_size=None, verbose=0, shuffle=True,
             optimizer=None, loss=None):
        self.gamma = 1.0 / (2.0 * i_r**2)
        num_feature = self.init_K.shape[1]
        print('=====> r: ', i_r, ' / epochs: ', i_epochs)
        self._make_keras_model(i_r, self.landmarks_)
        self.model.compile(loss=loss, optimizer=optimizer)
        Y = np.zeros((x.shape[0],1,1))
        hst = self.model.fit(x, Y,
                             batch_size=batch_size, epochs=i_epochs, verbose=verbose,
                             shuffle=shuffle)
        for k, v in hst.history.items():
            self.hst.setdefault(k, [])
            self.hst[k] = self.hst[k] + v
        self.landmarks_ = LM = self.model.get_layer('som').get_weights()[0]
        d2_mean = self._calc_mean_dist(x)
        if verbose is not None:
            print('mean distance to closest landmark : ', d2_mean)
            

#    def _fit(self, i_epochs, i_r,
#             x,
#             batch_size=None, verbose=None, shuffle=True,
#             optimizer=None, loss=None):
#        self.gamma = 1.0 / (2.0 * i_r**2)
#        num_feature = self.init_K.shape[1]
#        print('=====> r: ', i_r, ' / epochs: ', i_epochs)
#        self._make_keras_model(i_r, self.landmarks_)
#        self.model.compile(loss=loss, optimizer=optimizer)
#        Y = np.zeros((x.shape[0],1,1))
#        with tqdm(total=i_epochs, file=sys.stdout,
#                  disable=False if 0<verbose else True) as pbar:
#            for _ in range(i_epochs):
#                hst = self.model.fit(x, Y,
#                                     batch_size=batch_size, epochs=1, verbose=0,
#                                     shuffle=shuffle)
#                for k, v in hst.history.items():
#                    self.hst.setdefault(k, [])
#                    self.hst[k] = self.hst[k] + v
#                self.landmarks_ = LM = self.model.get_layer('som').get_weights()[0]
#                d2_mean = self._calc_mean_dist(x)
#                
#                if 1 < verbose:
#                    pbar.set_description('r: %f / gamma: %f / mean distance: %f' % (i_r, self.gamma, d2_mean))
#                pbar.update(1)
    
    def _calc_mean_dist(self, x):
        '''calc MeanDist2ClosestLM'''
        self.models['model_min_d'].get_layer('calc_d').set_weights([self.landmarks_])
        d2 = self.models['model_min_d'].predict(x)
        self.hst.setdefault('MeanDist2ClosestLM', [])
        d2_mean = d2.mean()
        self.hst['MeanDist2ClosestLM'].append(d2_mean)
        return d2_mean
        
    






