'''
Copyright (c) 2019 Norio Tamada
Released under the MIT license
https://github.com/darecophoenixx/wordroid.sblo.jp/blob/master/lib/keras_ex/gkernel/LICENSE.md
'''

'''
kerasを用いて、GPUで高速に処理できるように
'''

import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.cluster._kmeans import _labels_inertia
from sklearn.utils.extmath import row_norms
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn import metrics

from tensorflow.keras import initializers, constraints
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
import tensorflow as tf

from som import som as _som

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
        #kwargs['weights'] = [lm_init]
        kwargs['name'] = name
        #self.set_weights([lm_init])
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
    
    @tf.autograph.experimental.do_not_convert
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
    
    @tf.autograph.experimental.do_not_convert
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






class sksom_keras(object):
    
    def __init__(self, kshape, init_K,
                 form=None,
                 rand_stat=0,
                 r=None, epochs=500,
                 early_stopping=(5, 1.0e-7),
                 batch_size=1024,
                 loss='mae', optimizer=Adam(learning_rate=0.001),
                 verbose=0):
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
        self.labels_ = np.arange(self.kshape[0]*self.kshape[1])
        if form == 'sphere':
            self._calc_qd_sphere()
        elif form == 'hex':
            self._calc_qd_hex()
        else:
            self._calc_qd()
    
    def _calc_qd(self):
        qd = [np.array((ii,jj)) for ii in range(self.kshape[0]) for jj in range(self.kshape[1])]
        self.map_xy = np.vstack(qd)
        qd = np.array([np.square(p1 - p2).sum() for p1 in qd for p2 in qd]).reshape(np.array(self.kshape).prod(), np.array(self.kshape).prod())
        self.qd = qd
    
    def _calc_qd_hex(self):
        a = np.sqrt(3) / 2.0
        b = 3.0
        a, b = a / a / 2, b / a / 2
        xy0 = [np.array(((0,jj*2*a), (b/2,a+jj*2*a))) for jj in range(self.kshape[1])]
        xy0 = np.vstack(xy0)
        l = []
        for ii in range(self.kshape[0]):
            sho, amari = divmod(ii, 2)
            if amari == 0:
                l.append(xy0[::2,:] + np.array((sho*b,0)))
            else:
                l.append(xy0[1::2,:] + np.array((sho*b,0)))
        map_xy = np.vstack(l)
        self.map_xy = map_xy
        qd = np.array([np.square(p1 - p2).sum() for p1 in map_xy for p2 in map_xy]).reshape(np.array(self.kshape).prod(), np.array(self.kshape).prod())
        self.qd = qd
    
    def _calc_qd_sphere(self):
        longitude = [2*np.pi*ii/self.kshape[1] for ii in range(self.kshape[1])]
        latitude = np.linspace(-np.pi/2, np.pi/2, self.kshape[0]+1)[1:] - np.pi/self.kshape[0]/2
        pos0 = [np.array((np.cos(lo), np.sin(lo), np.sin(la))) for la in latitude for lo in longitude]
        pos = [np.array((ee[0]*np.sqrt(1-ee[2]**2), ee[1]*np.sqrt(1-ee[2]**2), ee[2])) for ee in pos0]
        self.map_xy = np.r_[pos]
        qd = np.array([np.arccos(np.dot(p0, p1).clip(-1,1)) for p0 in pos for p1 in pos])
        qd = qd.reshape((np.array(self.kshape).prod(), np.array(self.kshape).prod()))
        self.qd = qd * (1 / (2*np.pi/self.kshape[1]))
    
    def _make_keras_model(self, r, LM):
        inp = Input(shape=(self.init_K.shape[1],), name='inp')
        l_som = SOM(map_shape=self.kshape, lm_init=LM, r=r, qd=self.qd, name='som')
        oup = l_som(inp)

        model = Model(inp, oup, name='model')
        self.model = model
        return model
    
    def fit(self, x, nstep_r_reduce=50,
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
            self._fit(i_epochs, i_r, x,
                      batch_size=batch_size, verbose=verbose, shuffle=shuffle,
                      optimizer=optimizer, loss=loss)
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
        assert X.shape[1] == self.init_K.shape[1]
        #X_squared_norm = row_norms(X, squared=True)
        #sample_weight = np.ones((X.shape[0],), dtype=float)
        #labels, inertia = _labels_inertia(X.astype(float), sample_weight, X_squared_norm.astype(float), self.landmarks_.astype(float))
        d = metrics.pairwise_distances(X, self.landmarks_)
        labels = self.labels_[d.argmin(axis=1)]
        return labels
    
    def score(self, X, y=None):
        assert X.shape[1] == self.init_K.shape[1]
        X_squared_norm = row_norms(X, squared=True)
        sample_weight = np.ones((X.shape[0],), dtype=float)
        labels, inertia = _labels_inertia(X.astype(float), sample_weight, X_squared_norm.astype(float), self.landmarks_.astype(float))
        return -inertia
    
    def label2xy(self, labels):
        return np.c_[self.map_xy[labels,1], self.map_xy[labels,0]]
    
    def plot_hex(self, figsize=10, s=450, target=[0,1,2], ax=None, fig=None):
        '''
        s : ( figsize x 100(dpi) / kshape[1] ) ** 2
        ex ((10*100)/40+2)**2
        '''
        if not isinstance(figsize, tuple):
            figsize = (figsize, figsize)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=100)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis("off")
        ax.set_aspect('equal')
        codes = self.landmarks_[:,target].copy()
        for ii in range(len(target)):
            v_max = codes[:,ii].max()
            v_min = codes[:,ii].min()
            codes[:,ii] = (codes[:,ii] - v_min) / (v_max - v_min)
        for ii, ee in enumerate(self.map_xy):
            ax.scatter(ee[1], ee[0], marker='h', s=s, color=codes[ii])
        return ax


class sksom_keras2(sksom_keras):
    
    def __init__(self, kshape, init_K,
                 form=None,
                 rand_stat=0,
                 r=None, epochs=500,
                 early_stopping=(5, 1.0e-7),
                 batch_size=1024,
                 loss='mae', optimizer=Adam(learning_rate=0.001),
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
            
    def _calc_mean_dist(self, x):
        '''calc MeanDist2ClosestLM'''
        self.models['model_min_d'].get_layer('calc_d').set_weights([self.landmarks_])
        d2 = self.models['model_min_d'].predict(x)
        self.hst.setdefault('MeanDist2ClosestLM', [])
        d2_mean = d2.mean()
        self.hst['MeanDist2ClosestLM'].append(d2_mean)
        return d2_mean
    


class som(sksom_keras2):
    """class som (self-organizing map) implementation
    
    kshape and init_K must be given
    
    Parameters
    ----------
    
    kshape : sequence of length 2
        shape of som map
    
    init_K : ndarray of shape (n_landmarks, n_features)
        n_landmarks = kshape[0] * kshape[1]
    
    form : {None, 'hex', 'sphere'}, default=None
        The topology type when measuring distance in the map:
        
        None
            rect
        'hex'
            hex
        'sphere'
            sphere
    
    r : sequence of length 2 of float, or float, defalut=None
        radius in a first training phases
        
        sequence of length 2 of float
            decreases from r[0] to r[1] during training
        float
            use same radius during training
        None
            use (max(kshape) / 6, max(kshape) / 30)
    
    epochs : int, default=500
        number of epochs to train the model
    
    batch_size : int, default=1024
        number of samples per gradient update
    
    verbose : int, default=0
        verbosity mode
    
    early_stopping : , default=(5, 1.0e-7)
    
    loss : , default='mae'
        Loss function. Maybe be a string (name of loss function), or a tf.keras.losses.Loss instance
    
    optimizer : , default=Adam(learning_rate=0.001)
        String (name of optimizer) or tf.keras optimizer instance
    
    """
    
    def __init__(self, kshape, init_K,
                 form=None,
                 r=None, epochs=500,
                 early_stopping=(5, 1.0e-7),
                 batch_size=1024,
                 loss='mae', optimizer=Adam(learning_rate=0.001),
                 verbose=0):
        if r is None:
            r = (max(kshape) / 6.0, max(kshape) / 30.0)
        super().__init__(
            kshape=kshape, init_K=init_K, form=form,
            r=r,
            epochs=epochs, early_stopping=early_stopping, batch_size=batch_size,
            verbose=verbose,
            loss=loss, optimizer=optimizer
            )
    
    def fit(self, X, y=None,
            nstep_r_reduce=50,
            batch_size=None, epochs=500, verbose=None, shuffle=True,
            r=None, r2=None,
            optimizer=None, loss=None):
        """compute som map
        
        Parameters
        ----------
        .
            X (array-like of shape (n_samples, n_features))
                training instances
            
            y (ignored)
                Not used, present here for API consistency by convention.
            
            nstep_r_reduce (int, default=50)
                number of epochs reducing radius
            
            r (sequence of length 2 of float, or float, defalut=None)
                radius in a first training phases
                
                sequence of length 2 of float
                    decreases from r[0] to r[1] during training
                float
                    use same radius during training
                None
                    use self.r
            
            r2 (sequence of length 2 of float, or float, defalut=None)
                radius in a second training phases
            
                sequence of length 2 of float
                    decreases from r2[0] to r2[1] during training
                float
                    use same radius during training
                None
                    use 1.0
        
        Returns
        -------
        self
            Fitted estimator.
        """
        if r is None:
            r = self.r
        hst = super().fit(X,
                          nstep_r_reduce=nstep_r_reduce,
                          batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle,
                          r=r,
                          optimizer=optimizer, loss=loss)
        self.hst1 = hst.copy()
        
        if r2 is None:
            r2 = r[1] if isinstance(r, tuple) else r
        self.hst2 = super().fit(X,
                           nstep_r_reduce=nstep_r_reduce,
                           batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle,
                           r=r2,
                           optimizer=optimizer, loss=loss)
        return self



class sksom(TransformerMixin, ClusterMixin, BaseEstimator):
    """som (self-organizing map) scikit-learn api
    
    Kaggle notebook: [som] scikit-learn api
        https://www.kaggle.com/wordroid/som-scikit-learn-api
    
    Kaggle notebook: [som] moon data 10000 with GPU
        https://www.kaggle.com/wordroid/som-moon-data-10000-with-gpu
        
        Since it is implemented in tf.keras, it can be calculated very fast if you have a GPU.
    
    Parameters
    ----------
    
    kshape : sequence of length 2, default=(20, 30)
        shape of som map
    
    init_K : ndarray of shape (n_landmarks, n_features), default=None
        n_landmarks = kshape[0] * kshape[1]
    
    initialization : {'linear', None}, defalut='linear'
        Method for initialization:
        
        'linear'
            PCA
        None
            random initialization, landmarks are selected from input X
    
    form : {None, 'hex', 'sphere'}, default=None
        The topology type when measuring distance in the map:
        
        None
            rect
        'hex'
            hex
        'sphere'
            sphere
    
    rand_stat : int, RandomState instance or None, default=None
        RandomState for random initialization
    
    r1 : sequence of length 2 of float, or float, defalut=None
        radius in a first training phases
        
        sequence of length 2 of float
            decreases from r1[0] to r1[1] during training
        float
            use same radius during training
        None
            use (max(kshape)/6.0, 1.0)
    
    r2 : sequence of length 2 of float, or float, defalut=None
        radius in a second training phases
        
        sequence of length 2 of float
            decreases from r2[0] to r2[1] during training
        float
            use same radius during training
        None
            use 1.0
    
    epochs : int, default=500
        number of epochs to train the model
    
    batch_size : int, default=1024
        number of samples per gradient update
    
    verbose : int, default=0
        verbosity mode
    
    early_stopping : , default=(5, 1.0e-7)
    
    loss : , default='mae'
        Loss function. Maybe be a string (name of loss function), or a tf.keras.losses.Loss instance
    
    optimizer : , default=Adam(learning_rate=0.001)
        String (name of optimizer) or tf.keras optimizer instance
    
    
    Attributes
    ----------
    
    cluster_centers_ : ndarray of shape (kshape[0] * kshape[1], n_features)
        coordinates of som-map
    
    landmarks_ : ndarray of shape (kshape[0] * kshape[1], n_features)
        coordinates of som-map
    
    labels_ : ndarray of shape (n_samples,)
        Labels of each point
    """
    
    def __init__(self, kshape=(20, 30), init_K=None,
                 initialization='linear',
                 form=None,
                 rand_stat=None,
                 r1=None, r2=None,
                 epochs=500,
                 early_stopping=(5, 1.0e-7),
                 batch_size=1024,
                 loss='mae', optimizer=Adam(learning_rate=0.001),
                 verbose=0):
        self.early_stopping = early_stopping
        self.form = form
        self.kshape = kshape
        self.init_K = init_K
        self.r1 = r1
        self.r2 = r2
        self.epochs = epochs
        self.verbose = verbose
        self.optimizer = optimizer
        self.loss = loss
        self.batch_size = batch_size
        self.rand_stat = rand_stat
        self.initialization = initialization
    
    def fit(self, X, y=None):
        """compute som map
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            training instances
        
        y : Ignored
            Not used, present here for API consistency by convention.
        
        Returns
        -------
        self
            Fitted estimator.
        """
        if self.initialization == 'linear':
            sinit = _som.SimpleSOM(self.kshape, initialization_func=self.initialization)
            sinit._initialize(X)
            self.init_K = sinit.K
        elif self.initialization is None:
            sinit = _som.SimpleSOM(self.kshape, initialization_func=self.initialization)
            sinit._initialize(X)
            self.init_K = sinit.K
        else:
            raise Exception('not available initialization :', self.initialization)
            
        self.som = som(kshape=self.kshape, init_K=self.init_K,
                 form=self.form,
                 r=self.r1, epochs=self.epochs,
                 early_stopping=self.early_stopping,
                 batch_size=self.batch_size,
                 loss=self.loss, optimizer=self.optimizer,
                 verbose=self.verbose)
        self.som.fit(X, y=y,
            nstep_r_reduce=50,
            batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose, shuffle=True,
            r=self.r1, r2=self.r2,
            optimizer=self.optimizer, loss=self.loss)
        self.hst1, self.hst2 = self.som.hst1, self.som.hst2
        return self
    
    def predict(self, X):
        """Predict the closest landmark each sample in X belongs to.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            new data to predict
        
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            index of the landmark each sample belongs to
        """
        return self.som.predict(X)
    
    def score(self, X, y=None):
        """opposite of the value of X on the objective
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            new data
        
        y : ignored
            not used, present here for API consistency by convention
        
        Returns
        -------
        score : float
            opposite of the value of X on the objective
        """
        return self.som.score(X, y)
    
    def get_cluster_centers_(self):
        return self.som.landmarks_
    cluster_centers_ = property(get_cluster_centers_)
    landmarks_ = property(get_cluster_centers_)
    
    def get_labels_(self):
        return self.som.labels_
    labels_ = property(get_labels_)
    
    def label2xy(self, labels):
        """convert labels to som 2D-map
        
        Parameters
        ----------
        labels : ndarray of shape (n_samples,)
            index of the landmark each sample belongs to
        
        Returns
        -------
        xy-pos : ndarray of shape (n_samples, 2)
            coordinates of each labels
        """
        return self.som.label2xy(labels)
    
    def plot_hex(self, figsize=10, s=450, target=[0,1,2], ax=None, fig=None):
        """plot hex map
        
        Parameters
        ----------
        s : float, default=450
            ( figsize x 100(dpi) / kshape[1] ) ** 2
            ex ((10*100)/40+2)**2
        
        Returns
        -------
        ax : matplotlib.axes.Axes
            the matplotlib axes containing the plot
        """
        ax = self.som.plot_hex(figsize=figsize, s=s, target=target, ax=ax, fig=fig)
        return ax



conv2img = _som.conv2img

