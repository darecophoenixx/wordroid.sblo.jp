'''
Copyright (c) 2019 Norio Tamada
Released under the MIT license
https://github.com/darecophoenixx/wordroid.sblo.jp/blob/master/lib/keras_ex/gkernel/LICENSE.md
'''

from . import GaussianKernel, GaussianKernel2, GaussianKernel3


import copy

import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.optimizers import RMSprop, Adam
from keras.initializers import glorot_uniform
from keras.models import Sequential, Model
from keras.layers import Input, Dense
from keras import regularizers
from keras import backend as K


def make_model_gkernel1(
    nn=4, num_lm=2,
    random_state=None, lm=None, gamma=None):
    '''
    just activation
    '''
    inp = Input(shape=(nn,), name='inp')
    oup = inp
    
    if lm is None:
        rs = np.random.RandomState(random_state)
        lm = rs.random_sample((num_lm, nn))
    if gamma is None:
        gamma = 1/(2*np.sqrt(nn/6)*2)
    weights = [np.log(np.array([gamma]))]
    oup = GaussianKernel2(lm, weights=weights, name='gkernel', trainable=False)(oup)
    
    model = Model(inp, oup, name='model_gkernel')
    return lm, model

def make_model_gkernel2(
    nn=4, num_lm=2,
    random_state=None, lm=None, gamma=None):
    
    inp = Input(shape=(nn,), name='inp')
    oup = inp
    
    if lm is None:
        rs = np.random.RandomState(random_state)
        lm = rs.random_sample((num_lm, nn))
    if gamma is None:
        gamma = 1/(2*np.sqrt(nn/6)*2)
    weights = [np.log(np.array([gamma]))]
    oup = GaussianKernel2(lm, weights=weights, name='gkernel')(oup)
    
    model = Model(inp, oup, name='model_gkernel')
    return lm, model

def make_model_gkernel3(
    nn=4, num_lm=2,
    random_state=None, lm=None, gamma=None):

    inp = Input(shape=(nn,), name='inp')
    oup = inp
    
    if lm is None:
        rs = np.random.RandomState(random_state)
        lm = rs.random_sample((num_lm, nn))
    if gamma is None:
        gamma = 1/(2*np.sqrt(nn/6)*2)
    weights = [lm, np.log(np.array([gamma]))]
    oup = GaussianKernel3(num_landmark=num_lm, num_feature=nn, weights=weights, name='gkernel')(oup)
    
    model = Model(inp, oup, name='model_gkernel')
    return lm, model

def make_model_out(
    num_lm=2, num_cls=3,
    activation='softmax',
    reg_l1=0.0, reg_l2=0.0,
    random_state=None
):
    inp = Input(shape=(num_lm,), name='inp')
    oup = Dense(num_cls,
                activation=activation,
                kernel_initializer=glorot_uniform(random_state),
                kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2),
                bias_regularizer=regularizers.l1_l2(reg_l1, reg_l2)
               )(inp)
    
    model = Model(inp, oup, name='model_out')
    return model

def make_model(
    make_model_gkernel=make_model_gkernel2,
    make_model_out=make_model_out,
    reg_l1=0.0, reg_l2=0.0,
    nn=2,
    num_lm=2, lm=None, gamma=None,
    random_state=None,
    num_cls=2, activation='softmax',
    opt=0.02, lr=0.02,
    loss='categorical_crossentropy',
    session_clear=True,
    #gkernel_multipliers=1.0,
    lm_select_from_x=None,
    tol=None
):
    if session_clear:
        K.clear_session()
    
    lm, model_gkernel = make_model_gkernel(
        nn=nn, num_lm=num_lm,
        random_state=random_state, lm=lm, gamma=gamma
    )
    model_out = make_model_out(
        num_lm=num_lm, num_cls=num_cls,
        activation=activation,
        reg_l1=reg_l1, reg_l2=reg_l2,
        random_state=random_state
    )
    #model_gkernel.summary()
    #model_out.summary()
    
    inp = model_gkernel.inputs[0]
    oup = model_gkernel(inp)
    oup = model_out(oup)
    
    model = Model(inp, oup)
    #model.summary()
#     if isinstance(opt, float):
#         opt = Adam(opt)
#     learning_rate_multipliers = {
#         'gkernel': gkernel_multipliers,
#     }
#     print(learning_rate_multipliers)
#     opt = Adam_lr_mult(
#         lr,
#         multipliers=learning_rate_multipliers,
#         debug_verbose=False)
    opt = Adam(lr)
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    return model


class RBFBase(object):
    
    def _fit(self, x, y, sample_weight=None, **kwargs):
        sk_params_org = copy.deepcopy(self.sk_params)
        
        ### nn
        nn = x.shape[1]
        sk_params_org.update({'nn': self.sk_params.get('nn')})
        self.set_params(nn=nn)
        
        ### learning_rate
        if self.sk_params.get('lr') is None:
            lr = 0.02 # default lr
            sk_params_org.update({'lr': None})
            self.set_params(lr=lr)
        else:
            lr = self.sk_params['lr'] # for later using
        
        ### gamma
        if self.sk_params.get('gamma') == 'scale':
            sk_params_org.update({'gamma': 'scale'})
            self.set_params(gamma=1 / (nn * x.var()))
            #print('scale gamma >', self.sk_params['gamma'])
        
        ### tol
        tol = self.sk_params.get('tol', np.finfo(np.float32).eps*100)
        
        ### callbacks
        if self.sk_params.get('callbacks', None) is None:
            lr_reducer = ReduceLROnPlateau(monitor='loss', 
                               factor=1/2,
                               verbose=0,
                               cooldown=0,
                               patience=5,
                               min_lr=lr/64/2)
            #tol = np.sqrt(np.finfo(np.float32).eps)
            #tol = np.finfo(np.float32).eps
            #tol = 0.0
            early_stopping = EarlyStopping(monitor='loss', patience=10, min_delta=tol, restore_best_weights=True)
            callbacks0 = [lr_reducer, early_stopping]
            callbacks = [lr_reducer, early_stopping]
            sk_params_org.update({'callbacks': None})
            self.set_params(callbacks=callbacks)
        else:
            callbacks0 = self.sk_params['callbacks']
            callbacks = self.sk_params['callbacks']
        
        ### lm_select_from_x
        #print('''self.sk_params.get('lm_select_from_x') >''', self.sk_params.get('lm_select_from_x'))
        if self.sk_params.get('lm_select_from_x'):
            random_state = self.sk_params.get('random_state')
            rs = np.random.RandomState(random_state)
            lm = x[rs.choice(np.arange(x.shape[0]), self.sk_params['num_lm'], replace=False)]
            sk_params_org.update({'lm': self.sk_params.get('lm')})
            self.set_params(lm=lm)
        
        hst_all = {}
        if self.__class__.__name__ == 'RBFRegressor':
            kwargs.update({'sample_weight': sample_weight})
            hst = super().fit(x, y, **kwargs)
        else:
            hst = super().fit(x, y, sample_weight, **kwargs)
        # update history
        for k in hst.history:
            hst_all.setdefault(k, [])
            hst_all[k].extend(hst.history[k])
        
        fit_args = copy.deepcopy(self.filter_sk_params(Sequential.fit))
        fit_args.update(kwargs)
        fit_args.update({'sample_weight': sample_weight})
        
        def lr_schedule2(epoch):
            lr1 = lr / 2
            #print('lr >', lr1)
            return lr1
        def lr_schedule4(epoch):
            lr1 = lr / 4
            #print('lr >', lr1)
            return lr1
        def lr_schedule8(epoch):
            lr1 = lr / 8
            #print('lr >', lr1)
            return lr1
        
        batch_size = fit_args.get('batch_size', 32)
        
        # 2
        lr_scheduler = LearningRateScheduler(lr_schedule2)
        callbacks = callbacks0 + [lr_scheduler]
        fit_args['callbacks'] = callbacks
        fit_args['batch_size'] = batch_size * 2
        hst = self.model.fit(x, y, **fit_args)
        # update history
        for k in hst.history:
            hst_all.setdefault(k, [])
            hst_all[k].extend(hst.history[k])
        # 3
        lr_scheduler = LearningRateScheduler(lr_schedule4)
        callbacks = callbacks0 + [lr_scheduler]
        fit_args['callbacks'] = callbacks
        fit_args['batch_size'] = batch_size * 4
        hst = self.model.fit(x, y, **fit_args)
        # update history
        for k in hst.history:
            hst_all.setdefault(k, [])
            hst_all[k].extend(hst.history[k])
        # 4
        lr_scheduler = LearningRateScheduler(lr_schedule8)
        callbacks = callbacks0 + [lr_scheduler]
        fit_args['callbacks'] = callbacks
        fit_args['batch_size'] = batch_size * 8
        hst = self.model.fit(x, y, **fit_args)
        # update history
        for k in hst.history:
            hst_all.setdefault(k, [])
            hst_all[k].extend(hst.history[k])
        
        # 2
        lr_scheduler = LearningRateScheduler(lr_schedule2)
        callbacks = callbacks0 + [lr_scheduler]
        fit_args['callbacks'] = callbacks
        fit_args['batch_size'] = batch_size * 16
        hst = self.model.fit(x, y, **fit_args)
        # update history
        for k in hst.history:
            hst_all.setdefault(k, [])
            hst_all[k].extend(hst.history[k])
        # 3
        lr_scheduler = LearningRateScheduler(lr_schedule4)
        callbacks = callbacks0 + [lr_scheduler]
        fit_args['callbacks'] = callbacks
        fit_args['batch_size'] = batch_size * 32
        hst = self.model.fit(x, y, **fit_args)
        # update history
        for k in hst.history:
            hst_all.setdefault(k, [])
            hst_all[k].extend(hst.history[k])
        # 4
        lr_scheduler = LearningRateScheduler(lr_schedule8)
        callbacks = callbacks0 + [lr_scheduler]
        fit_args['callbacks'] = callbacks
        fit_args['batch_size'] = batch_size * 64
        hst = self.model.fit(x, y, **fit_args)
        # update history
        for k in hst.history:
            hst_all.setdefault(k, [])
            hst_all[k].extend(hst.history[k])
        
        #print(self.sk_params)
        #print(sk_params_org)
        self.set_params(**sk_params_org)
        #print(self.sk_params)
        return hst_all
    
    def current_gamma(self):
        for ew in self.model.layers[1].layers[1].get_weights():
            if len(ew.shape)==1:
                c_gamma = ew[0]
        return np.exp(c_gamma)
    
    def current_lm(self):
        model_gkernel = self.model.get_layer('model_gkernel')
        layer_gkernel = model_gkernel.get_layer('gkernel')
        try:
            lm = layer_gkernel.landmarks
        except:
            lm = layer_gkernel.get_weights()[0]
        return lm


class RBFClassifier(RBFBase, KerasClassifier):
    """RBF kernel Classification.
    
    Parameters
    ----------
    num_lm : int, optional (default=2)
        number of landmarks
    lm : array (num_lm, num_features), optional (default=None)
        initial landmarks
        if None, set by function `make_model_gkernelX`.
        if using make_model_gkernel2 for `make_model_gkernelX`,
        lm is fixed (no train).
        if using make_model_gkernel3 for `make_model_gkernelX`,
        lm is trained.
    lm_select_from_x : bool, optional (default=False)
        if True, lm are selected from input x.
    gamma : float or str, optional (default=None)
        RBF kernel parameter
        'scale'
        if using make_model_gkernel2 or make_model_gkernel3 for make_model_gkernel,
        gamma is trained
    make_model_gkernel : function, optional (default=make_model_gkernel2)
        function to configure RBF kernel
        make_model_gkernel2 : use fixed landmarks
        make_model_gkernel3 : train landmarks
    make_model_out : function, optional (default=make_model_out)
        dense layer just before output
    reg_l1 : float, optional (default=0.0)
        regularization parameter.
    reg_l2 : float, optional (default=0.0)
        regularization parameter.
    tol : float, optional (default=1e-3)
        Tolerance for stopping criterion.
    activation : str, optional (default='softmax')
        activation for output
    loss : str, optional (default='categorical_crossentropy')
        loss function
    lr : float, optional (default=0.02)
        leraning rate
    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator
    
    verbose : int, default: 0
        param for keras
    epochs : int, optional (default=1)
        param for keras
    batch_size : int, optional (default=32)
        param for keras
        doubles every stage
    
    """
    
    def __init__(self, build_fn=make_model, **sk_params):
        super().__init__(build_fn, **sk_params)
    
    def fit(self, x, y, sample_weight=None, **kwargs):
        loss_org = self.sk_params.get('loss', None)
        ### num_cls
        y = np.array(y)
        if len(y.shape) == 2 and y.shape[1] > 1:
            classes_ = np.arange(y.shape[1])
            if 'loss' in self.sk_params and self.sk_params.get('loss') is None:
                # use default 'categorical_crossentropy'
                del self.sk_params['loss']
        elif (len(y.shape) == 2 and y.shape[1] == 1) or len(y.shape) == 1:
            classes_ = np.unique(y)
            if self.sk_params.get('loss') is None:
                self.set_params(loss='sparse_categorical_crossentropy')
        n_classes_ = len(classes_)
        self.set_params(num_cls=self.sk_params.get('num_cls', n_classes_))
        hst = self._fit(x, y, sample_weight=None, **kwargs)
        self.set_params(loss=loss_org)
        return hst

class RBFRegressor(RBFBase, KerasRegressor):
    
    def __init__(self, build_fn=make_model, **sk_params):
        super().__init__(build_fn, **sk_params)
    
    def fit(self, x, y, sample_weight=None, **kwargs):
        ### num_cls
        self.set_params(num_cls=self.sk_params.get('num_cls', 1))
        
        ### activation
        self.set_params(activation=self.sk_params.get('activation', 'linear'))
        
        ### loss
        self.set_params(loss=self.sk_params.get('loss', 'mse'))
        
        hst = self._fit(x, y, sample_weight=None, **kwargs)
        return hst
