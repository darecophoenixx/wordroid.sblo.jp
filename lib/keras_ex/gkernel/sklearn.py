'''
Copyright (c) 2019 Norio Tamada
Released under the MIT license
https://github.com/darecophoenixx/wordroid.sblo.jp/blob/master/lib/keras_ex/gkernel/LICENSE.md
'''

from . import GaussianKernel, GaussianKernel2, GaussianKernel3


import copy

import numpy as np
#from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.optimizers import RMSprop, Adam
from keras.initializers import glorot_uniform
from keras.models import Sequential, Model
from keras.layers import Input, Dense
from keras import regularizers
from keras import backend as K

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, ClassifierMixin


# def make_model_gkernel1(
#     nn=4, num_lm=2,
#     random_state=None, lm=None, gamma=None):
#     '''
#     just activation
#     '''
#     inp = Input(shape=(nn,), name='inp')
#     oup = inp
    
#     if lm is None:
#         rs = np.random.RandomState(random_state)
#         lm = rs.random_sample((num_lm, nn))
#     if gamma is None:
#         gamma = 1/(2*np.sqrt(nn/6)*2)
#     weights = [np.log(np.array([gamma]))]
#     oup = GaussianKernel2(lm, weights=weights, name='gkernel', trainable=False)(oup)
    
#     model = Model(inp, oup, name='model_gkernel')
#     return lm, model

# def make_model_gkernel2(
#     nn=4, num_lm=2,
#     random_state=None, lm=None, gamma=None):
    
#     inp = Input(shape=(nn,), name='inp')
#     oup = inp
    
#     if lm is None:
#         rs = np.random.RandomState(random_state)
#         lm = rs.random_sample((num_lm, nn))
#     if gamma is None:
#         gamma = 1/(2*np.sqrt(nn/6)*2)
#     weights = [np.log(np.array([gamma]))]
#     oup = GaussianKernel2(lm, weights=weights, name='gkernel')(oup)
    
#     model = Model(inp, oup, name='model_gkernel')
#     return lm, model

# def make_model_gkernel3(
#     nn=4, num_lm=2,
#     random_state=None, lm=None, gamma=None):

#     inp = Input(shape=(nn,), name='inp')
#     oup = inp
    
#     if lm is None:
#         rs = np.random.RandomState(random_state)
#         lm = rs.random_sample((num_lm, nn))
#     if gamma is None:
#         gamma = 1/(2*np.sqrt(nn/6)*2)
#     weights = [lm, np.log(np.array([gamma]))]
#     oup = GaussianKernel3(num_landmark=num_lm, num_feature=nn, weights=weights, name='gkernel')(oup)
    
#     model = Model(inp, oup, name='model_gkernel')
#     return lm, model

# def make_model_out(
#     num_lm=2, num_cls=3,
#     activation='softmax',
#     reg_l1=0.0, reg_l2=0.0,
#     random_state=None
# ):
#     inp = Input(shape=(num_lm,), name='inp')
#     oup = Dense(num_cls,
#                 activation=activation,
#                 kernel_initializer=glorot_uniform(random_state),
#                 kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2),
#                 bias_regularizer=regularizers.l1_l2(reg_l1, reg_l2)
#                )(inp)
    
#     model = Model(inp, oup, name='model_out')
#     return model

# DEFAULT_LR = 0.05
# DEFAULT_LOSS = 'categorical_crossentropy'
# DEFAULT_EPOCHS_WARMUP = 3
# DEFAULT_BATCHSIZE_INTHEMIDDLE = 32
# def make_model(
#     make_model_gkernel=make_model_gkernel2,
#     make_model_out=make_model_out,
#     reg_l1=0.0, reg_l2=0.0,
#     nn=2,
#     num_lm=2, lm=None, gamma=None,
#     random_state=None,
#     num_cls=2, activation='softmax',
#     opt=0.02, lr=DEFAULT_LR,
#     loss=DEFAULT_LOSS,
#     session_clear=False,
#     #gkernel_multipliers=1.0,
#     batch_size_middle=DEFAULT_BATCHSIZE_INTHEMIDDLE,
#     lm_select_from_x=None,
#     tol=None,
#     epochs_warmup=DEFAULT_EPOCHS_WARMUP
# ):
#     if session_clear:
#         K.clear_session()
    
#     lm, model_gkernel = make_model_gkernel(
#         nn=nn, num_lm=num_lm,
#         random_state=random_state, lm=lm, gamma=gamma
#     )
#     model_gkernel.trainable = False
    
#     model_out = make_model_out(
#         num_lm=num_lm, num_cls=num_cls,
#         activation=activation,
#         reg_l1=reg_l1, reg_l2=reg_l2,
#         random_state=random_state
#     )
    
#     inp = model_gkernel.inputs[0]
#     oup = model_gkernel(inp)
#     oup = model_out(oup)
    
#     model = Model(inp, oup)
#     opt = Adam(lr)
#     model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
#     return model


# class RBFBase(object):
    
#     def _fit(self, x, y, sample_weight=None, **kwargs):
#         sk_params_org = copy.deepcopy(self.sk_params)
#         early_stopping_patience = 7
        
#         ### epochs_warmup
#         sk_params_org.update({'epochs_warmup': self.sk_params.get('epochs_warmup', DEFAULT_EPOCHS_WARMUP)})
        
#         ### nn
#         nn = x.shape[1]
#         sk_params_org.update({'nn': self.sk_params.get('nn')})
#         self.set_params(nn=nn)
        
#         ### learning_rate
#         if self.sk_params.get('lr') is None:
#             lr = DEFAULT_LR # default lr
#             sk_params_org.update({'lr': None})
#             self.set_params(lr=lr)
#         else:
#             lr = self.sk_params['lr'] # for later using
        
#         ### gamma
#         if self.sk_params.get('gamma') == 'scale':
#             sk_params_org.update({'gamma': 'scale'})
#             self.set_params(gamma=1 / (nn * x.var()))
        
#         ### tol
#         tol = self.sk_params.get('tol', float(np.sqrt(np.finfo(np.float32).eps)))
        
#         ### callbacks
#         if self.sk_params.get('callbacks', None) is None:
#             lr_reducer = ReduceLROnPlateau(monitor='loss',
#                                factor=1/2,
#                                verbose=0,
#                                cooldown=0,
#                                min_delta=tol*2,
#                                patience=5,
#                                min_lr=lr/64)
#             early_stopping = EarlyStopping(
#                 monitor='loss',
#                 patience=early_stopping_patience,
#                 min_delta=tol,
#                 restore_best_weights=True)
#             callbacks0 = [early_stopping]
#             callbacks = callbacks0 + [lr_reducer]
#             sk_params_org.update({'callbacks': None})
#             self.set_params(callbacks=callbacks)
#         else:
#             callbacks0 = self.sk_params['callbacks']
#             callbacks = self.sk_params['callbacks']
        
#         ### lm_select_from_x
#         if self.sk_params.get('lm_select_from_x'):
#             random_state = self.sk_params.get('random_state')
#             rs = np.random.RandomState(random_state)
#             lm = x[rs.choice(np.arange(x.shape[0]), self.sk_params['num_lm'], replace=False)]
#             sk_params_org.update({'lm': self.sk_params.get('lm')})
#             self.set_params(lm=lm)
        
        
        
#         epochs = kwargs.get('epochs', self.sk_params.get('epochs', 1))
#         batch_size = kwargs.get('batch_size', self.sk_params.get('batch_size', 32))
        
#         ##############################
#         # === warm up train
#         ##############################
#         epochs_warmup = self.sk_params.get('epochs_warmup', DEFAULT_EPOCHS_WARMUP)
#         hst_all = {}
#         kwargs.update({'epochs': epochs_warmup})
#         if self.__class__.__name__ == 'RBFRegressor':
#             kwargs.update({'sample_weight': sample_weight})
#             hst = super().fit(x, y, **kwargs)
#         else:
#             hst = super().fit(x, y, sample_weight, **kwargs)
#         hst_all = self.update_hst_all(hst_all, hst)
        
#         ##############################
#         # === start train
#         ##############################
#         model_gkernel = self.model.get_layer('model_gkernel')
#         model_gkernel.trainable = True
#         opt = Adam(lr)
#         loss = self.sk_params.get('loss', DEFAULT_LOSS)
#         self.model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
        
#         kwargs.update({'epochs': epochs}) # modosu
#         fit_args = copy.deepcopy(self.filter_sk_params(Sequential.fit))
#         fit_args.update(kwargs)
#         fit_args.update({'sample_weight': sample_weight})
#         fit_args['batch_size'] = batch_size
#         fit_args['epochs'] = epochs
#         hst = self.model.fit(x, y, **fit_args)
#         hst_all = self.update_hst_all(hst_all, hst)
        
#         def lr_schedule2(epoch):
#             lr1 = lr / 2
#             #print('lr >', lr1)
#             return lr1
#         def lr_schedule4(epoch):
#             lr1 = lr / 4
#             #print('lr >', lr1)
#             return lr1
#         def lr_schedule8(epoch):
#             lr1 = lr / 8
#             #print('lr >', lr1)
#             return lr1
# #        def lr_schedule2(epoch):
# #            div, _ = divmod(epoch,4)
# #            if divmod(div,2)[1] == 1:
# #                lr1 = lr * (1/2)
# #            elif divmod(div,2)[1] == 0:
# #                lr1 = lr * 1
# #            lr1 = lr1 / 2
# #            #print('lr >', lr1)
# #            return lr1
# #        def lr_schedule4(epoch):
# #            div, _ = divmod(epoch,4)
# #            if divmod(div,2)[1] == 1:
# #                lr1 = lr * (1/2)
# #            elif divmod(div,2)[1] == 0:
# #                lr1 = lr * 1
# #            lr1 = lr1 / 4
# #            #print('lr >', lr1)
# #            return lr1
# #        def lr_schedule8(epoch):
# #            div, _ = divmod(epoch,4)
# #            if divmod(div,2)[1] == 1:
# #                lr1 = lr * (1/2)
# #            elif divmod(div,2)[1] == 0:
# #                lr1 = lr * 1
# #            lr1 = lr1 / 8
# #            #print('lr >', lr1)
# #            return lr1
        
#         # 2
#         lr_scheduler = LearningRateScheduler(lr_schedule2)
#         callbacks = callbacks0 + [lr_scheduler]
#         fit_args['callbacks'] = callbacks
#         fit_args['batch_size'] = batch_size * 2
#         fit_args['epochs'] = epochs * 2
#         hst = self.model.fit(x, y, **fit_args)
#         hst_all = self.update_hst_all(hst_all, hst)
#         # 3
#         lr_scheduler = LearningRateScheduler(lr_schedule4)
#         callbacks = callbacks0 + [lr_scheduler]
#         fit_args['callbacks'] = callbacks
#         fit_args['batch_size'] = batch_size * 4
#         fit_args['epochs'] = epochs * 4
#         hst = self.model.fit(x, y, **fit_args)
#         hst_all = self.update_hst_all(hst_all, hst)
#         # 4
#         lr_scheduler = LearningRateScheduler(lr_schedule8)
#         callbacks = callbacks0 + [lr_scheduler]
#         fit_args['callbacks'] = callbacks
#         fit_args['batch_size'] = batch_size * 8
#         fit_args['epochs'] = epochs * 8
#         hst = self.model.fit(x, y, **fit_args)
#         hst_all = self.update_hst_all(hst_all, hst)
        
        
#         early_stopping = EarlyStopping(
#             monitor='loss',
#             patience=early_stopping_patience,
#             min_delta=tol/2,
#             restore_best_weights=True)
#         callbacks0 = [early_stopping]
#         # 2
#         lr_scheduler = LearningRateScheduler(lr_schedule2)
#         callbacks = callbacks0 + [lr_scheduler]
#         fit_args['callbacks'] = callbacks
#         fit_args['batch_size'] = batch_size * 8
#         fit_args['epochs'] = epochs * 4
#         hst = self.model.fit(x, y, **fit_args)
#         hst_all = self.update_hst_all(hst_all, hst)
#         # 3
#         lr_scheduler = LearningRateScheduler(lr_schedule4)
#         callbacks = callbacks0 + [lr_scheduler]
#         fit_args['callbacks'] = callbacks
#         fit_args['batch_size'] = batch_size * 8
#         fit_args['epochs'] = epochs * 8
#         hst = self.model.fit(x, y, **fit_args)
#         hst_all = self.update_hst_all(hst_all, hst)
#         # 4
#         lr_scheduler = LearningRateScheduler(lr_schedule8)
#         callbacks = callbacks0 + [lr_scheduler]
#         fit_args['callbacks'] = callbacks
#         fit_args['batch_size'] = batch_size * 8
#         fit_args['epochs'] = epochs * 16
#         hst = self.model.fit(x, y, **fit_args)
#         hst_all = self.update_hst_all(hst_all, hst)
        
#         self.set_params(**sk_params_org)
#         return hst_all
    
#     def update_hst_all(self, hst_all, hst):
#         for k in hst.history:
#             hst_all.setdefault(k, [])
#             hst_all[k].extend(hst.history[k])
#         return hst_all
        
#     def current_gamma(self):
#         for ew in self.model.layers[1].layers[1].get_weights():
#             if len(ew.shape)==1:
#                 c_gamma = ew[0]
#         return np.exp(c_gamma)
    
#     def current_lm(self):
#         model_gkernel = self.model.get_layer('model_gkernel')
#         layer_gkernel = model_gkernel.get_layer('gkernel')
#         try:
#             lm = layer_gkernel.landmarks
#         except:
#             lm = layer_gkernel.get_weights()[0]
#         return lm


# class RBFClassifier(RBFBase, KerasClassifier):
#     """RBF kernel Classification.
    
#     Parameters
#     ----------
#     num_lm : int, optional (default=2)
#         number of landmarks
#     lm : array shape (num_lm, num_features), optional (default=None)
#         initial landmarks
#         if set make_model_gkernel = make_model_gkernel2, lm is not learned, so it is not learned by default
#     lm_select_from_x : bool, optional (default=False)
#         if True, lm are selected from input x
#         even if lm is given, it is ignored
#     make_model_gkernel : function, optional (default=make_model_gkernel2)
#         function to configure RBF kernel
#         make_model_gkernel2 : use fixed landmarks, gamma is learned
#         make_model_gkernel3 : train landmarks and gamma
#         make_model_gkernel1 : use fixed landmarks and gamma, just activation
#     make_model_out : function, optional (default=make_model_out)
#         dense layer just before output
#     gamma : float or str, optional (default=None)
#         RBF kernel parameter
#         if using make_model_gkernel2 or make_model_gkernel3 for make_model_gkernel, gamma is trained
#         gamma='scale', use 1 / (nn * x.var())
#         gamma=None, use...
#     session_clear : bool, optional (default=False)
#         execute K.clear_session() or not
#     reg_l1 : float, optional (default=0.0)
#         regularization parameter.
#     reg_l2 : float, optional (default=0.0)
#         regularization parameter.
#     tol : float, optional (default=float(np.sqrt(np.finfo(np.float32).eps)))
#         tolerance for stopping criterion
#         tol halves in the middle of learning
#     activation : str, optional (default='softmax')
#         activation for output
#     loss : str, optional (default='categorical_crossentropy')
#         loss function
#     lr : float, optional (default=0.05)
#         learning rate
#         lr decreases with each stage of learning (learning rate annealing)
#     random_state : int, RandomState instance or None, optional (default=None)
#         The seed of the pseudo random number generator
    
#     verbose : int, default: 1
#         param for keras
#     epochs : int, optional (default=1)
#         param for keras
#         epochs increases with each stage of learning
#     epochs_warmup : int, optional (default=10)
#         epochs for warm-up start
#         set to 0, no warm-up
#         warm-up means that set model_gkernel.trainable=False, just train dense layer
#     batch_size : int, optional (default=32)
#         param for keras
#         batch_size at initial stage
#         doubles every stage. In the end of the stages, batch_size will be 64 times.
    
#     """
    
#     def __init__(self, build_fn=make_model, **sk_params):
#         super().__init__(build_fn, **sk_params)
    
#     def fit(self, x, y, sample_weight=None, **kwargs):
#         loss_org = self.sk_params.get('loss', None)
#         ### num_cls
#         y = np.array(y)
#         if len(y.shape) == 2 and y.shape[1] > 1:
#             classes_ = np.arange(y.shape[1])
#             if 'loss' in self.sk_params and self.sk_params.get('loss') is None:
#                 # use default 'categorical_crossentropy'
#                 del self.sk_params['loss']
#         elif (len(y.shape) == 2 and y.shape[1] == 1) or len(y.shape) == 1:
#             classes_ = np.unique(y)
#             if self.sk_params.get('loss') is None:
#                 self.set_params(loss='sparse_categorical_crossentropy')
#         n_classes_ = len(classes_)
#         self.set_params(num_cls=self.sk_params.get('num_cls', n_classes_))
#         hst = self._fit(x, y, sample_weight=sample_weight, **kwargs)
#         self.set_params(loss=loss_org)
#         return hst

# class RBFRegressor(RBFBase, KerasRegressor):
    
#     def __init__(self, build_fn=make_model, **sk_params):
#         super().__init__(build_fn, **sk_params)
    
#     def fit(self, x, y, sample_weight=None, **kwargs):
#         ### num_cls
#         self.set_params(num_cls=self.sk_params.get('num_cls', 1))
        
#         ### activation
#         self.set_params(activation=self.sk_params.get('activation', 'linear'))
        
#         ### loss
#         self.set_params(loss=self.sk_params.get('loss', 'mse'))
        
#         hst = self._fit(x, y, sample_weight=sample_weight, **kwargs)
#         return hst

class SimpleRBFClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self,
                 num_lm=2,
                 init_lm=None,
                 init_gamma=None,
                 logit=None,
                 random_state=None
                ):
        if logit is None:
            self.logit = LogisticRegression(penalty='none', solver='lbfgs', random_state=random_state)
        else:
            self.logit = logit
        self.num_lm = num_lm
        self.init_lm = init_lm
        self.init_gamma = init_gamma
        self.random_state = random_state
    
    def _initialize_lm(self, X, y, sample_weight=None):
        if self.init_lm is None:
            self.init_lm = KMeans(n_clusters=self.num_lm, random_state=self.random_state)
        if hasattr(self.init_lm, '__array__'):
            self.lm = self.init_lm
        elif self.init_lm == 'select_from_x':
            rs = np.random.RandomState(self.random_state)
            self.lm = X[rs.choice(np.arange(X.shape[0]), self.num_lm, replace=False)]
        else:
            '''fit kmeans'''
            self.init_lm.fit(X, sample_weight=sample_weight)
            self.lm = self.init_lm.cluster_centers_
    
    def calc_gamma_scale1(self):
        return 1 / (self.lm.var()*self.lm.shape[1])
    
    def calc_gamma_d2_median(self):
        d2 = -2*self.lm.dot(self.lm.T) + 2*np.square(self.lm).sum(axis=1)
        d2_median = np.median([d2[ii, jj] for ii in range(d2.shape[0]) for jj in range(d2.shape[0]) if ii<jj])
        return d2_median
    
    def _initialize_gamma(self, X, y, sample_weight=None):
        self.gamma = None
        if self.init_gamma is None:
            self.gamma = self.calc_gamma_scale1()
        elif isinstance(self.init_gamma, str):
            if self.init_gamma == 'scale1':
                self.gamma = self.calc_gamma_scale1()
            elif self.init_gamma == 'scale2':
                d2_median = self.calc_gamma_d2_median()
                self.gamma = 1 / (2 * d2_median)
            elif self.init_gamma == 'scale3':
                d2_median = self.calc_gamma_d2_median()
                self.gamma = 1 / (2 * d2_median / self.lm.shape[1])
        else:
            self.gamma = self.init_gamma
        
    def _initialize(self, X, y, sample_weight=None):
        self._initialize_lm(X, y, sample_weight)
        self._initialize_gamma(X, y, sample_weight)
    
    def _get_n_classes(self, y):
        y = np.array(y)
        if len(y.shape) == 2 and y.shape[1] > 1:
            classes_ = np.arange(y.shape[1])
        elif (len(y.shape) == 2 and y.shape[1] == 1) or len(y.shape) == 1:
            classes_ = np.unique(y)
        n_classes_ = len(classes_)
        return classes_, n_classes_
    
    def fit(self, X, y, sample_weight=None, **kwargs):
        self.classes_, self.n_classes_ = self._get_n_classes(y)
        self._initialize(X, y, sample_weight)
        X2 = self.calc_gauss(X)
        self.logit.fit(X2, y, sample_weight=sample_weight)
        return self
    
    def predict_proba(self, x, **kwargs):
        x2 = self.calc_gauss(x)
        return self.logit.predict_proba(x2)
    
    def predict(self, x, **kwargs):
        x2 = self.calc_gauss(x)
        return self.logit.predict(x2)
    
    def calc_gauss(self, x, lm=None, gamma=None):
        if lm is None:
            lm = self.lm
        if gamma is None:
            gamma = self.gamma
        x_2 = np.square(x).sum(axis=1).reshape((-1,1))
        lm_2 = np.square(lm).sum(axis=1).reshape((1,-1))
        d_2 = -2*x.dot(lm.T)
        d_2 += x_2
        d_2 += lm_2
        np.multiply(-gamma, d_2, out=d_2)
        np.exp(d_2, out=d_2)
        return d_2
    
    def calc_coef(self):
        m = LinearRegression()
        if hasattr(self.logit, 'coef_'):
            coef_ = self.logit.coef_.mean(axis=0)
            m.fit(self.lm, coef_.flatten())
        elif hasattr(self.logit, 'feature_importances_'):
            m.fit(self.lm, self.logit.feature_importances_.flatten())
        elif hasattr(self.logit, 'importances_'):
            m.fit(self.lm, self.logit.importances_.flatten())
        else:
            raise AttributeError('self.logit does not have coef_ family...')
        return m.coef_
    feature_importances_ = property(calc_coef)
