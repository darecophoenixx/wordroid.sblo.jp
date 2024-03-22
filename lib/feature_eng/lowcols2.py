'''
Copyright (c) 2018 Norio Tamada
Released under the MIT license
https://github.com/darecophoenixx/wordroid.sblo.jp/blob/master/lib/feature_eng/LICENSE.md
'''

'''
ネガティブサンプリングを導入（行側のみ）

２回最適化を行う

１回目→ネガティブサンプリングを考慮しない
ネガティブサンプリングを考慮しないのでgkernel3でガンマも学習される
user x prod : gkernel3

２回目→ネガティブサンプリングを考慮
１回目のgkernel3で学習されたガンマを利用する
ネガティブサンプリングはgkernelではなく関数で実装されている
user x prod : gkernel
user x neg_user : Lambda

'''

import numpy as np
from math import ceil
import random
import gc

from keras_ex.gkernel import GaussianKernel, GaussianKernel2, GaussianKernel3

from tensorflow.keras.layers import Input, Embedding, LSTM, GRU, Dense, Dropout, Lambda, \
    Conv1D, Conv2D, Conv3D, \
    Conv2DTranspose, \
    AveragePooling1D, \
    MaxPooling1D, MaxPooling2D, MaxPooling3D, \
    GlobalAveragePooling1D, \
    GlobalMaxPooling1D, GlobalMaxPooling2D, \
    concatenate, Flatten, Average, Activation, \
    RepeatVector, Permute, Reshape, Dot, \
    multiply, dot, add, \
    PReLU, \
    Bidirectional, TimeDistributed, \
    SpatialDropout1D, \
    BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import losses
from tensorflow.keras.callbacks import BaseLogger, ProgbarLogger, Callback, History,\
    LearningRateScheduler, EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.constraints import MaxNorm, NonNeg
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras import backend as K



def make_model(num_user=20, num_product=39, num_features=12, num_neg=1,
                reg_gamma=0.0, embeddings_val=0.5, seed=None,
                rscore=None, cscore=None, loss_wgt_neg=0.01,
                gamma=None):
    if rscore is None:
        user_embedding = Embedding(output_dim=num_features, input_dim=num_user,
                                   embeddings_initializer=initializers.RandomUniform(minval=-embeddings_val, maxval=embeddings_val, seed=seed),
                                   embeddings_regularizer=regularizers.l2(reg_gamma),
                                   name='user_embedding', trainable=True)
    else:
        user_embedding = Embedding(output_dim=num_features, input_dim=num_user,
                                   weights=[rscore],
                                   embeddings_regularizer=regularizers.l2(reg_gamma),
                                   name='user_embedding', trainable=True)

    input_user = Input(shape=(1,), name='input_user')
    input_neg_user = Input(shape=(1, num_neg), name='input_neg_user')
    
    embed_user = user_embedding(input_user)
    #print('embed_user >', K.int_shape(embed_user))
    embed_neg_user = user_embedding(input_neg_user)
    #print('embed_neg_user >', K.int_shape(embed_neg_user))
    model_user = Model(input_user, embed_user)
    
    if cscore is None:
        if np.__version__ == '1.16.3': # for kaggle
            print('numpy.__version__ is 1.16.3. use RandomState...')
            rng = np.random.RandomState(seed)
            init_wgt = (rng.random_sample((num_product, num_features)) - 0.5) * 2 * embeddings_val
        else:
            rng = np.random.default_rng(seed)
            init_wgt = (rng.random((num_product, num_features)) - 0.5) * 2 * embeddings_val
    else:
        init_wgt = cscore
    if gamma is None:
        gamma = 1./(init_wgt.var() * init_wgt.shape[1])
    def reshape_embed_prod(embed_prod):
        embed_prod2 = K.sum(K.square(embed_prod), axis=2)
        #print('embed_prod2 >', K.int_shape(embed_prod2))
        embed_prod2 = K.expand_dims(embed_prod2)
        #print('embed_prod2 >', K.int_shape(embed_prod2))
        return embed_prod2
    '''calc prob2'''
    def calc_prob2(x, nn1, nn2):
        embed_prod = x[0] # (None, stack_size, num_features)
        #print('embed_prod (in calc_prob2) >', K.int_shape(embed_prod))
        embed_neg = x[1] # (None, stack_size, num_neg, num_features)
        #print('embed_neg (in calc_prob2) >', K.int_shape(embed_neg))
        embed_prod2 = reshape_embed_prod(embed_prod) # (None, stack_size, 1)
        #print('embed_prod2 (in calc_prob2) >', K.int_shape(embed_prod2))
        embed_prod2 = K.repeat_elements(embed_prod2, nn2, axis=2) # (None, stack_size, num_neg)
        #print('embed_prod2 (in calc_prob2) >', K.int_shape(embed_prod2))
        
        embed_neg2 = K.sum(K.square(embed_neg), axis=3) # (None, stack_size, num_neg)
        #print('embed_neg2 (in calc_prob2) >', K.int_shape(embed_neg2))
        
        embed_prod_x_embed_neg = K.batch_dot(
            K.reshape(embed_prod, (-1, num_features)), 
            K.reshape(embed_neg, (-1, nn2, num_features)), 
            axes=(1,2)) # (None, num_neg)
        #print('embed_prod_x_embed_neg (in calc_prob2) >', K.int_shape(embed_prod_x_embed_neg))
        embed_prod_x_embed_neg = K.reshape(embed_prod_x_embed_neg, (-1, nn1, nn2)) # (None, stack_size, num_neg)
        #print('embed_prod_x_embed_neg (in calc_prob2) >', K.int_shape(embed_prod_x_embed_neg))
        d2 = embed_prod2 + embed_neg2 - 2*embed_prod_x_embed_neg # (None, stack_size, stack_size)
        #print('d2 (in calc_prob2) >', K.int_shape(d2))
        prob = K.exp(-1. * gamma * d2)
        #print('prob (in calc_prob2) >', K.int_shape(prob))
        return prob
    
    '''=== user x neg_user ==='''
    prob3 = Lambda(calc_prob2, name='y_neg_user', arguments={'nn1': 1, 'nn2': num_neg})([embed_user, embed_neg_user])
    #print('prob3 >', K.int_shape(prob3))
    model_prob3 = Model([input_user, input_neg_user], prob3, name='model_prob3')

    '''=== user x prod ==='''
    '''gkernel3'''
    weights1 = [init_wgt, np.log(np.array([gamma]))]
    layer_gk3 = GaussianKernel3(num_product, num_features, name='gkernel3', weights=weights1)
    oup_gk3 = layer_gk3(Flatten()(embed_user))
    oup_gk3 = Activation('linear', name='y')(oup_gk3)
    model_gk3 = Model(input_user, oup_gk3)
    '''gkernel'''
    weights1 = [init_wgt]
    layer_gk1 = GaussianKernel(num_product, num_features, name='gkernel1', weights=weights1, kernel_gamma=gamma)
    oup_gk1 = layer_gk1(Flatten()(embed_user))
    oup_gk1 = Activation('linear', name='y')(oup_gk1)
    model_gk1 = Model(input_user, oup_gk1)
    
    model = Model(input_user, oup_gk3)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    
    model_gamma_fixed = Model(input_user, oup_gk1)
    model_gamma_fixed.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    
    model2 = Model([input_user, input_neg_user], [oup_gk1, prob3])
    model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'],
                  loss_weights={'y': 1.0, 'y_neg_user': loss_wgt_neg})
    model3 = Model([input_user, input_neg_user], [oup_gk1, prob3])
    model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'],
                  loss_weights={'y': 1.0, 'y_neg_user': loss_wgt_neg/10})
    models = {
        'model': model,
        'model_gamma_fixed': model_gamma_fixed,
        'model2': model2,
        'model3': model3,
        'model_user': model_user,
        'model_gk1': model_gk1,
        'model_gk3': model_gk3,
        'model_prob3': model_prob3,
    }
    return models


class Seq(Sequence):
    
    def __init__(self, user_id_list, X_df_values, batch_size, num_neg=1, only_y=False):
        self.user_id_list = list(user_id_list)
        self.X_df_values = X_df_values
        self.num_neg = num_neg
        self.batch_size = batch_size
        self.only_y = only_y
        
        self.len = ceil(len(self.user_id_list) / self.batch_size)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        idx_rng_from, idx_rng_to = (idx * self.batch_size), (idx * self.batch_size + self.batch_size)
        input_user = self.user_id_list[idx_rng_from:idx_rng_to]
        ll = len(input_user)
        if self.only_y:
            X = {
                'input_user': np.array(input_user),
                }
        else:
            try:
                input_neg_user = random.sample(self.user_id_list, self.num_neg*ll)
            except ValueError:
                input_neg_user = random.choices(self.user_id_list, k=self.num_neg*ll)
#            try:
#                input_neg_user = np.random.choice(self.user_id_list, size=self.num_neg*ll, replace=False)
#            except ValueError:
#                input_neg_user = np.random.choice(self.user_id_list, size=self.num_neg*ll, replace=True)
            X = {
                'input_user': np.array(input_user),
                'input_neg_user': np.array(input_neg_user).reshape((ll, 1, self.num_neg)),
                }
            
        y = self.X_df_values[idx_rng_from:idx_rng_to,:]
        return (
                X,
                {'y': y} if self.only_y else {'y': y, 'y_neg_user': np.zeros((ll, self.num_neg))}
               )
    
#    def on_epoch_end(self):
#        gc.collect()


class WD2vec(object):
    
    def __init__(self, X_df):
        self.X_df = X_df
        
    def make_model(self, num_features=12, num_neg=2,
                         reg_gamma=0.0, embeddings_val=0.1, loss_wgt_neg=0.01,
                         seed=None, rscore=None, cscore=None, gamma=None):
        num_user = self.X_df.shape[0]
        num_product = self.X_df.shape[1]
        self.num_features = num_features
        self.num_neg = num_neg
        self.models = make_model(num_user=num_user, num_product=num_product, num_features=num_features, num_neg=num_neg,
                                 reg_gamma=reg_gamma, embeddings_val=embeddings_val, seed=seed, loss_wgt_neg=loss_wgt_neg,
                                 rscore=rscore, cscore=cscore, gamma=gamma)
        return self.models
    
    def get_seq(self, batch_size=32, only_y=False):
        seq = Seq(user_id_list=np.arange(self.X_df.shape[0]),
                  X_df_values=self.X_df.values,
                  batch_size=batch_size,
                  num_neg=self.num_neg, only_y=only_y)
        return seq
    
    def train(self, epochs=5, batch_size=32, verbose=2,
              use_multiprocessing=False, workers=1, shuffle=True,
              callbacks=None, callbacks_add=None, lr0=0.005,
              flag_early_stopping=True, model_seq=['model', 'model2'],
              base=8):
        def lr_schedule(epoch, lrx):
            def reduce(epoch, lr):
                if divmod(epoch,4)[1] == 3:
                    lr *= (1/8)
                elif divmod(epoch,4)[1] == 2:
                    lr *= (1/4)
                elif divmod(epoch,4)[1] == 1:
                    lr *= (1/2)
                elif divmod(epoch,4)[1] == 0:
                    pass
                return lr

            lra = lr0
            epoch1 = int(epochs / 8)
            epoch2 = epoch1
            epoch3 = epoch1
            epoch4 = epoch1

            if epoch1+epoch2+epoch3+epoch4 <= epoch:
                epoch = epoch - (epoch1+epoch2+epoch3+epoch4)
                lra = lra / 2

            if epoch<epoch1:
                lr = lra
                lr = reduce(epoch, lr)
            elif epoch<epoch1+epoch2:
                lr = lra/2
                lr = reduce(epoch, lr)
            elif epoch<epoch1+epoch2+epoch3:
                lr = lra/4
                lr = reduce(epoch, lr)
            elif epoch<epoch1+epoch2+epoch3+epoch4:
                lr = lra/8
                lr = reduce(epoch, lr)
            else:
                lr = lra/64

            if verbose == 0:
                pass
            else:
                print('Learning rate: ', lr)
            return lr
        if callbacks is None:
            lr_scheduler = LearningRateScheduler(lr_schedule)
            eraly_stopping = EarlyStopping(monitor='loss', patience=3)
            callbacks = [eraly_stopping, lr_scheduler]
            #callbacks = [lr_scheduler]
        if callbacks_add is not None:
            callbacks = callbacks + callbacks_add
        
        if model_seq[0] == 'model':
            seq = self.get_seq(batch_size=batch_size, only_y=True)
            model_using = self.models['model']
        elif model_seq[0] == 'model_gamma_fixed':
            seq = self.get_seq(batch_size=batch_size, only_y=True)
            model_using = self.models['model_gamma_fixed']
        elif model_seq[0] == 'model2':
            seq = self.get_seq(batch_size=batch_size)
            model_using = self.models['model2']
        else:
            print('model_seq[0] >', model_seq[0])
            raise Exception('bad model type')
        res = model_using.fit(seq, steps_per_epoch=len(seq),
                        epochs=epochs,
                        verbose=verbose,
                        shuffle=shuffle,
                        use_multiprocessing=use_multiprocessing, workers=workers,
                        callbacks=callbacks)
        lr2 = res.history['lr'][-1]
        res2 = self.train2(epochs=int(epochs/2),
                      batch_size=batch_size,
                      verbose=verbose,
                      use_multiprocessing=use_multiprocessing,
                      shuffle=shuffle,
                      workers=workers,
                      callbacks=None,
                      lr0=lr2, base=base, flag_early_stopping=flag_early_stopping)
        return res, res2
    
    def train2(self, epochs=5, batch_size=32, verbose=1,
              use_multiprocessing=False, workers=1, shuffle=True,
              callbacks=None, lr0=0.001, base=8,
              flag_early_stopping=True, use_model='model2'):
        def lr_schedule(epoch, lr, epochs=epochs, lr0=lr0, base=base, verbose=verbose):
            b = 1 / np.log((epochs-1+np.exp(1)))
            a = 1 / np.log((epoch+np.exp(1))) / (1-b) - b/(1-b)
            lr = a*(1-1/base)*lr0 + lr0/base
            if verbose == 0:
                pass
            else:
                print('Learning rate: ', lr)
            return lr
        if callbacks is None:
            lr_scheduler = LearningRateScheduler(lr_schedule)
            early_stopping = EarlyStopping(monitor='loss', patience=3)
            callbacks = [early_stopping, lr_scheduler] if flag_early_stopping else [lr_scheduler]
            #callbacks = [lr_scheduler]
        
        gamma = self.gamma
        if 0 < verbose:
            print('applied gamma >', gamma)
        rscore = self.wgt_row
        cscore = self.wgt_col
        K.clear_session()
        self.make_model(num_features=self.num_features, num_neg=self.num_neg,
                         reg_gamma=0.0, embeddings_val=0.1, loss_wgt_neg=0.01,
                         rscore=rscore, cscore=cscore, gamma=gamma)
        
        model_using = self.models[use_model]
        seq = self.get_seq(batch_size=batch_size)
        res = model_using.fit(seq, steps_per_epoch=len(seq),
                        epochs=epochs,
                        use_multiprocessing=use_multiprocessing,
                        shuffle=shuffle,
                        verbose=verbose,
                        callbacks=callbacks)
        return res
    
    def get_wgt_byrow(self):
        wgt = self.models['model'].get_layer('user_embedding').get_weights()[0]
        return wgt
    wgt_row = property(get_wgt_byrow)
    
    def get_wgt_bycol(self):
        wgt = self.models['model'].get_layer('gkernel3').get_weights()[0]
        return wgt
    wgt_col = property(get_wgt_bycol)
    
    def get_gamma(self):
        logged_gamma = self.models['model'].get_layer('gkernel3').get_weights()[1][0]
        gamma = np.exp(logged_gamma)
        return gamma
    gamma = property(get_gamma)



