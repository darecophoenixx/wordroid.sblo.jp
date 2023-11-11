'''
Copyright (c) 2018 Norio Tamada
Released under the MIT license
https://github.com/darecophoenixx/wordroid.sblo.jp/blob/master/lib/feature_eng/LICENSE.md
'''

'''
gkernelを使わない
変数間の相関係数行列を考慮
'''

import numpy as np
from math import ceil
import random




from keras_ex.gkernel import GaussianKernel, GaussianKernel2, GaussianKernel3

from tensorflow.keras.layers import Input, Embedding, LSTM, GRU, Dense, Dropout, Lambda, \
    Conv1D, Conv2D, Conv3D, \
    Conv2DTranspose, \
    AveragePooling1D, \
    MaxPooling1D, MaxPooling2D, MaxPooling3D, \
    GlobalAveragePooling1D, \
    GlobalMaxPooling1D, GlobalMaxPooling2D, \
    LocallyConnected1D, LocallyConnected2D, \
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


SIGMA2 = 0.2**2
def make_model(num_user=20, num_product=10,
               num_neg=2, stack_size=5, num_features=8, gamma=0.0,
               rscore=None, cscore=None, seed=None,
               embeddings_val=0.1, sigma2=SIGMA2, loss_wgt_neg=0.001):

    if rscore is None:
        user_embedding = Embedding(output_dim=num_features, input_dim=num_user,
                                   embeddings_initializer=initializers.RandomUniform(minval=-embeddings_val, maxval=embeddings_val, seed=seed),
                                   embeddings_regularizer=regularizers.l2(gamma),
                                   name='user_embedding', trainable=True)
    else:
        user_embedding = Embedding(output_dim=num_features, input_dim=num_user,
                                   weights=[rscore],
                                   embeddings_regularizer=regularizers.l2(gamma),
                                   name='user_embedding', trainable=True)
    if cscore is None:
        prod_embedding = Embedding(output_dim=num_features, input_dim=num_product,
                                   embeddings_initializer=initializers.RandomUniform(minval=-embeddings_val, maxval=embeddings_val),
                                   embeddings_regularizer=regularizers.l2(gamma),
                                   name='prod_embedding', trainable=True)
    else:
        prod_embedding = Embedding(output_dim=num_features, input_dim=num_product,
                                   weights=[cscore],
                                   embeddings_regularizer=regularizers.l2(gamma),
                                   name='prod_embedding', trainable=True)

    input_user = Input(shape=(1,), name='input_user')
    input_neg_user = Input(shape=(1, num_neg), name='input_neg_user')
    input_prod = Input(shape=(1, num_product,), name='input_prod')
    input_prod_cor = Input(shape=(1,), name='input_prod_cor')
    input_neg_prod = Input(shape=(1,), name='input_neg_prod')
    input_neg_prod2 = Input(shape=(1, num_neg), name='input_neg_prod2')

    embed_user = user_embedding(input_user)
    #print('embed_user >', K.int_shape(embed_user))
    embed_neg_user = user_embedding(input_neg_user)
    #print('embed_neg_user >', K.int_shape(embed_neg_user))
    embed_prod = prod_embedding(input_prod)
    #print('embed_prod >', K.int_shape(embed_prod))
    embed_prod_cor = prod_embedding(input_prod_cor)
    
    embed_neg_prod = prod_embedding(input_neg_prod)
    #print('embed_neg_prod >', K.int_shape(embed_neg_prod))
    embed_neg_prod2 = prod_embedding(input_neg_prod2)

    model_user = Model(input_user, embed_user)
    model_neg_user = Model(input_neg_user, embed_neg_user)
    model_prod = Model(input_prod, embed_prod)
    model_neg_prod = Model(input_neg_prod, embed_neg_prod)

    input_user_vec = Input(shape=(1, num_features), name='input_user_vec')
    input_neg_user_vec = Input(shape=(1, num_neg, num_features), name='input_neg_user_vec')
    input_prod_vec = Input(shape=(1, num_product, num_features), name='input_prod_vec')
    input_neg_prod_vec = Input(shape=(1, num_features), name='input_neg_prod_vec')
    input_neg_prod2_vec = Input(shape=(1, num_neg, num_features), name='input_neg_prod2_vec')

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
            K.reshape(embed_prod, (-1, num_features)), # (None*stack_size, num_features)
            K.reshape(embed_neg, (-1, nn2, num_features)), # (None*stack_size, num_neg, num_features)
            axes=(1,2)) # (None, num_neg)
        #print('embed_prod_x_embed_neg (in calc_prob2) >', K.int_shape(embed_prod_x_embed_neg))
        embed_prod_x_embed_neg = K.reshape(embed_prod_x_embed_neg, (-1, nn1, nn2)) # (None, stack_size, num_neg)
        #print('embed_prod_x_embed_neg (in calc_prob2) >', K.int_shape(embed_prod_x_embed_neg))
        d2 = embed_prod2 + embed_neg2 - 2*embed_prod_x_embed_neg # (None, stack_size, stack_size)
        #print('d2 (in calc_prob2) >', K.int_shape(d2))
        prob = K.exp(-1./(num_features*sigma2) * d2)
        #print('prob (in calc_prob2) >', K.int_shape(prob))
        return prob
    '''user x prod'''
    prob1 = Lambda(calc_prob2, name='calc_prob1', arguments={'nn1': 1, 'nn2': num_product})([embed_user, embed_prod])
    #print('prob1 >', K.int_shape(prob1))
    model_prob1 = Model([input_user, input_prod], prob1, name='model_prob1')
    
    prob1_cnfm = Lambda(calc_prob2, arguments={'nn1': 1, 'nn2': num_product})([input_user_vec, input_prod_vec])
    model_prob1_cnfm = Model([input_prod_vec, input_user_vec], prob1_cnfm, name='model_prob1_cnfm')
    
    '''prod_cor x prod'''
    prob_cor = Lambda(calc_prob2, name='calc_prob_cor', arguments={'nn1': 1, 'nn2': num_product})([embed_prod_cor, embed_prod])
    #print('prob_cor >', K.int_shape(prob_cor))
    model_prob_cor = Model([input_prod_cor, input_prod], prob_cor, name='model_prob_cor')

    '''neg_prod x neg_prod2'''
    prob2 = Lambda(calc_prob2, name='calc_prob2', arguments={'nn1': 1, 'nn2': num_neg})([embed_neg_prod, embed_neg_prod2])
    #print('prob2 >', K.int_shape(prob2))
    model_prob2 = Model([input_neg_prod, input_neg_prod2], prob2, name='model_prob2')
    
#    prob2_cnfm = Lambda(calc_prob2, name='calc_prob2_cnfm', arguments={'nn1': stack_size, 'nn2': num_neg})([input_prod_vec, input_neg_prod_vec])
#    model_prob2_cnfm = Model([input_prod_vec, input_neg_prod_vec], prob2_cnfm, name='model_prob2_cnfm')
    
    '''user x neg_user'''
    prob3 = Lambda(calc_prob2, name='calc_prob3', arguments={'nn1': 1, 'nn2': num_neg})([embed_user, embed_neg_user])
    #print('prob3 >', K.int_shape(prob3))
    model_prob3 = Model([input_user, input_neg_user], prob3, name='model_prob3')
    
    prob3_cnfm = Lambda(calc_prob2, name='calc_prob3_cnfm', arguments={'nn1': 1, 'nn2': num_neg})([input_user_vec, input_neg_user_vec])
    model_prob3_cnfm = Model([input_user_vec, input_neg_user_vec], prob3_cnfm, name='model_prob3_cnfm')
    
    
    #print('prob >', K.int_shape(prob))
    #print('Flatten()(prob2) >', K.int_shape(Flatten()(prob2)))
    prob = Flatten(name='y')(prob1)
    #print('prob >', K.int_shape(prob))
    
    model = Model([input_user, input_prod], prob)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    
    neg_prob = concatenate([prob3, prob2], axis=2)
    neg_prob = Flatten(name='neg_y')(neg_prob)
#    model2 = Model([input_user, input_prod, input_neg_user, input_neg_prod, input_neg_prod2], [prob, neg_prob])
#    model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'],
#                  loss_weights={'y': 1.0, 'neg_y': loss_wgt_neg})
    
    cor_prob = Flatten(name='cor_y')(prob_cor)
    neg_prob = Flatten(name='neg_y')(prob3)
    model2 = Model([input_user, input_prod, input_prod_cor, input_neg_user], [prob, neg_prob, cor_prob])
    model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'],
                  loss_weights={'y': 1.0, 'neg_y': loss_wgt_neg, 'cor_y': 1.0})
    model3 = Model([input_user, input_prod, input_prod_cor], [prob, cor_prob])
    model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'],
                  loss_weights={'y': 1.0, 'cor_y': 1.0})
    models = {
        'model': model,
        'model2': model2,
        'model3': model3,
        'model_user': model_user,
        'model_neg_user': model_neg_user,
        'model_prod': model_prod,
        'model_neg_prod': model_neg_prod,
        'model_prob1': model_prob1,
        'model_prob2': model_prob2,
        'model_prob3': model_prob3,
        'model_prob_cor': model_prob_cor,
        'model_prob1_cnfm': model_prob1_cnfm,
        #'model_prob2_cnfm': model_prob2_cnfm,
        'model_prob3_cnfm': model_prob3_cnfm,
    }
    return models




class Seq(Sequence):
    
    def __init__(self, user_id_list, X_df_values, cor_mat, batch_size, num_neg=1):
        self.user_id_list = list(user_id_list)
        self.X_df_values = X_df_values.astype(np.float32)
        self.num_neg = num_neg
        self.batch_size = batch_size
        self.cor_mat = cor_mat
        
        self.len = ceil(len(self.user_id_list) / self.batch_size)
        self.prod_id_list = list(range(self.X_df_values.shape[1]))
        self.prods_idx = np.arange(self.X_df_values.shape[1]).reshape(1, self.X_df_values.shape[1])
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        bs = self.batch_size
        idx_rng_from, idx_rng_to = (idx*bs), ((idx*bs+bs) if (idx*bs+bs)<len(self.user_id_list) else len(self.user_id_list))
        input_user = self.user_id_list[idx_rng_from:idx_rng_to]
        ll = len(input_user)
        try:
            input_neg_user = random.sample(self.user_id_list, ll*self.num_neg)
        except ValueError:
            input_neg_user = random.choices(self.user_id_list, ll*self.num_neg)
        input_neg_prod = random.choices(self.prod_id_list, k=ll)
        input_neg_prod2 = random.choices(self.prod_id_list, k=ll*self.num_neg)
        input_prod_cor = np.random.choice(self.prods_idx[0,:], size=(ll,1))
        
        y = self.X_df_values[idx_rng_from:idx_rng_to,:]
        #neg_y = np.zeros((ll, self.num_neg*2))
        neg_y = np.zeros((ll, self.num_neg))
        cor_y = self.cor_mat[input_prod_cor[:,0]]
        return (
                {
                'input_user': np.array(input_user),
                'input_neg_user': np.array(input_neg_user).reshape((ll, 1, self.num_neg)),
                'input_prod': np.tile(self.prods_idx, reps=(ll,1,1)),
                'input_neg_prod': np.array(input_neg_prod).reshape((ll, 1)),
                'input_prod_cor': input_prod_cor,
                'input_neg_prod2': np.array(input_neg_prod2).reshape((ll, 1, self.num_neg)),
                },
                {
                'y': y,
                'neg_y': neg_y,
                'cor_y': cor_y,
                }
               )


class WD2vec(object):
    """Calculate feature vector.
    
    The correlation coefficient (distance similarity) of 
    the feature vector on the column side matches the 
    correlation coefficient of original data.
    
    Parameters
    ----------
    .
        X_df (pandas.DataFrame of shape (n_samples, n_features))
            training instances
        
        cor_mat (ndarray of shape (n_features, n_features))
            correlation matrix to match
    
    Attributes
    ----------
    
    wgt_row : ndarray of shape (n_samples, n_features)
        feature vector of row side
    
    wgt_col : ndarray of shape (n_features, n_features)
        feature vector of col side
    
    gamma : float
        RBF parameter gamma
    """
    
    def __init__(self, X_df, cor_mat):
        assert X_df.shape[1] == cor_mat.shape[0]
        self.X_df = X_df
        self.cor_mat = cor_mat
        
    def make_model(self, num_features=12, num_neg=1,
                         gamma=0.0, embeddings_val=0.1, loss_wgt_neg=0.001,
                         seed=None, rscore=None, cscore=None):
        self.num_features = num_features
        num_user = self.X_df.shape[0]
        num_product = self.X_df.shape[1]
        self.num_neg = num_neg
        self.models = make_model(num_user=num_user, num_product=num_product, num_features=num_features, num_neg=num_neg,
                                 gamma=gamma, embeddings_val=embeddings_val, seed=seed, loss_wgt_neg=loss_wgt_neg,
                                 rscore=rscore, cscore=cscore)
        return self.models
    
    def get_seq(self, batch_size=32):
        seq = Seq(user_id_list=np.arange(self.X_df.shape[0]),
                  X_df_values=self.X_df.values,
                  cor_mat=self.cor_mat,
                  batch_size=batch_size,
                  num_neg=self.num_neg)
        return seq
    
    def train(self, epochs=5, batch_size=128, verbose=2,
              use_multiprocessing=False, workers=1, shuffle=True,
              callbacks=None, callbacks_add=None, lr0=0.02, flag_early_stopping=True,
              patience=5,
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
            eraly_stopping = EarlyStopping(monitor='loss', patience=patience)
            callbacks = [eraly_stopping, lr_scheduler]
            #callbacks = [lr_scheduler]
        if callbacks_add is not None:
            callbacks = callbacks + callbacks_add
        model = self.models['model2']
        seq = self.get_seq(batch_size=batch_size)
        res = model.fit(seq, steps_per_epoch=len(seq),
                        epochs=epochs,
                        verbose=verbose,
                        shuffle=shuffle,
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
              callbacks=None, lr0=0.001, base=8, flag_early_stopping=True):
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
            early_stopping = EarlyStopping(monitor='loss', patience=5)
            callbacks = [early_stopping, lr_scheduler] if flag_early_stopping else [lr_scheduler]
            #callbacks = [lr_scheduler]
        model = self.models['model2']
        seq = self.get_seq(batch_size=batch_size)
        res = model.fit(seq, steps_per_epoch=len(seq),
                        epochs=epochs,
                        shuffle=shuffle,
                        verbose=verbose,
                        callbacks=callbacks)
        return res
    
    def get_wgt_byrow(self):
        wgt = self.models['model'].get_layer('user_embedding').get_weights()[0]
        return wgt
    wgt_row = property(get_wgt_byrow)
    
    def get_wgt_bycol(self):
        wgt = self.models['model'].get_layer('prod_embedding').get_weights()[0]
        return wgt
    wgt_col = property(get_wgt_bycol)
    
    def get_gamma(self):
        gamma = 1 / (self.num_features * SIGMA2)
        return gamma
    gamma = property(get_gamma)



