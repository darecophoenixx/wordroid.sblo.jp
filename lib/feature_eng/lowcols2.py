'''
Copyright (c) 2018 Norio Tamada
Released under the MIT license
https://github.com/darecophoenixx/wordroid.sblo.jp/blob/master/lib/feature_eng/LICENSE.md
'''

import numpy as np
from math import ceil
import random


#class Seq(object):
#    
#    def __init__(self, doc_seq, batch_size=32, shaffle=False, state=None):
#        self.doc_seq = doc_seq
#        self.shaffle = shaffle
#        self.state = state
#        self.batch_size = batch_size
#        
#        self.product_list = list(self.doc_seq.word_dic.token2id.keys())
#        self.user_list = np.array(list(self.doc_seq.doc_dic.token2id.keys()), dtype=str)
#
#        '''estimate self length'''
#        self.initialize_it()
#        self.len = 1
#        for _ in self.it:
#            self.len += 1
#        
#        self.initialize_it()
#        
#        # others
#        self.num_class = max(self.doc_seq.word_dic.keys()) + 1
#    
#    def initialize_it(self):
#        if self.shaffle:
#            '''not implemented yet'''
#            #random.seed(self.state)
#            #random.shuffle(self.user_list)
#        
#        self.it = iter(range(0, len(self.user_list), self.batch_size))
#        self.idx_next = self.it.__next__()
#    
#    def __len__(self):
#        return self.len
#    
#    def __iter__(self):
#        return self
#    
#    def __next__(self):
#        idx = self.idx_next
#        self.users_part = self.user_list[idx:((idx+self.batch_size) if idx+self.batch_size<len(self.user_list) else len(self.user_list))]
#        res = self.getpart(self.users_part)
#        try:
#            self.idx_next = self.it.__next__()
#        except StopIteration:
#            self.initialize_it()
#        return res
#    
#    def __getitem__(self, iuser):
#        ret_user, ret_y = self.get_data(iuser)
#        return ({'input_user': ret_user}, ret_y)
#    
#    def get_data(self, iuser):
#        user_id = self.doc_seq.doc_dic.token2id[iuser]
#        prods = self.doc_seq[user_id]
#        prods_id = [self.doc_seq.word_dic.token2id[e1] for e1 in prods]
#        cats = to_categorical(prods_id, num_classes=self.num_class)
#        cat = cats.sum(axis=0)
#        
#        return (user_id, cat)
#    
#    def getpart(self, users_part):
#        x_input_user = []
#        y = []
#        for iuser in users_part:
#            x_train, y_train = self[iuser]
#            x_input_user.append(x_train['input_user'])
#            y.append(y_train.tolist())
#        return ({
#            'input_user': np.array(x_input_user),
#            },
#            np.array(y))


from keras_ex.gkernel import GaussianKernel, GaussianKernel2, GaussianKernel3

from keras.layers import Input, Embedding, LSTM, GRU, Dense, Dropout, Lambda, \
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
from keras.models import Model, Sequential
from keras import losses
from keras.callbacks import BaseLogger, ProgbarLogger, Callback, History,\
    LearningRateScheduler, EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from keras import regularizers
from keras import initializers
from keras.metrics import categorical_accuracy
from keras.constraints import maxnorm, non_neg
from keras.optimizers import RMSprop
from keras.utils import to_categorical, Sequence
from keras import backend as K


SIGMA2 = 0.5**2
def make_model(num_user=20, num_product=39, num_features=12, num_neg=1,
                gamma=0.0, embeddings_val=0.5, seed=None,
                rscore=None, cscore=None, sigma2=SIGMA2, loss_wgt_neg=0.1):
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

    input_user = Input(shape=(1,), name='input_user')
    input_neg_user = Input(shape=(1, num_neg), name='input_neg_user')

    embed_user = user_embedding(input_user)
    #print('embed_user >', K.int_shape(embed_user))
    embed_neg_user = user_embedding(input_neg_user)
    #print('embed_neg_user >', K.int_shape(embed_neg_user))
    model_user = Model(input_user, embed_user)
    
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
        prob = K.exp(-1./(num_features*sigma2) * d2)
        #print('prob (in calc_prob2) >', K.int_shape(prob))
        return prob
    '''user x neg_user'''
    prob3 = Lambda(calc_prob2, name='y_neg_user', arguments={'nn1': 1, 'nn2': num_neg})([embed_user, embed_neg_user])
    #print('prob3 >', K.int_shape(prob3))
    model_prob3 = Model([input_user, input_neg_user], prob3, name='model_prob3')

    
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
    gamma = 1./(init_wgt.var() * init_wgt.shape[1])
    weights1 = [init_wgt, np.log(np.array([gamma]))]
    layer_gk1 = GaussianKernel3(num_product, num_features, name='gkernel1', weights=weights1)
    oup = layer_gk1(Flatten()(embed_user))
    oup =Activation('linear', name='y')(oup)
    model_gk1 = Model(input_user, oup)
    
    model = Model(input_user, oup)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    model2 = Model([input_user, input_neg_user], [oup, prob3])
    model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'],
                  loss_weights={'y': 1.0, 'y_neg_user': loss_wgt_neg})
    model3 = Model([input_user, input_neg_user], [oup, prob3])
    model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'],
                  loss_weights={'y': 1.0, 'y_neg_user': loss_wgt_neg/10})
    models = {
        'model': model,
        'model2': model2,
        'model3': model3,
        'model_user': model_user,
        'model_gk1': model_gk1,
        'model_prob3': model_prob3,
    }
    return models


class Seq(Sequence):
    
    def __init__(self, user_id_list, X_df_values, batch_size, num_neg=1):
        self.user_id_list = list(user_id_list)
        self.X_df_values = X_df_values
        self.num_neg = num_neg
        self.batch_size = batch_size
        
        self.len = ceil(len(self.user_id_list) / self.batch_size)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        bs = self.batch_size
        idx_rng_from, idx_rng_to = (idx*bs), ((idx*bs+bs) if (idx*bs+bs)<len(self.user_id_list) else len(self.user_id_list))
        input_user = self.user_id_list[idx_rng_from:idx_rng_to]
        ll = len(input_user)
        input_neg_user = random.sample(self.user_id_list, self.num_neg*ll)
        y = self.X_df_values[idx_rng_from:idx_rng_to,:]
        y_neg_user = np.zeros((ll, self.num_neg))
        return (
                {
                'input_user': np.array(input_user),
                'input_neg_user': np.array(input_neg_user).reshape((ll, 1, self.num_neg)),
                },
                {
                'y': y,
                'y_neg_user': y_neg_user,
                }
               )


class WD2vec(object):
    
    def __init__(self, X_df):
        self.X_df = X_df
        
    def make_model(self, num_features=12, num_neg=2,
                         gamma=0.0, embeddings_val=0.1, loss_wgt_neg=0.05,
                         seed=None, rscore=None, cscore=None):
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
                  batch_size=batch_size,
                  num_neg=self.num_neg)
        return seq
    
    def train(self, epochs=5, batch_size=32, verbose=2,
              use_multiprocessing=False, workers=1, shuffle=True,
              callbacks=None, lr0=0.005, flag_early_stopping=True,
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
        model = self.models['model']
        model2 = self.models['model2']
        model3 = self.models['model3']
        seq = self.get_seq(batch_size=batch_size)
        res = model3.fit(seq, steps_per_epoch=len(seq),
                        epochs=epochs,
                        verbose=verbose,
                        shuffle=shuffle,
                        callbacks=callbacks)
#        res = model.fit(np.arange(self.X_df.shape[0]), self.X_df.values,
#                                  batch_size=batch_size,
#                                  epochs=epochs,
#                                  verbose=verbose,
#                                  shuffle=shuffle,
#                                  callbacks=callbacks)
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
            early_stopping = EarlyStopping(monitor='loss', patience=3)
            callbacks = [early_stopping, lr_scheduler] if flag_early_stopping else [lr_scheduler]
            #callbacks = [lr_scheduler]
        model2 = self.models['model2']
        seq = self.get_seq(batch_size=batch_size)
        res = model2.fit(seq, steps_per_epoch=len(seq),
                        epochs=epochs,
                        shuffle=shuffle,
                        verbose=verbose,
                        callbacks=callbacks)
        return res
    
    def get_wgt_byrow(self):
        wgt = self.models['model'].get_layer('user_embedding').get_weights()[0]
        return wgt
    
    def get_wgt_bycol(self):
        wgt = self.models['model'].get_layer('gkernel1').get_weights()[0]
        return wgt
    
    def get_gamma(self):
        logged_gamma = self.models['model'].get_layer('gkernel1').get_weights()[1][0]
        gamma = np.exp(logged_gamma)
        return gamma


