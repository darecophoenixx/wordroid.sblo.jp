'''
Copyright (c) 2018 Norio Tamada
Released under the MIT license
https://github.com/darecophoenixx/wordroid.sblo.jp/blob/master/lib/feature_eng/LICENSE.md
'''

import numpy as np


class Seq(object):
    
    def __init__(self, doc_seq, batch_size=32, shaffle=False, state=None):
        self.doc_seq = doc_seq
        self.shaffle = shaffle
        self.state = state
        self.batch_size = batch_size
        
        self.product_list = list(self.doc_seq.word_dic.token2id.keys())
        self.user_list = np.array(list(self.doc_seq.doc_dic.token2id.keys()), dtype=str)

        '''estimate self length'''
        self.initialize_it()
        self.len = 1
        for _ in self.it:
            self.len += 1
        
        self.initialize_it()
        
        # others
        self.num_class = max(self.doc_seq.word_dic.keys()) + 1
    
    def initialize_it(self):
        if self.shaffle:
            '''not implemented yet'''
            #random.seed(self.state)
            #random.shuffle(self.user_list)
        
        self.it = iter(range(0, len(self.user_list), self.batch_size))
        self.idx_next = self.it.__next__()
    
    def __len__(self):
        return self.len
    
    def __iter__(self):
        return self
    
    def __next__(self):
        idx = self.idx_next
        self.users_part = self.user_list[idx:((idx+self.batch_size) if idx+self.batch_size<len(self.user_list) else len(self.user_list))]
        res = self.getpart(self.users_part)
        try:
            self.idx_next = self.it.__next__()
        except StopIteration:
            self.initialize_it()
        return res
    
    def __getitem__(self, iuser):
        ret_user, ret_y = self.get_data(iuser)
        return ({'input_user': ret_user}, ret_y)
    
    def get_data(self, iuser):
        user_id = self.doc_seq.doc_dic.token2id[iuser]
        prods = self.doc_seq[user_id]
        prods_id = [self.doc_seq.word_dic.token2id[e1] for e1 in prods]
        cats = to_categorical(prods_id, num_classes=self.num_class)
        cat = cats.sum(axis=0)
        
        return (user_id, cat)
    
    def getpart(self, users_part):
        x_input_user = []
        y = []
        for iuser in users_part:
            x_train, y_train = self[iuser]
            x_input_user.append(x_train['input_user'])
            y.append(y_train.tolist())
        return ({
            'input_user': np.array(x_input_user),
            },
            np.array(y))


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

#class Seq2(Sequence):
#    
#    def __init__(self, seq):
#        self.seq = seq
#    
#    def __len__(self):
#        return len(self.seq)
#    
#    def __getitem__(self, idx):
#        bs = self.seq.batch_size
#        user_part = self.seq.user_list[(idx*bs):((idx*bs+bs) if (idx*bs+bs)<len(self.seq.user_list) else len(self.seq.user_list))]
#        res = self.seq.getpart(user_part)
#        return res

def make_model(num_user=20, num_product=39, num_features=12,
                gamma=0.0, embeddings_val=0.5, seed=None,
                rscore=None, cscore=None):
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

    embed_user = Flatten()(user_embedding(input_user))

    model_user = Model(input_user, embed_user)
    
    rng = np.random.default_rng(seed)
    if cscore is None:
        init_wgt = (rng.random((num_product, num_features)) - 0.5) * 2 * embeddings_val
    else:
        init_wgt = cscore
    gamma = 1./(init_wgt.var() * init_wgt.shape[1])
    weights1 = [init_wgt, np.log(np.array([gamma]))]
    layer_gk1 = GaussianKernel3(num_product, num_features, name='gkernel1', weights=weights1)
    oup = layer_gk1(embed_user)
    model_gk1 = Model(input_user, oup)
    
    
    model = Model(input_user, oup)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    models = {
        'model': model,
        'model_user': model_user,
        'model_gk1': model_gk1,
    }
    return models

#class WD2vec_low(object):
#    
#    def __init__(self, doc_seq):
#        self.doc_seq = doc_seq
#        
#    def make_model(self, num_user=20, num_product=39, num_features=12,
#                         gamma=0.0, embeddings_val=0.5):
#        self.models = make_model(num_user=num_user, num_product=num_product, num_features=num_features,
#                                 gamma=gamma, embeddings_val=embeddings_val)
#        return self.models
#    
#    def train(self, epochs=5, batch_size=32, verbose=1,
#              use_multiprocessing=False, workers=1):
#        model = self.models['model']
#        seq = Seq(self.doc_seq, batch_size)
#        seq2 = Seq2(seq)
#        res = model.fit_generator(seq2,
#                                  steps_per_epoch=len(seq2),
#                                  epochs=epochs,
#                                  verbose=verbose,
#                                  use_multiprocessing=use_multiprocessing,
#                                  workers=workers)
#        return res
#    
#    def get_wgt_byrow(self, l=None):
#        wgt = self.models['model'].get_layer('user_embedding').get_weights()[0]
#        if l:
#            wgt = wgt[[self.doc_seq.doc_dic.token2id[ee] for ee in l]]
#        return wgt
#    
#    def get_wgt_bycol(self, l=None):
#        wgt = self.models['model'].get_layer('gkernel1').get_weights()[0]
#        if l:
#            wgt = wgt[[self.doc_seq.word_dic.token2id[ee] for ee in l]]
#        return wgt

class WD2vec(object):
    
    def __init__(self, X_df):
        self.X_df = X_df
        
    def make_model(self, num_features=12,
                         gamma=0.0, embeddings_val=0.5,
                         seed=None, rscore=None, cscore=None):
        num_user = self.X_df.shape[0]
        num_product = self.X_df.shape[1]
        self.models = make_model(num_user=num_user, num_product=num_product, num_features=num_features,
                                 gamma=gamma, embeddings_val=embeddings_val, seed=seed,
                                 rscore=rscore, cscore=cscore)
        return self.models
    
    def train(self, epochs=5, batch_size=32, verbose=1,
              use_multiprocessing=False, workers=1, shuffle=True,
              callbacks=None, lr0=0.001, flag_early_stopping=True,
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
        model = self.models['model']
        res = model.fit(np.arange(self.X_df.shape[0]), self.X_df.values,
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  verbose=verbose,
                                  shuffle=shuffle,
                                  callbacks=callbacks)
        lr2 = res.history['lr'][-1]
        res2 = self.train2(epochs=epochs,
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
        model = self.models['model']
        res = model.fit(np.arange(self.X_df.shape[0]), self.X_df.values,
                                  batch_size=batch_size,
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



