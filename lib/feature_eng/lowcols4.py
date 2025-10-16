'''
Copyright (c) 2018 Norio Tamada
Released under the MIT license
https://github.com/darecophoenixx/wordroid.sblo.jp/blob/master/lib/feature_eng/LICENSE.md
'''

import numpy as np




from keras_ex.gkernel import GaussianKernel, GaussianKernel2, GaussianKernel3

from keras.layers import (Input, Embedding, LSTM, GRU, Dense, Dropout, Lambda,
    Flatten)
from keras.models import Model, Sequential
from keras import losses
from keras.callbacks import ProgbarLogger, Callback, History,\
    LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
# from keras.wrappers.scikit_learn import KerasClassifier
from keras import regularizers
from keras import initializers
from keras.metrics import categorical_accuracy
from keras.optimizers import Adam
from tensorflow.keras.constraints import MaxNorm, NonNeg
from keras.optimizers import RMSprop
from keras.utils import to_categorical, Sequence
from keras import backend as K


def make_model(num_user=20, num_product=39, num_features=12,
                gamma=0.0, embeddings_val=0.5, seed=None,
                rscore=None, cscore=None, learning_rate=0.001):
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
    weights1 = init_wgt
    layer_gk1 = GaussianKernel(num_product, num_features, name='gkernel1', kernel_gamma=1.0/(num_features*0.2**2))
    oup = layer_gk1(embed_user)
    layer_gk1.set_weights([weights1])
    model_gk1 = Model(input_user, oup)
    
    
    model = Model(input_user, oup)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['binary_accuracy'])
    models = {
        'model': model,
        'model_user': model_user,
        'model_gk1': model_gk1,
    }
    return models


class WD2vec(object):
    
    def __init__(self, X_df):
        self.X_df = X_df
        
    def make_model(self, num_features=12,
                         gamma=0.0, embeddings_val=0.5, learning_rate=0.001,
                         seed=None, rscore=None, cscore=None):
        num_user = self.X_df.shape[0]
        num_product = self.X_df.shape[1]
        self.models = make_model(num_user=num_user, num_product=num_product, num_features=num_features,
                                 gamma=gamma, embeddings_val=embeddings_val, seed=seed, learning_rate=learning_rate,
                                 rscore=rscore, cscore=cscore)
        return self.models
    
    def train(self, epochs=5, epochs2=10, batch_size=32, verbose=1,
              use_multiprocessing=False, workers=1, shuffle=True,
              callbacks=None, callbacks1=[], callbacks2=[],
              lr0=0.001, flag_early_stopping=True,
              base=8):
        
        # def lr_schedule(epoch, lrx):
        #     def reduce(epoch, lr):
        #         if divmod(epoch,4)[1] == 3:
        #             lr *= (1/8)
        #         elif divmod(epoch,4)[1] == 2:
        #             lr *= (1/4)
        #         elif divmod(epoch,4)[1] == 1:
        #             lr *= (1/2)
        #         elif divmod(epoch,4)[1] == 0:
        #             pass
        #         return lr

        #     lra = lr0
        #     epoch1 = int(epochs / 8)
        #     epoch2 = epoch1
        #     epoch3 = epoch1
        #     epoch4 = epoch1

        #     if epoch1+epoch2+epoch3+epoch4 <= epoch:
        #         epoch = epoch - (epoch1+epoch2+epoch3+epoch4)
        #         lra = lra / 2

        #     if epoch<epoch1:
        #         lr = lra
        #         lr = reduce(epoch, lr)
        #     elif epoch<epoch1+epoch2:
        #         lr = lra/2
        #         lr = reduce(epoch, lr)
        #     elif epoch<epoch1+epoch2+epoch3:
        #         lr = lra/4
        #         lr = reduce(epoch, lr)
        #     elif epoch<epoch1+epoch2+epoch3+epoch4:
        #         lr = lra/8
        #         lr = reduce(epoch, lr)
        #     else:
        #         lr = lra/64

        #     if verbose == 0:
        #         pass
        #     else:
        #         print('Learning rate: ', lr)
        #     return lr
        
        if callbacks is None:
            lr_scheduler = LearningRateScheduler(lr_schedule)
            eraly_stopping = EarlyStopping(monitor='loss', patience=3)
            callbacks = [eraly_stopping, lr_scheduler]
        callbacks = callbacks + callbacks1
        model = self.models['model']
        res = model.fit(np.arange(self.X_df.shape[0]), self.X_df.values,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=verbose,
                        shuffle=shuffle,
                        callbacks=callbacks)
        
        lr2 = res.history['learning_rate'][-1] / 2.0
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

            lra = lr2
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
        
        lr_scheduler = LearningRateScheduler(lr_schedule)
        eraly_stopping = EarlyStopping(monitor='loss', patience=3, min_delta=0.00001)
        callbacks = [eraly_stopping, lr_scheduler]
        callbacks = callbacks + callbacks2
        res2 = model.fit(np.arange(self.X_df.shape[0]), self.X_df.values,
                         batch_size=batch_size,
                         epochs=epochs,
                         verbose=verbose,
                         shuffle=shuffle,
                         callbacks=callbacks)

        
        # res2 = self.train2(epochs=epochs2,
        #               batch_size=batch_size,
        #               verbose=verbose,
        #               use_multiprocessing=use_multiprocessing,
        #               shuffle=shuffle,
        #               workers=workers,
        #               callbacks=None, callbacks2=callbacks2,
        #               lr0=lr2, base=base, flag_early_stopping=flag_early_stopping)
        return res, res2
    
    def train2(self, epochs=5, batch_size=32, verbose=1,
              use_multiprocessing=False, workers=1, shuffle=True,
              callbacks=None, callbacks2=[], lr0=0.001, base=8, flag_early_stopping=True):
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
            early_stopping = EarlyStopping(monitor='loss', patience=2)
            callbacks = [early_stopping, lr_scheduler] if flag_early_stopping else [lr_scheduler]
        callbacks = callbacks + callbacks2
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
