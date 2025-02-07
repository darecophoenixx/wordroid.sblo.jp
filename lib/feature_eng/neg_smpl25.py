'''
Copyright (c) 2018 Norio Tamada
Released under the MIT license
https://github.com/darecophoenixx/wordroid.sblo.jp/blob/master/lib/feature_eng/LICENSE.md
'''

'''
neg_smpl4
ネガティブサンプリングを精緻にした
（必ず0のものから選択するようにした）
'''

'''
neg_smpl6
user --- prod          --- neg_prod x nn
     --- neg_user x nn
行側のネガティブサンプリングを追加
'''

'''
neg_smpl7
neg_user x nn --- [user = prod] --- neg_prod x nn
ネガティブサンプリングは、すべてが対象（1のものもネガティブサンプリングされる）
'''

'''
neg_smpl8
neg_user x nn --- [user = prod] --- neg_prod x nn
上記(neg_smpl7)をスタックする
ネガティブサンプリングは、すべてが対象（1のものもネガティブサンプリングされる）
'''

'''
neg_smpl10
neg_user x nn --- [user = prod] --- neg_prod x nn
上記形式だが、user毎に、作成してバッチに送る
上記形式(neg_smpl8)を保持しつつ、スピードアップを図る
'''

'''
neg_smpl16 <- neg_smpl10
ネガティブサンプリングのロス関数を別にして重みを下げる
アイテム側はフワッと離れてくれれば良い
'''

'''
neg_smpl24 <- neg_smpl16
確率として入力する
（値+1） ÷ 2
値：1 → 1
値：0.1 → 0.55
値0.1以下はオミットする
loss=binary_crossentropy
metrics=mse
'''

'''
neg_smpl25 <- neg_smpl24
sparse_matのgetnnz()を利用してデータの数をそろえることができる
行側もランダムにできる
このタイプはすでに確率になっているマトリックスを利用
'''

import sys
import itertools
import random
from collections.abc import Mapping
import logging

import numpy as np
import scipy
import gensim
from tqdm import tqdm

from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Lambda, \
    Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Conv2DTranspose, \
    GlobalAveragePooling1D, MaxPooling1D, MaxPooling2D, \
    concatenate, Flatten, Average, Activation, \
    RepeatVector, Permute, Reshape, Dot, \
    multiply, dot, add
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import losses
from tensorflow.keras.callbacks import ProgbarLogger, Callback, History, LearningRateScheduler
#from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.constraints import NonNeg, MaxNorm, UnitNorm
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K

from keras.utils import PyDataset

__all__ = ['WordAndDoc2vec', ]

from feature_eng.similarity import MySparseMatrixSimilarity

SIGMA2 = 0.2**2




class WordAndDocSimilarity(object):

    def __init__(self, wgt_row, row_dic, wgt_col, col_dic):
        self.wgt_row = wgt_row
        self.wgt_col = wgt_col
        self.row_dic = row_dic
        self.col_dic = col_dic
        self.num_features = wgt_row.shape[1]
        self.gamma = 1 / (self.num_features * SIGMA2)

    def _get_sim(self, mx, d, query):
        res = calc_gsim(query, mx, self.gamma)
        res2 = [(d[ii], ee) for ii, ee in enumerate(res)]
        return res2

    def get_sim_bycol(self, query):
        return self._get_sim(self.wgt_col, self.col_dic, query)

    def get_sim_byrow(self, query):
        return self._get_sim(self.wgt_row, self.row_dic, query)

    def get_fet_bycol(self, token):
        idx = self.col_dic.token2id[token]
        return self.wgt_col[idx]

    def get_fet_byrow(self, token):
        idx = self.row_dic.token2id[token]
        return self.wgt_row[idx]


class Dic4seq(Mapping):
    '''
    REQUIRE:
      __getitem__() : list of product (by row/user)
    '''

    # def __init__(self, wtsmart_csr_prob, idfs=None, nn=5, pow=1.5):
    #     self.csr = wtsmart_csr_prob
    #     self.len = self.csr.nnz
    #     if idfs is not None:
    #         assert self.csr.shape[1] == idfs.shape[1]
    #         idfs2 = np.power(idfs, pow)
    #         low = 0.5
    #         self.idfs_sc = np.ceil((idfs2 - idfs2.min()) / (idfs2.max() - idfs2.min()) * (nn-low) + low).astype(int)
    #     else:
    #         self.idfs_sc = None
    #     idx_mm = np.zeros(dtype="uint32", shape=(self.csr.nnz, 2))
    #     idx_mm[:,0], idx_mm[:,1] = wtsmart_csr_prob.nonzero()
    #     self.idx_mm = idx_mm
    #     if self.idfs_sc is not None:
    #         l = []
    #         for ii, icol in itertools.islice(enumerate(self.idx_mm[:,1]), None):
    #             idf = self.idfs_sc[0,icol]
    #             l.extend([self.idx_mm[[ii]]] * idf)
    #             #l.extend([self.idx_mm[[ii]]] * 2) # for test
    #         self.idx_mm = np.concatenate(l)
    #         self.len = self.idx_mm.shape[0]
    def __init__(self, wtsmart_csr_prob, idfs=None, nn=10):
        self.csr = wtsmart_csr_prob
        self.len = wtsmart_csr_prob.count_nonzero() # self.csr.nnz
        # idx_mmは拡張しない
        idx_mm = np.zeros(dtype="uint32", shape=(self.len, 2))
        idx_mm[:,0], idx_mm[:,1] = wtsmart_csr_prob.nonzero()
        self.idx_mm = idx_mm
        # self.indexを拡張する
        self.index = np.arange(self.len)
        if idfs is not None:
            index_ex = []
            with tqdm(total=self.len) as pbar:
                for idx in self.index:
                    icol = self.idx_mm[idx,1]
                    idf = idfs[0, icol]
                    m = nn / idf
                    m = int(1 if m < 1 else m)
                    index_ex.extend([idx] * m)
                    pbar.update(1)
            self.index = index_ex = np.array(index_ex)
            self.len = len(self.index)

    def __getitem__(self, idx_seq2):
        '''
        idx_seq2 は Seq2.index
        
        ex.
        array([356, 63, 3, ...])
        '''
        # self.indexは拡張されている（可能性がある）
        idx = self.index[idx_seq2]
        row_idx, col_idx = self.idx_mm[idx,0], self.idx_mm[idx,1]
        target_val = np.array(self.csr[row_idx, col_idx])[0]
        vals3 = (target_val + 1) / 2
        ret = [(row_idx[ii], col_idx[ii], vals3[ii]) for ii in range(len(row_idx))]
        return ret

    def __len__(self):
        return self.len

    def __iter__(self):
        '''no use'''
        #return self.idx_mm
        return iter(self.idx_mm)


class Seq(object):

    def __init__(self, dic4seq,
                 num_neg=2, stack_size=5,
                 batch_size=32, shuffle=False, state=None):
        self.dic4seq = dic4seq
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.stack_size = stack_size
        self.state = state
        self.num_neg = num_neg

        self.row_indeces = list(range(self.dic4seq.csr.shape[0]))
        self.col_indeces = list(range(self.dic4seq.csr.shape[1]))

        y = [0]*self.num_neg + [1] + [0]*self.num_neg
        self.y = np.array([[[1]] * self.stack_size])
        neg_y = [0]*self.num_neg + [0]*self.num_neg
        self.neg_y = np.array([[neg_y] * self.stack_size])

        '''estimate self length'''
        self.initialize_it()
        self.len = 1
        for _ in self.it:
            self.len += 1

        self.initialize_it()

    def initialize_it(self):
        self.it = iter(range(0, len(self.dic4seq), self.batch_size*self.stack_size))
        self.idx_next = self.it.__next__()

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __next__(self):
        idx = self.idx_next
        self.part = self.stacked_comb[idx:((idx+self.batch_size) if idx+self.batch_size<len(self.stacked_comb) else len(self.stacked_comb))]
        try:
            self.idx_next = self.it.__next__()
        except StopIteration:
            self.initialize_it()
        return None

    def get_combs(self, idx_seq2):
        '''
        user_part は row_index, col_index のndarray
        
        ex.
        array([[ 0,  3],
               [ 0, 11],
               [ 0, 22],
               [ 0, 24],
               [ 0, 25]], dtype=uint8)
        '''
        combs = self.dic4seq[idx_seq2]
        return combs

    def get_stacked_combs(self, idx_seq2):
        '''
        Seq2から呼ばれる
        '''
        combs = self.get_combs(idx_seq2)
        stacked_combs = [[ee2 if ee2 is not None else random.choice(combs) for ee2 in ee] for ee in itertools.zip_longest(*[iter(combs)]*self.stack_size)]
        return stacked_combs

    def __getitem__(self, combs):
        users, neg_user, prods, neg_prod, y, neg_y = self.get_data(combs)
        return ({'input_user': users, 'input_neg_user': neg_user, 'input_prod': prods, 'input_neg_prod': neg_prod}, {'y': y, 'neg_y': neg_y})

    def get_data(self, combs):
        users_id, prods_id, prob = list(zip(*combs))
        y = np.array(prob).reshape((1,self.stack_size,1))

        neg_user = self.get_neg_user(prods_id)
        neg_prod = self.get_neg_prod(users_id)
        _, neg_y = self.get_ret_y()
        return (np.array([users_id]), neg_user, np.array([prods_id]), neg_prod, y, neg_y)

    def get_ret_y(self):
        return self.y, self.neg_y

    def get_neg_prod(self, user_id):
        pop_prod = self.col_indeces
        try:
            neg = random.sample(pop_prod, self.stack_size*self.num_neg)
        except ValueError:
            neg = random.choices(pop_prod, k=self.stack_size*self.num_neg)
        return np.array(neg).reshape(1, self.stack_size, self.num_neg)

    def get_neg_user(self, prod_id):
        pop_user = self.row_indeces
        try:
            neg = random.sample(pop_user, self.stack_size*self.num_neg)
        except ValueError:
            neg = random.choices(pop_user, k=self.stack_size*self.num_neg)
        return np.array(neg).reshape(1, self.stack_size, self.num_neg)

    def get_part(self, stacked_combs):
        x_input_user = []
        x_input_prod = []
        x_input_neg_user = []
        x_input_neg_prod = []
        y = []
        neg_y = []
        for combs in stacked_combs:
            x_train, y_train = self[combs]
            x_input_user.append(x_train['input_user'])
            x_input_prod.append(x_train['input_prod'])
            x_input_neg_user.append(x_train['input_neg_user'])
            x_input_neg_prod.append(x_train['input_neg_prod'])
            y.append(y_train['y'])
            neg_y.append(y_train['neg_y'])
        return (
              (
                  np.concatenate(x_input_user, axis=0), # input_user
                  np.concatenate(x_input_neg_user, axis=0), # input_neg_user
                  np.concatenate(x_input_prod, axis=0), # input_prod
                  np.concatenate(x_input_neg_prod, axis=0), # input_neg_prod
              ),
              (
                  np.concatenate(y, axis=0), # y
                  np.concatenate(neg_y, axis=0), # neg_y
              )
          )


class Seq2(PyDataset):

    def __init__(self, seq, **kwargs):
        super().__init__(**kwargs)
        self.seq = seq
        self.cnt = 0
        self.index = np.arange(len(self.seq.dic4seq))

        bs = self.seq.batch_size * self.seq.stack_size
        self.index_l = []
        for ii in range(len(self.seq)):
            self.index_l.append((self.index[ii*bs], self.index[((ii*bs+bs) if (ii*bs+bs)<len(self.seq.dic4seq) else len(self.seq.dic4seq))-1]))

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        bs = self.seq.batch_size * self.seq.stack_size
        l, u = self.index_l[idx]
        idx_seq2 = self.index[l:(u+1)]
        #user_part = self.seq.user_list[idx_user_part]
        #user_part = self.seq.dic4seq.idx_mm[idx_user_part]
        stacked_combs = self.seq.get_stacked_combs(idx_seq2)
        res = self.seq.get_part(stacked_combs)
        return res


def make_model(num_user=20, num_product=10,
               num_neg=2, stack_size=5, num_features=8, gamma=0.0, uninorm=None,
               embeddings_val=0.1, sigma2=SIGMA2, loss_wgt_neg=0.1,
               wgt_user=None, wgt_prod=None):

    user_embedding = Embedding(output_dim=num_features, input_dim=num_user,
                               embeddings_initializer=initializers.RandomUniform(minval=-embeddings_val, maxval=embeddings_val),
                               embeddings_regularizer=regularizers.l2(gamma),
                               embeddings_constraint=None if uninorm is None else UnitNorm(axis=1),
                               name='user_embedding', trainable=True)
    prod_embedding = Embedding(output_dim=num_features, input_dim=num_product,
                               embeddings_initializer=initializers.RandomUniform(minval=-embeddings_val, maxval=embeddings_val),
                               embeddings_regularizer=regularizers.l2(gamma),
                               embeddings_constraint=None if uninorm is None else UnitNorm(axis=1),
                               name='prod_embedding', trainable=True)
    # if wgt_user is None:
    #     prod_embedding = Embedding(output_dim=num_features, input_dim=num_product,
    #                                embeddings_initializer=initializers.RandomUniform(minval=-embeddings_val, maxval=embeddings_val),
    #                                embeddings_regularizer=regularizers.l2(gamma),
    #                                embeddings_constraint=None if uninorm is None else UnitNorm(axis=1),
    #                                name='prod_embedding', trainable=True)
    # else:
    #     prod_embedding = Embedding(output_dim=num_features, input_dim=num_product,
    #                                weights=[wgt_prod],
    #                                embeddings_initializer=initializers.RandomUniform(minval=-embeddings_val, maxval=embeddings_val),
    #                                embeddings_regularizer=regularizers.l2(gamma),
    #                                embeddings_constraint=None if uninorm is None else UnitNorm(axis=1),
    #                                name='prod_embedding', trainable=True)

    input_user = Input(shape=(stack_size,), name='input_user')
    input_neg_user = Input(shape=(stack_size, num_neg), name='input_neg_user')
    input_prod = Input(shape=(stack_size,), name='input_prod')
    input_neg_prod = Input(shape=(stack_size, num_neg), name='input_neg_prod')

    embed_user = user_embedding(input_user)
    #print('embed_user >', K.int_shape(embed_user))
    if wgt_user is not None:
        user_embedding.set_weights([wgt_user])
    embed_neg_user = user_embedding(input_neg_user)
    #print('embed_neg_user >', K.int_shape(embed_neg_user))
    embed_prod = prod_embedding(input_prod)
    #print('embed_prod >', K.int_shape(embed_prod))
    if wgt_prod is not None:
        prod_embedding.set_weights([wgt_prod])
    embed_neg_prod = prod_embedding(input_neg_prod)
    #print('embed_neg_prod >', K.int_shape(embed_neg_prod))

    model_user = Model(input_user, embed_user)
    model_neg_user = Model(input_neg_user, embed_neg_user)
    model_prod = Model(input_prod, embed_prod)
    model_neg_prod = Model(input_neg_prod, embed_neg_prod)

    input_user_vec = Input(shape=(stack_size, num_features), name='input_user_vec')
    input_neg_user_vec = Input(shape=(stack_size, num_neg, num_features), name='input_neg_user_vec')
    input_prod_vec = Input(shape=(stack_size, num_features), name='input_prod_vec')
    input_neg_prod_vec = Input(shape=(stack_size, num_neg, num_features), name='input_neg_prod_vec')

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
    '''user x prod'''
    embed_prod_a = Lambda(lambda x: K.expand_dims(x, axis=2))(embed_prod)
    #print('embed_prod_a >', K.int_shape(embed_prod_a))
    prob1 = Lambda(calc_prob2, name='y', arguments={'nn1': stack_size, 'nn2': 1})([embed_user, embed_prod_a])
    #print('prob1 >', K.int_shape(prob1))
    model_prob1 = Model([input_prod, input_user], prob1, name='model_prob1')

    prob1_cnfm = Lambda(calc_prob2, arguments={'nn1': stack_size, 'nn2': 1})([input_user_vec, Lambda(lambda x: K.expand_dims(x, axis=1))(input_prod_vec)])
    model_prob1_cnfm = Model([input_prod_vec, input_user_vec], prob1_cnfm, name='model_prob1_cnfm')

    '''prod x neg_prod'''
    prob2 = Lambda(calc_prob2, name='calc_prob2', arguments={'nn1': stack_size, 'nn2': num_neg})([embed_prod, embed_neg_prod])
    #print('prob2 >', K.int_shape(prob2))
    model_prob2 = Model([input_neg_prod, input_prod], prob2, name='model_prob2')

    prob2_cnfm = Lambda(calc_prob2, name='calc_prob2_cnfm', arguments={'nn1': stack_size, 'nn2': num_neg})([input_prod_vec, input_neg_prod_vec])
    model_prob2_cnfm = Model([input_prod_vec, input_neg_prod_vec], prob2_cnfm, name='model_prob2_cnfm')

    '''user x neg_user'''
    #embed_neg_user_a = Lambda(lambda x: K.expand_dims(x, axis=1))(embed_neg_user)
    prob3 = Lambda(calc_prob2, name='calc_prob3', arguments={'nn1': stack_size, 'nn2': num_neg})([embed_user, embed_neg_user])
    #print('prob3 >', K.int_shape(prob3))
    model_prob3 = Model([input_user, input_neg_user], prob3, name='model_prob3')

    prob3_cnfm = Lambda(calc_prob2, name='calc_prob3_cnfm', arguments={'nn1': stack_size, 'nn2': num_neg})([input_user_vec, input_neg_user_vec])
    model_prob3_cnfm = Model([input_user_vec, input_neg_user_vec], prob3_cnfm, name='model_prob3_cnfm')


    #print('prob >', K.int_shape(prob))
    #print('Flatten()(prob2) >', K.int_shape(Flatten()(prob2)))
    prob_neg = concatenate([prob3, prob2], axis=2, name='neg_y')
    #print('prob >', K.int_shape(prob))
    model = Model([input_user, input_neg_user, input_prod, input_neg_prod], [prob1, prob_neg])
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mse'],
    #               loss_weights={'y': 1.0, 'neg_y': loss_wgt_neg})
    # model.compile(loss={'y': 'binary_crossentropy', 'neg_y': 'binary_crossentropy'}, optimizer='adam', metrics={'y': 'binary_crossentropy', 'neg_y': 'binary_crossentropy'}, #metrics={'y': 'mse', 'neg_y': 'mse'},
    #               loss_weights={'y': 1.0, 'neg_y': loss_wgt_neg})
    model.compile(loss=['binary_crossentropy', 'binary_crossentropy'], optimizer='adam', metrics=['binary_crossentropy', 'binary_crossentropy'],
                  loss_weights=[1.0, loss_wgt_neg])
    models = {
        'model': model,
        'model_user': model_user,
        'model_neg_user': model_neg_user,
        'model_prod': model_prod,
        'model_neg_prod': model_neg_prod,
        'model_prob1': model_prob1,
        'model_prob2': model_prob2,
        'model_prob3': model_prob3,
        'model_prob1_cnfm': model_prob1_cnfm,
        'model_prob2_cnfm': model_prob2_cnfm,
        'model_prob3_cnfm': model_prob3_cnfm,
    }
    return models


class IlligalDocIndexException(Exception):
    pass


class SeqUserListShuffle(Callback):

    def on_epoch_begin(self, epoch, logs=None):
        random.shuffle(self.model.user_seq2.index)
        print('random.shuffle(self.model.seq2.index)')
        print(self.model.user_seq2.index[:5])



class WordAndDoc2vec(object):

    def __init__(self,
                 wtsmart_csr_prob, word_dic, doc_dic,
                 idfs=None,
                 logging=False,
                 load=False
                 ):
        if load:
            return
        self.logging = logging
        self.init()

        self.word_dic = word_dic
        self.doc_dic = doc_dic
        self.idfs = idfs

        print('max(doc_dic.keys()) + 1 >>>', max(doc_dic.keys()) + 1)

        num_features = max(self.word_dic.keys()) + 1
        print('num_features >>>', num_features)

        self.corpus_csc = wtsmart_csr_prob
        print('corpus_csc.shape >>>', wtsmart_csr_prob.shape)
        self.num_user = self.corpus_csc.shape[0]
        self.num_product = self.corpus_csc.shape[1]

        print('### creating Dic4seq...')
        self.dic4seq = Dic4seq(self.corpus_csc, idfs=self.idfs)
        print(self.dic4seq)

    def init(self):
        if self.logging:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    def make_model(self, num_neg=2, num_features=8, sigma2=SIGMA2,
                   gamma=0.0, embeddings_val=0.1, uninorm=None, stack_size=5, loss_wgt_neg=0.1,
                   wgt_user=None, wgt_prod=None):
        self.num_neg = num_neg
        self.stack_size = stack_size
        self.num_features = num_features

        models = make_model(num_user=self.num_user, num_product=self.num_product,
                            num_neg=num_neg, num_features=num_features,
                            gamma=gamma, uninorm=uninorm, sigma2=sigma2,
                            embeddings_val=embeddings_val, stack_size=stack_size, loss_wgt_neg=loss_wgt_neg,
                            wgt_user=wgt_user, wgt_prod=wgt_prod)
        self.models = models
        self.model = models['model']
        return models

    def get_seq(self, batch_size=32, shuffle=False, state=None):
        '''
        "shuffle" not implemented yet
        '''
        seq = Seq(self.dic4seq,
                  num_neg=self.num_neg,
                  batch_size=batch_size,
                  stack_size=self.stack_size,
                  shuffle=shuffle, state=state)
        return seq

    def train(self, epochs=50, batch_size=32, shuffle=False, state=None,
              verbose=1,
              use_multiprocessing=False, workers=1,
              callbacks=None):
        self.seq = self.get_seq(batch_size, shuffle=shuffle, state=state)
        print('len(seq) >>>', len(self.seq))
        self.seq2 = Seq2(self.seq, workers=workers, use_multiprocessing=use_multiprocessing)

        def lr_schedule(epoch, lr, epochs=epochs, lr0=0.02, base=8):
            b = 1 / np.log((epochs-1+np.exp(1)))
            a = 1 / np.log((epoch+np.exp(1))) / (1-b) - b/(1-b)
            lr = a*(1-1/base)*lr0 + lr0/base
            print('Learning rate: ', lr)
            return lr
        self.model.user_seq2 = self.seq2
        user_list_sh = SeqUserListShuffle()
        if callbacks is None:
            lr_scheduler = LearningRateScheduler(lr_schedule)
            callbacks = [lr_scheduler, user_list_sh]
        else:
            callbacks.append(user_list_sh)
        self.hst = self.model.fit(self.seq2, steps_per_epoch=None, #len(self.seq) // batch_size,
                                            epochs=epochs,
                                            verbose=verbose,
                                            callbacks=callbacks)
        return self.hst

    def get_wgt_byrow(self):
        wgt_user = self.model.get_layer('user_embedding').get_weights()[0]
        return wgt_user
    wgt_row = property(get_wgt_byrow)

    def get_wgt_bycol(self):
        wgt = self.model.get_layer('prod_embedding').get_weights()[0]
        return wgt
    wgt_col = property(get_wgt_bycol)

    def get_sim(self):
        sim = WordAndDocSimilarity(self.wgt_row, self.doc_dic, self.wgt_col, self.word_dic)
        return sim
    sim = property(get_sim)


def calc_gsim(vec, m, gamma=1.0):
    vec2 = (vec**2).sum()
    m2 = (m**2).sum(axis=1)
    vec_m = m.dot(vec)
    d2 = vec2 + m2 - 2*vec_m
    ret = np.exp(-gamma * d2)
    return ret
