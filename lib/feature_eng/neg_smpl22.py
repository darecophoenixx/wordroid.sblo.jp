'''
Copyright (c) 2023 Norio Tamada
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
neg_smpl20
user --- prod
user --- neg_prod x nn
'''

'''
neg_smpl21
user --- prod --- neg_prod x nn
prod --- user --- neg_user x nn
'''

'''
neg_smpl22
user --- prod --- neg_prod x nn
prod --- user --- neg_user x nn
'''

import sys
import itertools
import random
from collections.abc import Mapping
import logging

import numpy as np
import scipy
import gensim

from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Lambda, \
    Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Conv2DTranspose, \
    GlobalAveragePooling1D, MaxPooling1D, MaxPooling2D, \
    concatenate, Flatten, Average, Activation, \
    RepeatVector, Permute, Reshape, Dot, \
    multiply, dot, add
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import losses
from tensorflow.keras.callbacks import BaseLogger, ProgbarLogger, Callback, History, LearningRateScheduler
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.constraints import NonNeg, MaxNorm
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K


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


class Sentences(object):

    def __init__(self, docs, word_dic):
        self.docs = docs
        self.word_dic = word_dic

    def __len__(self):
        return len(self.docs)

    def __iter__(self):
        for irow in self.docs:
            yield [self.word_dic.token2id[ee] for ee in irow]


class Dic4seq(Mapping):
    '''
    REQUIRE:
      keys()        : all row/user names (implement __iter__)
      __getitem__() : list of product (by row/user)
      to_ids()
      to_ids_row()
      col_keys()    : all column/item names
    '''

    def __init__(self, mysim, row_dic, col_dic, max_count=10):
        self.max_count = max_count
        self.mysim = mysim
        self.row_dic = row_dic
        self.col_dic = col_dic

        self.row_dic[0] # make id2token
        self.col_dic[0] # make id2token

    def __getitem__(self, key):
        doc_id = self.row_dic.token2id[key]
        c = gensim.matutils.scipy2sparse(self.mysim.getCorpusByDoc(doc_id, method='WT_SMART2'))
        keys, vals = list(zip(*c))
        max_val = max(vals)
        ret = []
        for idx, v in c:
            cnt0 = int(v / max_val * self.max_count)
            #cnt = cnt0 if cnt0 !=0 else 1
            cnt = cnt0
            #print(cnt0, cnt, self.col_dic[idx])
            ret.extend([self.col_dic[idx]]*cnt)
        return ret

    def __len__(self):
        return self.mysim.index_csc.shape[0]

    def __iter__(self):
        return iter(list(self.row_dic.token2id.keys()))

    def to_ids_row(self, l):
        '''
        input : list of rowkey
        output: list of idx
        '''
        return [self.row_dic.token2id[k] for k in l]

    def to_ids(self, l):
        '''
        input : list of colkey
        output: list of idx
        '''
        return [self.col_dic.token2id[k] for k in l]

    def col_keys(self):
        return list(self.col_dic.token2id.keys())


class Seq(object):

    def __init__(self, dic4seq, row_dic, col_dic, csc, row_csr=None, col_csr=None,
                 num_neg=2, stack_size=5,
                 batch_size=32, shuffle=False, state=None):
        self.dic4seq = dic4seq
        self.row_dic = row_dic
        self.col_dic = col_dic
        self.row_csr = row_csr
        self.col_csr = col_csr
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.stack_size = stack_size
        self.state = state
        self.num_neg = num_neg

        self.row_indeces = list(range(csc.shape[0]))
        self.col_indeces = list(range(csc.shape[1]))
        self.user_list = list(self.dic4seq.keys())

        y = [0]*self.num_neg + [1] + [0]*self.num_neg
        self.y = np.array([[[1]] * self.stack_size])
        neg_y = [0]*self.num_neg
        self.neg_y = np.array([[neg_y] * self.stack_size])

        '''estimate self length'''
        self.initialize_it()
        self.len = 1
        for _ in self.it:
            self.len += 1

        self.initialize_it()

    def initialize_it(self):
        self.it = iter(range(0, len(self.user_list), self.batch_size))
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

    def get_combs(self, user_part):
        users_id = self.dic4seq.to_ids_row(user_part)
        combs = []
        for iuser in user_part:
            p = self.dic4seq[iuser]
            combs.extend([(iuser, ee) for ee in p])
        return combs

    def get_stacked_combs(self, user_part):
        combs = self.get_combs(user_part)
        stacked_combs = [[ee2 if ee2 is not None else random.choice(combs) for ee2 in ee] for ee in itertools.zip_longest(*[iter(combs)]*self.stack_size)]
        return stacked_combs

    def __getitem__(self, combs):
        ret_user = []
        ret_neg_user =[]
        ret_prod = []
        ret_neg_prod = []
        ret_y = []
        users, prods, neg_prod, y, neg_y = self.get_data(combs)
        return ({'input_user': users, 'input_prod': prods, 'input_user_neg_prod': neg_prod}, {'y': y, 'neg_y': neg_y})

    def get_data(self, combs):
        users, prods = list(zip(*combs))
        users_id = [self.row_dic.token2id[k] for k in users]
        prods_id = [self.col_dic.token2id[k] for k in prods]

        #neg_user = self.get_neg_user(prods_id)
        neg_prod = self.get_neg_prod(users_id)
        y, neg_y = self.get_ret_y()
        return (np.array([users_id]), np.array([prods_id]), neg_prod, y, neg_y)

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
        x_input_neg_prod = []
        y = []
        neg_y = []
        for idx in stacked_combs:
            x_train, y_train = self[idx]
            x_input_user.append(x_train['input_user'])
            x_input_prod.append(x_train['input_prod'])
            x_input_neg_prod.append(x_train['input_user_neg_prod'])
            y.append(y_train['y'])
            neg_y.append(y_train['neg_y'])
        return ({
            'input_prod': np.concatenate(x_input_prod, axis=0),
            'input_user_neg_prod': np.concatenate(x_input_neg_prod, axis=0),
            'input_user': np.concatenate(x_input_user, axis=0),
            },
            {
                'y': np.concatenate(y, axis=0),
                'user_neg_prod': np.concatenate(neg_y, axis=0),
            })


class Seq2(Sequence):

    def __init__(self, seq, seq_col, col_prob):
        self.seq = seq
        self.seq_col = seq_col
        self.col_prob = col_prob
        self.cnt = 0
        self.cumsum = np.cumsum(col_prob)

    def __len__(self):
        return len(self.seq)

    def get_part_col(self):
        col_part = np.random.choice(self.seq_col.user_list, size=self.seq_col.batch_size, replace=False, p=self.col_prob)
        return col_part

    def getitem(self, idx):
        bs = self.seq.batch_size
        user_part = self.seq.user_list[(idx*bs):((idx*bs+bs) if (idx*bs+bs)<len(self.seq.user_list) else len(self.seq.user_list))]
        stacked_combs = self.seq.get_stacked_combs(user_part)
        xtrain, ytrain = self.seq.get_part(stacked_combs)
        return xtrain, ytrain

    def getitem_col(self, length):
        # 行側のデータの長さに合わせる（=length）
        combs = []
        while True:
          #icol = np.random.choice(self.seq_col.user_list, size=1, p=self.col_prob)
          icol = random.choices(self.seq_col.user_list, cum_weights=self.cumsum)
          combs0 = self.seq_col.get_combs(icol)
          combs.extend(combs0)
          if length < len(combs):
            combs = combs[:length]
            break
        stacked_combs_col = [[ee2 if ee2 is not None else random.choice(combs) for ee2 in ee] for ee in itertools.zip_longest(*[iter(combs)]*self.seq_col.stack_size)]
        xtrain_col, ytrain_col = self.seq_col.get_part(stacked_combs_col)
        return xtrain_col, ytrain_col

    def __getitem__(self, idx):
        # === 行側のデータを取得する
        xtrain_row, ytrain_row = self.getitem(idx)
        length = xtrain_row['input_user'].shape[0] * xtrain_row['input_user'].shape[1]
        # === 列側のデータを取得する
        xtrain_col, ytrain_col = self.getitem_col(length)

        input_user = xtrain_row['input_user']
        input_user_col = xtrain_col['input_prod']
        
        input_prod = xtrain_row['input_prod']
        input_prod_col = xtrain_col['input_user']
        
        input_user_neg_prod = xtrain_row['input_user_neg_prod']
        input_user_col_neg_prod = xtrain_col['input_user_neg_prod']
        
        y = ytrain_row['y']
        y_col = ytrain_col['y']

        user_neg_prod = ytrain_row['user_neg_prod']
        user_col_neg_prod = ytrain_col['user_neg_prod']
        xtrain = {
          'input_user': input_user, 'input_prod': input_prod,
          'input_user_col': input_user_col, 'input_prod_col': input_prod_col,
          'input_user_neg_prod': input_user_neg_prod,
          'input_user_col_neg_prod': input_user_col_neg_prod,
        }
        ytrain = {
            'y': y, 'y_col': y_col,
            'user_neg_prod': user_neg_prod,
            'user_col_neg_prod': user_col_neg_prod,
        }
        return xtrain, ytrain


def make_model(num_user=20, num_product=10,
               num_neg=2, stack_size=5, num_features=8, gamma=0.0, maxnorm=None,
               embeddings_val=0.1, sigma2=SIGMA2,
               loss_wgt_y_col=0.1, loss_wgt_user_neg_prod=1.0, loss_wgt_user_col_neg_prod=0.1,
               wgt_user=None, wgt_prod=None):

    if wgt_user is None:
        user_embedding = Embedding(output_dim=num_features, input_dim=num_user,
                                   embeddings_initializer=initializers.RandomUniform(minval=-embeddings_val, maxval=embeddings_val),
                                   embeddings_regularizer=regularizers.l2(gamma),
                                   embeddings_constraint=None if maxnorm is None else MaxNorm(max_value=maxnorm, axis=1),
                                   name='user_embedding', trainable=True)
    else:
        user_embedding = Embedding(output_dim=num_features, input_dim=num_user,
                                   weights=[wgt_user],
                                   embeddings_initializer=initializers.RandomUniform(minval=-embeddings_val, maxval=embeddings_val),
                                   embeddings_regularizer=regularizers.l2(gamma),
                                   embeddings_constraint=None if maxnorm is None else MaxNorm(max_value=maxnorm, axis=1),
                                   name='user_embedding', trainable=True)
    if wgt_user is None:
        prod_embedding = Embedding(output_dim=num_features, input_dim=num_product,
                                   embeddings_initializer=initializers.RandomUniform(minval=-embeddings_val, maxval=embeddings_val),
                                   embeddings_regularizer=regularizers.l2(gamma),
                                   embeddings_constraint=None if maxnorm is None else MaxNorm(max_value=maxnorm, axis=1),
                                   name='prod_embedding', trainable=True)
    else:
        prod_embedding = Embedding(output_dim=num_features, input_dim=num_product,
                                   weights=[wgt_prod],
                                   embeddings_initializer=initializers.RandomUniform(minval=-embeddings_val, maxval=embeddings_val),
                                   embeddings_regularizer=regularizers.l2(gamma),
                                   embeddings_constraint=None if maxnorm is None else MaxNorm(max_value=maxnorm, axis=1),
                                   name='prod_embedding', trainable=True)

    input_user = Input(shape=(stack_size,), name='input_user')
    input_prod = Input(shape=(stack_size,), name='input_prod')
    input_user_neg_prod = Input(shape=(stack_size, num_neg), name='input_user_neg_prod')

    input_user_col = Input(shape=(stack_size,), name='input_user_col')
    input_prod_col = Input(shape=(stack_size,), name='input_prod_col')
    input_user_col_neg_prod = Input(shape=(stack_size, num_neg), name='input_user_col_neg_prod')

    embed_user = user_embedding(input_user)
    #print('embed_user >', K.int_shape(embed_user))
    embed_prod = prod_embedding(input_prod)
    #print('embed_prod >', K.int_shape(embed_prod))
    embed_user_neg_prod = prod_embedding(input_user_neg_prod)
    #print('embed_neg_prod >', K.int_shape(embed_neg_prod))

    embed_user_col = user_embedding(input_user_col)
    embed_prod_col = prod_embedding(input_prod_col)
    embed_user_col_neg_prod = user_embedding(input_user_col_neg_prod)

    model_user = Model(input_user, embed_user)
    model_prod = Model(input_prod, embed_prod)
    model_user_neg_prod = Model(input_user_neg_prod, embed_user_neg_prod)
    model_user_col = Model(input_user_col, embed_user_col)
    model_prod_col = Model(input_prod_col, embed_prod_col)
    model_user_col_neg_prod = Model(input_user_col_neg_prod, embed_user_col_neg_prod)

    input_user_vec = Input(shape=(stack_size, num_features), name='input_user_vec')
    input_prod_vec = Input(shape=(stack_size, num_features), name='input_prod_vec')
    input_user_neg_prod_vec = Input(shape=(stack_size, num_neg, num_features), name='input_user_neg_prod_vec')
    input_user_col_vec = Input(shape=(stack_size, num_features), name='input_user_col_vec')
    input_prod_col_vec = Input(shape=(stack_size, num_features), name='input_prod_col_vec')
    input_user_col_neg_prod_vec = Input(shape=(stack_size, num_neg, num_features), name='input_user_col_neg_prod_vec')

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

    '''user_col x prod_col'''
    embed_prod_col_a = Lambda(lambda x: K.expand_dims(x, axis=2))(embed_prod_col)
    prob1_col = Lambda(calc_prob2, name='y_col', arguments={'nn1': stack_size, 'nn2': 1})([embed_user_col, embed_prod_col_a])
    model_prob1_col = Model([input_prod_col, input_user_col], prob1_col, name='model_prob1_col')

    prob1_col_cnfm = Lambda(calc_prob2, arguments={'nn1': stack_size, 'nn2': 1})([input_user_col_vec, Lambda(lambda x: K.expand_dims(x, axis=1))(input_prod_col_vec)])
    model_prob1_col_cnfm = Model([input_prod_col_vec, input_user_col_vec], prob1_col_cnfm, name='model_prob1_col_cnfm')

    # '''prod x neg_prod'''
    # prob2 = Lambda(calc_prob2, name='calc_prob2', arguments={'nn1': stack_size, 'nn2': num_neg})([embed_prod, embed_neg_prod])
    # #print('prob2 >', K.int_shape(prob2))
    # model_prob2 = Model([input_neg_prod, input_prod], prob2, name='model_prob2')

    # prob2_cnfm = Lambda(calc_prob2, name='calc_prob2_cnfm', arguments={'nn1': stack_size, 'nn2': num_neg})([input_prod_vec, input_neg_prod_vec])
    # model_prob2_cnfm = Model([input_prod_vec, input_neg_prod_vec], prob2_cnfm, name='model_prob2_cnfm')

    # '''user x neg_user'''
    # #embed_neg_user_a = Lambda(lambda x: K.expand_dims(x, axis=1))(embed_neg_user)
    # prob3 = Lambda(calc_prob2, name='calc_prob3', arguments={'nn1': stack_size, 'nn2': num_neg})([embed_user, embed_neg_user])
    # #print('prob3 >', K.int_shape(prob3))
    # model_prob3 = Model([input_user, input_neg_user], prob3, name='model_prob3')

    # prob3_cnfm = Lambda(calc_prob2, name='calc_prob3_cnfm', arguments={'nn1': stack_size, 'nn2': num_neg})([input_user_vec, input_neg_user_vec])
    # model_prob3_cnfm = Model([input_user_vec, input_neg_user_vec], prob3_cnfm, name='model_prob3_cnfm')

    '''user x user_neg_prod'''
    prob4 = Lambda(calc_prob2, name='calc_prob4', arguments={'nn1': stack_size, 'nn2': num_neg})([embed_user, embed_user_neg_prod])
    #print('prob4 >', K.int_shape(prob4))
    model_prob4 = Model([input_user, input_user_neg_prod], prob4, name='model_prob4')

    prob4_cnfm = Lambda(calc_prob2, name='calc_prob4_cnfm', arguments={'nn1': stack_size, 'nn2': num_neg})([input_user_vec, input_user_neg_prod_vec])
    model_prob4_cnfm = Model([input_user_vec, input_user_neg_prod_vec], prob4_cnfm, name='model_prob4_cnfm')

    '''user_col x user_col_neg_prod'''
    prob4_col = Lambda(calc_prob2, name='calc_prob4_col', arguments={'nn1': stack_size, 'nn2': num_neg})([embed_user_col, embed_user_col_neg_prod])
    model_prob4_col = Model([input_user_col, input_user_col_neg_prod], prob4_col, name='model_prob4_col')

    prob4_col_cnfm = Lambda(calc_prob2, name='calc_prob4_col_cnfm', arguments={'nn1': stack_size, 'nn2': num_neg})([input_user_col_vec, input_user_col_neg_prod_vec])
    model_prob4_col_cnfm = Model([input_user_col_vec, input_user_col_neg_prod_vec], prob4_col_cnfm, name='model_prob4_col_cnfm')


    #print('prob >', K.int_shape(prob))
    #print('Flatten()(prob2) >', K.int_shape(Flatten()(prob2)))
    #prob_neg = concatenate([prob4, prob5], axis=2, name='neg_y')
    prob_user_neg_prod = Lambda(lambda x: x, name='user_neg_prod')(prob4)
    prob_user_col_neg_prod = Lambda(lambda x: x, name='user_col_neg_prod')(prob4_col)
    #print('prob >', K.int_shape(prob))
    model = Model([input_user, input_prod, input_user_neg_prod, input_user_col, input_prod_col, input_user_col_neg_prod], [prob1, prob1_col, prob_user_neg_prod, prob_user_col_neg_prod])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'],
                  loss_weights={'y': 1.0, 'user_neg_prod': loss_wgt_user_neg_prod, 'y_col': loss_wgt_y_col, 'user_col_neg_prod': loss_wgt_user_col_neg_prod})
    models = {
        'model': model,
        'model_user': model_user,
        'model_prod': model_prod,
        'model_user_neg_prod': model_user_neg_prod,
        'model_user_col': model_user_col,
        'model_prod_col': model_prod_col,
        'model_user_col_neg_prod': model_user_col_neg_prod,
        'model_prob1': model_prob1,
        'model_prob1_cnfm': model_prob1_cnfm,
        'model_prob1_col': model_prob1_col,
        'model_prob1_col_cnfm': model_prob1_col_cnfm,
        'model_prob4': model_prob4,
        'model_prob4_col': model_prob4_col,
        'model_prob1_cnfm': model_prob1_cnfm,
        'model_prob4_cnfm': model_prob4_cnfm,
        'model_prob4_col_cnfm': model_prob4_col_cnfm,
    }
    return models


class IlligalDocIndexException(Exception):
    pass


class SeqUserListShuffle(Callback):

    def on_epoch_end(self, epoch, logs=None):
        random.shuffle(self.model.user_seq.user_list)
        print('random.shuffle(self.model.seq.user_list)')
        print(self.model.user_seq.user_list[:5])



class WordAndDoc2vec(object):

    def __init__(self,
                 corpus_csr, corpus_csr_col,
                 word_dic, doc_dic,
                 col_prob,
                 logging=False,
                 load=False
                 ):
        if load:
            return
        self.logging = logging
        self.init()

        self.word_dic = word_dic
        self.doc_dic = doc_dic

        print('max(doc_dic.keys()) + 1 >>>', max(doc_dic.keys()) + 1)

        num_features = max(self.word_dic.keys()) + 1
        print('num_features >>>', num_features)

        self.corpus_csr = corpus_csr
        print('corpus_csr.shape >>>', corpus_csr.shape)
        self.num_user = self.corpus_csr.shape[0]
        self.num_product = self.corpus_csr.shape[1]

        print('### creating MySparseMatrixSimilarity...')
        self.mysim = MySparseMatrixSimilarity(self.corpus_csr, num_features=num_features, use_getCorpusByDoc=True)
        print(self.mysim)

        print('### creating Dic4seq...')
        self.dic4seq = Dic4seq(self.mysim, self.doc_dic, self.word_dic)
        print(self.dic4seq)

        self.user_list = list(self.dic4seq.keys())

        print('### creating MySparseMatrixSimilarity col...')
        self.corpus_csr_col = corpus_csr_col
        self.mysim_col = MySparseMatrixSimilarity(self.corpus_csr_col, num_features=self.corpus_csr_col.shape[1], use_getCorpusByDoc=True)
        print(self.mysim_col)

        print('### creating Dic4seq col...')
        self.dic4seq_col = Dic4seq(self.mysim_col, row_dic=self.word_dic, col_dic=self.doc_dic)
        print(self.dic4seq_col)
        self.col_prob = col_prob

    def init(self):
        if self.logging:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    def make_model(self, num_neg=2, num_features=8, sigma2=SIGMA2,
                   gamma=0.0, embeddings_val=0.1, maxnorm=None, stack_size=5,
                   loss_wgt_y_col=0.1, loss_wgt_user_neg_prod=1.0, loss_wgt_user_col_neg_prod=0.1,
                   wgt_user=None, wgt_prod=None):
        self.num_neg = num_neg
        self.stack_size = stack_size
        self.num_features = num_features

        models = make_model(num_user=self.num_user, num_product=self.num_product, sigma2=sigma2,
                            num_neg=num_neg, num_features=num_features, gamma=gamma,
                            embeddings_val=embeddings_val, maxnorm=maxnorm, stack_size=stack_size,
                            loss_wgt_y_col=loss_wgt_y_col, loss_wgt_user_neg_prod=loss_wgt_user_neg_prod, loss_wgt_user_col_neg_prod=loss_wgt_user_col_neg_prod,
                            wgt_user=wgt_user, wgt_prod=wgt_prod)
        self.models = models
        self.model = models['model']
        return models

    def get_seq(self, batch_size=32, batch_size_col=None, shuffle=False, state=None):
        '''
        "shuffle" not implemented yet
        '''
        if batch_size_col is None:
          batch_size_col = batch_size
        seq = Seq(self.dic4seq,
                  row_dic=self.dic4seq.row_dic,
                  col_dic=self.dic4seq.col_dic,
                  csc=self.corpus_csr,
                  num_neg=self.num_neg,
                  batch_size=batch_size,
                  stack_size=self.stack_size,
                  shuffle=shuffle, state=state)
        seq_col = Seq(self.dic4seq_col,
                      row_dic=self.dic4seq_col.row_dic,
                      col_dic=self.dic4seq_col.col_dic,
                      csc=self.corpus_csr_col,
                      num_neg=self.num_neg,
                      batch_size=batch_size_col,
                      stack_size=self.stack_size,
                      shuffle=shuffle, state=state)
        return seq, seq_col

    def train(self, epochs=50, batch_size=32, batch_size_col=None, shuffle=False, state=None,
              verbose=1,
              use_multiprocessing=False, workers=1,
              callbacks=None):
        self.seq, self.seq_col = self.get_seq(batch_size, batch_size_col=batch_size_col, shuffle=shuffle, state=state)
        print('len(seq) >>>', len(self.seq))
        self.seq2 = Seq2(self.seq, self.seq_col, col_prob=self.col_prob)

        def lr_schedule(epoch, lr, epochs=epochs, lr0=0.02, base=8):
            b = 1 / np.log((epochs-1+np.exp(1)))
            a = 1 / np.log((epoch+np.exp(1))) / (1-b) - b/(1-b)
            lr = a*(1-1/base)*lr0 + lr0/base
            print('Learning rate: ', lr)
            return lr
        self.model.user_seq = self.seq
        user_list_sh = SeqUserListShuffle()
        if callbacks is None:
            lr_scheduler = LearningRateScheduler(lr_schedule)
            callbacks = [lr_scheduler, user_list_sh]
        else:
            callbacks.append(user_list_sh)
        self.hst = self.model.fit(self.seq2, steps_per_epoch=len(self.seq),
                                            epochs=epochs,
                                            verbose=verbose,
                                            callbacks=callbacks,
                                            use_multiprocessing=use_multiprocessing,
                                            workers=workers)
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
