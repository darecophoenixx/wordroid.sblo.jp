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
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.constraints import MaxNorm, NonNeg
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K

__all__ = ['WordAndDoc2vec', ]
           
from feature_eng.similarity import MySparseMatrixSimilarity

SIGMA2 = 0.5**2




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
        #c = self.corpus[idx]
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
    
    def __init__(self, comb, row_dic, col_dic, row_csr, col_csr, 
                 num_neg=2, stack_size=5,
                 batch_size=32, shuffle=False, state=None):
        self.comb = comb
        self.row_dic = row_dic
        self.col_dic = col_dic
        self.row_csr = row_csr
        self.col_csr = col_csr
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.stack_size = stack_size
        self.state = state
        self.num_neg = num_neg
        
        self.row_indeces = range(row_csr.shape[0])
        self.col_indeces = range(col_csr.shape[0])
        
        ret_y = [0]*self.num_neg + [1]*1 + [0]*self.num_neg
        self.ret_y = np.array([[ret_y] * self.stack_size])
        
        if self.shuffle:
            random.seed(self.state)
            random.shuffle(self.comb)
        self.stacked_comb = [ee for ee in itertools.zip_longest(*[iter(self.comb)]*self.stack_size)]
        
        '''estimate self length'''
        self.initialize_it()
        self.len = 1
        for _ in self.it:
            self.len += 1
        
        self.initialize_it()
    
    def initialize_it(self):
        self.it = iter(range(0, len(self.stacked_comb), self.batch_size))
        self.idx_next = self.it.__next__()

    def __len__(self):
        return self.len
    
    def __iter__(self):
        return self
    
    def __next__(self):
        idx = self.idx_next
        self.part = self.stacked_comb[idx:((idx+self.batch_size) if idx+self.batch_size<len(self.stacked_comb) else len(self.stacked_comb))]
        #res = self.getpart(self.users_part)
        try:
            self.idx_next = self.it.__next__()
        except StopIteration:
            self.initialize_it()
        return None
    
    def __getitem__(self, idx):
        combs = list(self.stacked_comb[idx])
        ret_user = []
        ret_neg_user =[]
        ret_prod = []
        ret_neg_prod = []
        ret_y = []
        for ii, icomb in enumerate(combs):
            if icomb is None:
                icomb = random.choice(self.comb)
                combs[ii] = icomb
        users, neg_user, prods, neg_prod, y = self.get_data(combs)
        return ({'input_user': users, 'input_neg_user': neg_user, 'input_prod': prods, 'input_neg_prod': neg_prod}, y)
    
    def get_data(self, icomb):
        users, prods = list(zip(*icomb))
        users_id = [self.row_dic.token2id[k] for k in users]
        prods_id = [self.col_dic.token2id[k] for k in prods]
        
        neg_user = self.get_neg_user(prods_id)
        neg_prod = self.get_neg_prod(users_id)
        ret_y = self.get_ret_y()
        return (np.array([users_id]), neg_user, np.array([prods_id]), neg_prod, ret_y)
    
    def get_ret_y(self):
        return self.ret_y
    
    def get_neg_prod(self, user_id):
        #pop_prod = self.col_indeces.difference(self.row_csr.getrow(user_id).nonzero()[1])
        pop_prod = self.col_indeces
        try:
            neg = random.sample(pop_prod, self.stack_size*self.num_neg)
        except ValueError:
            neg = random.choices(pop_prod, k=self.stack_size*self.num_neg)
        return np.array(neg).reshape(1, self.stack_size, self.num_neg)
    
    def get_neg_user(self, prod_id):
        #pop_user = self.row_indeces.difference(self.col_csr.getrow(prod_id).nonzero()[1])
        pop_user = self.row_indeces
        try:
            neg = random.sample(pop_user, self.stack_size*self.num_neg)
        except ValueError:
            neg = random.choices(pop_user, k=self.stack_size*self.num_neg)
        return np.array(neg).reshape(1, self.stack_size, self.num_neg)
        
    def get_part(self, part):
        x_input_user = []
        x_input_prod = []
        x_input_neg_user = []
        x_input_neg_prod = []
        y = []
        for idx in part:
            x_train, y_train = self[idx]
            x_input_user.append(x_train['input_user'])
            x_input_prod.append(x_train['input_prod'])
            x_input_neg_user.append(x_train['input_neg_user'])
            x_input_neg_prod.append(x_train['input_neg_prod'])
            y.append(y_train)
        return ({
            'input_prod': np.concatenate(x_input_prod, axis=0),
            'input_neg_prod': np.concatenate(x_input_neg_prod, axis=0),
            'input_user': np.concatenate(x_input_user, axis=0),
            'input_neg_user': np.concatenate(x_input_neg_user, axis=0),
            },
            np.concatenate(y, axis=0))



class Seq2(Sequence):
    
    def __init__(self, seq):
        self.seq = seq
        self.len_stacked_comb = len(self.seq.stacked_comb)
        self.index_list = np.arange(self.len_stacked_comb)
    
    def __len__(self):
        return len(self.seq)
    
    def __getitem__(self, idx):
        bs = self.seq.batch_size
        part = self.index_list[(idx*bs):((idx*bs+bs) if (idx*bs+bs)<self.len_stacked_comb else self.len_stacked_comb)]
        res = self.seq.get_part(part)
        return res


def make_model(num_user=20, num_product=10,
               num_neg=2, stack_size=5, num_features=8, gamma=0.0, maxnorm=None,
               embeddings_val=0.1, sigma2=SIGMA2):

    user_embedding = Embedding(output_dim=num_features, input_dim=num_user,
                               embeddings_initializer=initializers.RandomUniform(minval=-embeddings_val, maxval=embeddings_val),
                               embeddings_regularizer=regularizers.l2(gamma),
                               embeddings_constraint=None if maxnorm is None else MaxNorm(max_value=maxnorm, axis=1),
                               name='user_embedding', trainable=True)
    prod_embedding = Embedding(output_dim=num_features, input_dim=num_product,
                               embeddings_initializer=initializers.RandomUniform(minval=-embeddings_val, maxval=embeddings_val),
                               embeddings_regularizer=regularizers.l2(gamma),
                               embeddings_constraint=None if maxnorm is None else MaxNorm(max_value=maxnorm, axis=1),
                               name='prod_embedding', trainable=True)

    input_user = Input(shape=(stack_size,), name='input_user')
    input_neg_user = Input(shape=(stack_size, num_neg), name='input_neg_user')
    input_prod = Input(shape=(stack_size,), name='input_prod')
    input_neg_prod = Input(shape=(stack_size, num_neg), name='input_neg_prod')

    embed_user = user_embedding(input_user)
    #print('embed_user >', K.int_shape(embed_user))
    embed_neg_user = user_embedding(input_neg_user)
    #print('embed_neg_user >', K.int_shape(embed_neg_user))
    embed_prod = prod_embedding(input_prod)
    #print('embed_prod >', K.int_shape(embed_prod))
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
    prob1 = Lambda(calc_prob2, name='calc_prob1', arguments={'nn1': stack_size, 'nn2': 1})([embed_user, embed_prod_a])
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
    prob = concatenate([prob3, prob1, prob2], axis=2)
    #print('prob >', K.int_shape(prob))
    model = Model([input_user, input_neg_user, input_prod, input_neg_prod], prob)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
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


class WordAndDoc2vec(object):
    
    def __init__(self,
                 corpus_csr, word_dic, doc_dic,
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
        
        print('### creating MySparseMatrixSimilarity...')
        self.mysim = MySparseMatrixSimilarity(self.corpus_csr, num_features=num_features)
        print(self.mysim)
        
        print('### creating Dic4seq...')
        self.dic4seq = Dic4seq(self.mysim, self.doc_dic, self.word_dic)
        print(self.dic4seq)
        
        print('### creating Comb...')
        self.user_list = list(self.dic4seq.keys())
        comb = []
        for iuser in self.user_list:
            p = self.dic4seq[iuser]
            res = [(iuser, ee) for ee in p]
            comb.extend(res)
        self.comb = comb
        print('len(comb) >', len(self.comb))
    
    def init(self):
        if self.logging:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    
    def make_model(self, num_neg=2, stack_size=5, num_features=8,
                   gamma=0.0, embeddings_val=0.1, maxnorm=None):
        #self.num_user = len(self.doc_seq)
        self.num_user = self.corpus_csr.shape[0]
        self.num_product = self.mysim.index_csc.shape[1]
        self.num_neg = num_neg
        self.stack_size = stack_size
        self.num_features = num_features
        
        models = make_model(num_user=self.num_user, num_product=self.num_product,
                   num_neg=num_neg, stack_size=stack_size, num_features=num_features, gamma=gamma,
                   embeddings_val=embeddings_val, maxnorm=maxnorm)
        self.models = models
        self.model = models['model']
        return models
    
    def get_seq(self, batch_size=32, shuffle=False, state=None):
        '''
        "shuffle" not implemented yet
        '''
        seq = Seq(self.comb,
                  row_dic=self.doc_dic,
                  col_dic=self.word_dic,
                  row_csr=self.corpus_csr,
                  col_csr=self.corpus_csr.T.tocsr(),
                  num_neg=self.num_neg,
                  stack_size=self.stack_size,
                  batch_size=batch_size,
                  shuffle=shuffle, state=state)
        return seq
    
    def train(self, epochs=50, batch_size=32, shuffle=True, state=None,
              verbose=1,
              use_multiprocessing=False, workers=1,
              callbacks=None):
        self.seq = self.get_seq(batch_size, shuffle=shuffle, state=state)
        print('len(seq) >>>', len(self.seq))
        self.seq2 = Seq2(self.seq)
        
        def lr_schedule(epoch, lr, epochs=epochs, lr0=0.02, base=8):
            b = 1 / np.log((epochs-1+np.exp(1)))
            a = 1 / np.log((epoch+np.exp(1))) / (1-b) - b/(1-b)
            lr = a*(1-1/base)*lr0 + lr0/base
            print('Learning rate: ', lr)
            return lr
        if callbacks is None:
            lr_scheduler = LearningRateScheduler(lr_schedule)
            callbacks = [lr_scheduler]
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
