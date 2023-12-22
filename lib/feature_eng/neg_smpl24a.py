'''
Copyright (c) 2024 Norio Tamada
Released under the MIT license
https://github.com/darecophoenixx/wordroid.sblo.jp/blob/master/lib/feature_eng/LICENSE.md
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
#from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.constraints import NonNeg, MaxNorm
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K

__all__ = ['WordAndDoc2vec', ]

from feature_eng.similarity import MySparseMatrixSimilarity
from feature_eng.neg_smpl24 import WordAndDoc2vec as WordAndDoc2vec_org
from feature_eng.neg_smpl24 import (
    WordAndDocSimilarity,
    make_model,
    calc_gsim
)



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
        # key : doc name (user name)
        doc_id = self.row_dic.token2id[key]
        #c = gensim.matutils.scipy2sparse(self.mysim.getCorpusByDoc(doc_id, method='WT_SMART2'))
        c = gensim.matutils.scipy2sparse(self.csr[doc_id,:])
        col_ids, vals = list(zip(*c))
        max_val = max(vals)
        vals2 = np.array(list(vals)) / max_val # max=1で基準化
        vals3 = (vals2 + 1) / 2
        col_keys = [self.col_dic[idx] for idx in col_ids]
        c2 = list(zip(col_keys, vals3.tolist()))
        return c2
    
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



class WordAndDoc2vec(WordAndDoc2vec_org):
    def __init__(self,
                 wtsmart_csr, word_dic, doc_dic,
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
        
        self.corpus_csc = wtsmart_csr
        print('corpus_csc.shape >>>', wtsmart_csr.shape)
        self.num_user = self.corpus_csc.shape[0]
        self.num_product = self.corpus_csc.shape[1]
        
        print('### creating Dic4seq...')
        self.dic4seq = Dic4seq(self.corpus_csc, self.doc_dic, self.word_dic)
        print(self.dic4seq)
        
        self.user_list = list(self.dic4seq.keys())
    
    # def make_model(self, num_neg=2, num_features=8,
    #                gamma=0.0, embeddings_val=0.1, maxnorm=None, stack_size=5, loss_wgt_neg=0.1,
    #                wgt_user=None, wgt_prod=None):
    #     self.num_neg = num_neg
    #     self.stack_size = stack_size
    #     self.num_features = num_features
        
    #     models = make_model(num_user=self.num_user, num_product=self.num_product,
    #                         num_neg=num_neg, num_features=num_features, gamma=gamma,
    #                         embeddings_val=embeddings_val, maxnorm=maxnorm, stack_size=stack_size, loss_wgt_neg=loss_wgt_neg,
    #                         wgt_user=wgt_user, wgt_prod=wgt_prod)
    #     self.models = models
    #     self.model = models['model']
    #     return models
