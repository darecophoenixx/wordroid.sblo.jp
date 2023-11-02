'''
Copyright (c) 2018 Norio Tamada
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
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.constraints import NonNeg, MaxNorm
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K

__all__ = ['WordAndDoc2vec', ]

from feature_eng.neg_smpl22 import WordAndDoc2vec as WordAndDoc2vec_org
from feature_eng.neg_smpl22 import (
    WordAndDocSimilarity,
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
    
    def __init__(self, wtsmart_csr, row_dic, col_dic, max_count=10):
        self.max_count = max_count
        self.csr = wtsmart_csr
        self.row_dic = row_dic
        self.col_dic = col_dic
        
        self.row_dic[0] # make id2token
        self.col_dic[0] # make id2token
    
    def __getitem__(self, key):
        doc_id = self.row_dic.token2id[key]
        #c = gensim.matutils.scipy2sparse(self.mysim.getCorpusByDoc(doc_id, method='WT_SMART2'))
        c = gensim.matutils.scipy2sparse(self.csr[doc_id,:])
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
        return self.csr.shape[0]
    
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
                 wtsmart_csr, corpus_csr_col,
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
        
        self.corpus_csr = wtsmart_csr
        print('corpus_csr.shape >>>', wtsmart_csr.shape)
        self.num_user = self.corpus_csr.shape[0]
        self.num_product = self.corpus_csr.shape[1]
        
        print('### creating Dic4seq...')
        self.dic4seq = Dic4seq(self.corpus_csr, self.doc_dic, self.word_dic)
        print(self.dic4seq)
        
        self.user_list = list(self.dic4seq.keys())

        print('### creating Dic4seq col...')
        self.corpus_csr_col = corpus_csr_col
        self.dic4seq_col = Dic4seq(self.corpus_csr_col, row_dic=self.word_dic, col_dic=self.doc_dic)
        print(self.dic4seq_col)
        self.col_prob = col_prob

