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

import sys, os
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

from feature_eng.similarity import MySparseMatrixSimilarity
from feature_eng.neg_smpl25 import WordAndDoc2vec as WordAndDoc2vec_org
from feature_eng.neg_smpl25 import (
    WordAndDocSimilarity,
    make_model,
    calc_gsim
)
from feature_eng.neg_smpl25 import Dic4seq as Dic4seq_org

class Dic4seq(Dic4seq_org):
    '''
    REQUIRE:
      __getitem__() : list of product (by row/user)
    '''

    def __init__(self, wtsmart_csr_prob, idfs=None, nn=10,
                 dir='.'):
        self.path = os.path.join(dir, 'dic4seq')

        self.csr = wtsmart_csr_prob
        self.len = wtsmart_csr_prob.count_nonzero() # self.csr.nnz
        #idx_mm = np.zeros(dtype="uint32", shape=(wtsmart_csr_prob.count_nonzero(), 2))
        idx_mm = np.memmap(self.path, dtype="uint32", mode="w+",
                           shape=(wtsmart_csr_prob.count_nonzero(), 2))
        idx_mm[:,0], idx_mm[:,1] = wtsmart_csr_prob.nonzero()
        self.idx_mm = idx_mm
        if idfs is not None:
            l = []
            cnt = 0
            for ii, icol in itertools.islice(enumerate(self.idx_mm[:,1]), None):
                idf = idfs[0, icol]
                m = nn / idf
                m = int(1 if m < 1 else m)
                cnt += m
            self.path_ext = os.path.join(dir, 'dic4seq_ext')
            idx_mm_ext = np.memmap(self.path_ext, dtype="uint32", mode="w+",
                                   shape=(cnt, 2))
            idx = 0
            for ii, icol in itertools.islice(enumerate(self.idx_mm[:,1]), None):
                idf = idfs[0, icol]
                m = nn / idf
                m = int(1 if m < 1 else m)
                idx_mm_ext[idx:(idx+m),:] = self.idx_mm[[ii],:]
                idx += m
            self.idx_mm = idx_mm_ext
            self.len = self.idx_mm.shape[0]



class WordAndDoc2vec(WordAndDoc2vec_org):

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
