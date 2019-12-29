'''
Copyright (c) 2020 Norio Tamada
Released under the MIT license
https://github.com/darecophoenixx/wordroid.sblo.jp/blob/master/lib/feature_eng/LICENSE.md
'''

import itertools
import random
from collections import Mapping
#import logging

import numpy as np
import scipy
from sklearn.metrics.pairwise import euclidean_distances
import gensim
from gensim import utils
from keras.layers import Input, Embedding, LSTM, Dense, Dropout, Lambda, \
    Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Conv2DTranspose, \
    GlobalAveragePooling1D, MaxPooling1D, MaxPooling2D, \
    concatenate, Flatten, Average, Activation, \
    RepeatVector, Permute, Reshape, Dot, \
    multiply, dot, add
from keras.models import Model, Sequential
from keras import losses
from keras.callbacks import BaseLogger, ProgbarLogger, Callback, History, LearningRateScheduler
from keras.wrappers.scikit_learn import KerasClassifier
from keras import regularizers
from keras import initializers
from keras.metrics import categorical_accuracy
from keras.constraints import MaxNorm
from keras.utils import Sequence
from keras import backend as K


from .neg_smpl import (
    MySparseMatrixSimilarity,
    Sentences, Dic4seq,
    Seq2,
    IlligalDocIndexException
)


class Seq(object):
    
    def __init__(self, dic4seq,
                 batch_size=32, max_num_prod=5, shuffle=False, state=None):
        self.dic4seq = dic4seq
        self.shuffle = shuffle
        self.state = state
        self.max_num_prod = max_num_prod
        self.batch_size = batch_size
        #self.num_neg = num_neg
        
        self.product_list = list(self.dic4seq.col_keys())
        self.smpl = self.dic4seq.to_ids(self.product_list)
        
        self.user_list = np.array(list(self.dic4seq.keys()), dtype=str)
        

        '''estimate self length'''
        self.initialize_it()
        self.len = 1
        for _ in self.it:
            self.len += 1
        
        self.initialize_it()
    
    def initialize_it(self):
        if self.shuffle:
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
        ret_user, ret_prods, ret_neg_prods, ret_neg_prods_user, ret_y = self.get_data(iuser)
        return ({
            'input_user': ret_user,
            'input_prod': ret_prods,
            'input_neg_prod': ret_neg_prods,
            'input_neg_prod_user': ret_neg_prods_user,
        }, ret_y)
    
    def get_data(self, iuser):
        user_id = self.dic4seq.to_ids_row([iuser])[0]
        prods = self.dic4seq[iuser]
        prods_p = np.random.choice(prods, self.max_num_prod, replace=True)
        prods_id = self.dic4seq.to_ids(prods_p)
        
        _, neg = self.get_neg2(prods)
        neg_prod_users = []
        for iprod in neg:
            neg_prod_user = np.random.choice(self.dic4seq.mysim.index_csc[:,iprod].indices)
            neg_prod_users.append(neg_prod_user)
        ret_y = [1]*self.max_num_prod + [0]*self.max_num_prod + [1]*self.max_num_prod
        return [user_id], prods_id, neg, neg_prod_users, ret_y
    
#     def get_neg(self, prods):
#         neg = random.sample(self.smpl, self.num_neg*self.max_num_prod)
#         neg = np.array(neg).reshape(self.max_num_prod, self.num_neg).tolist()
#         ret_y = [0]*(self.num_neg*self.max_num_prod)
#         return (ret_y, neg)
        
    def get_neg2(self, prods):
        smpl = list(set(self.smpl).difference(prods+[0]))
        neg = random.choices(smpl, k=self.max_num_prod)
        #ret_y = [0]*self.max_num_prod
        return None, neg
        
    def getpart(self, users_part):
        x_input_user = []
        x_input_prod = []
        x_input_neg_prod = []
        x_input_neg_prod_user = []
        y = []
        for iuser in users_part:
            x_train, y_train = self[iuser]
            x_input_user.append(x_train['input_user'])
            x_input_prod.append(x_train['input_prod'])
            x_input_neg_prod.append(x_train['input_neg_prod'])
            x_input_neg_prod_user.append(x_train['input_neg_prod_user'])
            y.append(y_train)
        return ({
            'input_user': np.array(x_input_user),
            'input_prod': np.array(x_input_prod),
            'input_neg_prod': np.array(x_input_neg_prod),
            'input_neg_prod_user': np.array(x_input_neg_prod_user),
            },
            np.array(y))

def debug_print(*args, debug=False):
    if debug:
        print(*args)

def make_model(num_user=20, num_product=10, max_num_prod=5,
               num_neg=3, num_features=8, gamma=0.0,
               maxnorm=3.0, embeddings_val=0.5,
               debug=False
              ):
    
    user_embedding = Embedding(output_dim=num_features, input_dim=num_user,
                               embeddings_initializer=initializers.RandomUniform(minval=-embeddings_val, maxval=embeddings_val),
                               embeddings_constraint=MaxNorm(max_value=maxnorm, axis=1),
                               name='user_embedding', trainable=True)
    prod_embedding = Embedding(output_dim=num_features, input_dim=num_product,
                               embeddings_initializer=initializers.RandomUniform(minval=-embeddings_val, maxval=embeddings_val),
                               embeddings_constraint=MaxNorm(max_value=maxnorm, axis=1),
                               name='prod_embedding', trainable=True)
    
    input_user = Input(shape=(1,), name='input_user')
    input_prod = Input(shape=(max_num_prod,), name='input_prod')
    input_neg_prod = Input(shape=(max_num_prod, ), name='input_neg_prod')
    input_neg_prod_user = Input(shape=(max_num_prod, ), name='input_neg_prod_user')
    
    embed_user = user_embedding(input_user)
    embed_prod = prod_embedding(input_prod)
    embed_neg_prod = prod_embedding(input_neg_prod)
    embed_neg_prod_user = user_embedding(input_neg_prod_user)
    
    model_user = Model(input_user, embed_user)
    model_prod = Model(input_prod, embed_prod)
    model_neg_prod = Model(input_neg_prod, embed_neg_prod)
    model_neg_prod_user = Model(input_neg_prod_user, embed_neg_prod_user)
    
    input_user_vec = Input(shape=(1, num_features), name='input_user_vec')
    input_prod_vec = Input(shape=(max_num_prod, num_features), name='input_prod_vec')
    input_neg_prod_vec = Input(shape=(max_num_prod, num_features), name='input_neg_prod_vec')
    input_neg_prod_user_vec = Input(shape=(max_num_prod, num_features), name='input_neg_prod_user_vec')
    gamma = 1./(2.*num_features*0.1)
    def calc_dist2(x):
        debug_print('########## calc_dist2', debug=debug)
        embed_a = x[0]
        debug_print('embed_a >', K.int_shape(embed_a), debug=debug)
        embed_b = x[1]
        debug_print('embed_b >', K.int_shape(embed_b), debug=debug)
        embed_b = K.expand_dims(embed_b, axis=2)
        debug_print('embed_b >', K.int_shape(embed_b), debug=debug)
        embed_a2 = K.sum(K.square(embed_a), axis=2, keepdims=True)
        debug_print('embed_a2 >', K.int_shape(embed_a2), debug=debug)
        embed_b2 = K.sum(K.square(embed_b), axis=3, keepdims=False)
        debug_print('embed_b2 >', K.int_shape(embed_b2), debug=debug)
        
        embed_a_b = K.batch_dot(embed_a, embed_b, axes=(2,3))
        debug_print('embed_a_b >', K.int_shape(embed_a_b), debug=debug)
        dist2 = embed_a2 + embed_b2 - 2*embed_a_b
        debug_print('dist2 >', K.int_shape(dist2), debug=debug)
        debug_print('##########', debug=debug)
        return dist2
    def calc_dist2_a(x):
        debug_print('########## calc_dist2_a', debug=debug)
        embed_a = x[0]
        debug_print('embed_a >', K.int_shape(embed_a), debug=debug)
        embed_b = x[1]
        debug_print('embed_b >', K.int_shape(embed_b), debug=debug)
        embed_a2 = K.sum(K.square(embed_a), axis=2, keepdims=True)
        debug_print('embed_a2 >', K.int_shape(embed_a2), debug=debug)
        embed_b2 = K.sum(K.square(embed_b), axis=2, keepdims=True)
        debug_print('embed_b2 >', K.int_shape(embed_b2), debug=debug)
        
        embed_a_b = K.batch_dot(embed_b, embed_a, axes=(2,2))
        debug_print('embed_a_b >', K.int_shape(embed_a_b), debug=debug)
        dist2 = embed_a2 + embed_b2 - 2*embed_a_b
        debug_print('dist2 >', K.int_shape(dist2), debug=debug)
        debug_print('##########', debug=debug)
        return dist2
    debug_print('########## model_dist2_a', debug=debug)
    model_dist2_a = Model([input_user_vec, input_prod_vec], Lambda(calc_dist2_a)([input_user_vec, input_prod_vec]))
    debug_print('########## model_prob', debug=debug)
    model_prob = Model(
        [input_user_vec, input_prod_vec],
        Lambda(lambda x: K.exp(-gamma*x))(
            model_dist2_a([input_user_vec, input_prod_vec])
        )
    )
    
    debug_print('########## model_userXprod', debug=debug)
    prob_userXprod = model_prob([embed_user, embed_prod])
    model_userXprod = Model([input_user, input_prod], prob_userXprod, name='model_userXprod')
    
    debug_print('########## model_userXneg_prod', debug=debug)
    prob_userXneg_prod = model_prob([embed_user, embed_neg_prod])
    model_userXneg_prod = Model([input_user, input_neg_prod], prob_userXneg_prod, name='model_userXneg_prod')
    
    debug_print('########## model_userXneg_prod_user', debug=debug)
    prob_userXneg_prod_user = Lambda(calc_dist2)([embed_neg_prod, embed_neg_prod_user])
    prob_userXneg_prod_user = Lambda(lambda x: K.exp(-gamma*x))(prob_userXneg_prod_user)
    model_userXneg_prod_user = Model([input_neg_prod, input_neg_prod_user], prob_userXneg_prod_user, name='model_userXneg_prod_user')
    
    debug_print('########## model', debug=debug)
    prob_all = concatenate(
        [
            Flatten()(prob_userXprod),
            Flatten()(prob_userXneg_prod),
            Flatten()(prob_userXneg_prod_user),
        ]
    )
    model = Model([input_user, input_prod, input_neg_prod, input_neg_prod_user], prob_all)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    models = {
        'model_neg_prod': model_neg_prod,
        'model_neg_prod_user': model_neg_prod_user,
        'model_user': model_user,
        'model_prod': model_prod,
        'model_dist2_a': model_dist2_a,
        'model_prob': model_prob,
        'model_userXprod': model_userXprod,
        'model_userXneg_prod': model_userXneg_prod,
        'model_userXneg_prod_user': model_userXneg_prod_user,
        'model': model,
    }
    return models

class WordAndDoc2vec(object):
    
    def __init__(self,
                 doc_seq, word_dic, doc_dic,
                 logging=False,
                 load=False
                 ):
        if load:
            return
        self.logging = logging
        self.init()
        
        self.doc_seq = doc_seq
        self.word_dic = word_dic
        self.doc_dic = doc_dic
        
        print('len(doc_seq) >>>', len(doc_seq))
        print('max(doc_dic.keys()) + 1 >>>', max(doc_dic.keys()) + 1)
        if len(doc_seq) != (max(doc_dic.keys()) + 1):
            raise IlligalDocIndexException('num of doc_seq is [%s]. But doc_dic is [%s].' % (len(doc_seq), max(doc_dic.keys()) + 1))
        
        num_features = max(self.word_dic.keys()) + 1
        print('num_features >>>', num_features)
        
        print('### creating corpus_csr...')
        #corpus_csr = gensim.matutils.corpus2csc((self.word_dic.doc2bow(ee) for ee in self.doc_seq), num_terms=len(self.word_dic)).T
        corpus_csr = gensim.matutils.corpus2csc((self.word_dic.doc2bow(ee) for ee in self.doc_seq), num_terms=num_features).T
        print('corpus_csr.shape >>>', corpus_csr.shape)
        
        self.create_tfidf()
        print('### creating MySparseMatrixSimilarity...')
        #self.mysim = MySparseMatrixSimilarity(corpus_csr, num_features=len(self.word_dic), tfidf=self.tfidf)
        self.mysim = MySparseMatrixSimilarity(corpus_csr, num_features=num_features, tfidf=self.tfidf)
        print(self.mysim)
        
        print('### creating Dic4seq...')
        self.dic4seq = Dic4seq(self.mysim, self.doc_dic, self.word_dic)
        print(self.dic4seq)
    
    def _create_path(self, path):
        path_doc_seq = path + '_doc_seq'
        path_word_dic = path + '_word_dic'
        path_doc_dic = path + '_doc_dic'
        path_mysim = path + '_mysim'
        return path_doc_seq, path_word_dic, path_doc_dic, path_mysim
    
    def save(self, path):
        path_doc_seq, path_word_dic, path_doc_dic, path_mysim = self._create_path(path)
        with open(path_doc_seq, 'wb') as fw:
            import pickle
            pickle.dump(self.doc_seq, fw)
        self.word_dic.save(path_word_dic)
        self.doc_dic.save(path_doc_dic)
        self.mysim.save(path_mysim)
    
    @classmethod
    def load(cls, path):
        path_doc_seq, path_word_dic, path_doc_dic, path_mysim = cls._create_path(cls, path)
        with open(path_doc_seq, 'rb') as f:
            import pickle
            doc_seq = pickle.load(f)
        mysim = MySparseMatrixSimilarity.load(path_mysim)
        word_dic = gensim.corpora.dictionary.Dictionary.load(path_word_dic)
        doc_dic = gensim.corpora.dictionary.Dictionary.load(path_doc_dic)
        
        wd2v = WordAndDoc2vec(doc_seq, word_dic, doc_dic, load=True)
        wd2v.doc_seq = doc_seq
        wd2v.word_dic = word_dic
        wd2v.doc_dic = doc_dic
        wd2v.mysim = mysim
        wd2v.tfidf = mysim.tfidf
        wd2v.dic4seq = Dic4seq(mysim, doc_dic, word_dic)
        return wd2v
    
    def init(self):
        if self.logging:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    
    def create_tfidf(self):
        print('### creating tfidf...')
        sentences = Sentences(self.doc_seq, self.word_dic)
        
        tfidf = gensim.models.TfidfModel((self.word_dic.doc2bow(ee) for ee in self.doc_seq), id2word=self.word_dic)
        self.tfidf = tfidf
    
    def make_model(self, num_features=8, max_num_prod=30,
                   gamma=0.0, embeddings_val=0.1,
                   debug=False
                  ):
        self.num_user = len(self.doc_seq)
        self.num_product = self.mysim.index.shape[1]
        #self.max_num_prod = self.dic4seq.max_count
        self.max_num_prod = max_num_prod
        #self.num_neg = num_neg
        self.num_features = num_features
        
        models = make_model(num_user=self.num_user, num_product=self.num_product, max_num_prod=self.max_num_prod,
                   num_features=num_features, gamma=gamma,
                   embeddings_val=embeddings_val, debug=debug)
        self.models = models
        self.model = models['model']
        return models
    
    def get_seq(self, batch_size=32, shuffle=False, state=None):
        '''
        "shuffle" not implemented yet
        '''
        seq = Seq(self.dic4seq,
                  #num_neg=self.num_neg,
                  max_num_prod=self.max_num_prod,
                  batch_size=batch_size,
                  shuffle=shuffle, state=state)
        return seq
    
    def train(self, epochs=50, batch_size=32, verbose=1,
              use_multiprocessing=False, workers=1,
              callbacks=None):
        self.seq = self.get_seq(batch_size)
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
        self.hst = self.model.fit_generator(self.seq2, steps_per_epoch=len(self.seq),
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
        return get_sim(self.wgt_row, self.doc_dic, self.wgt_col, self.word_dic)
    
#        wgt_col = self.wgt_col
#        wgt_col_unit = np.zeros(shape=wgt_col.shape)
#        for ii in range(wgt_col_unit.shape[0]):
#            wgt_col_unit[ii] = gensim.matutils.unitvec(wgt_col[ii].copy())
#        
#        wgt_row = self.wgt_row
#        wgt_row_unit = np.zeros(shape=wgt_row.shape)
#        for ii in range(wgt_row_unit.shape[0]):
#            wgt_row_unit[ii] = gensim.matutils.unitvec(wgt_row[ii].copy())
#        
#        sim = WordAndDocSimilarity(wgt_row_unit, self.doc_dic, wgt_col_unit, self.word_dic)
#        return sim
    sim = property(get_sim)


class Similarity(gensim.similarities.MatrixSimilarity):
    
    def get_similarities(self, query):
        is_corpus, query = utils.is_corpus(query)
        if is_corpus:
            query = numpy.asarray(
                [matutils.sparse2full(vec, self.num_features) for vec in query],
                dtype=self.index.dtype
            )
        else:
            if scipy.sparse.issparse(query):
                query = query.toarray()  # convert sparse to dense
            elif isinstance(query, np.ndarray):
                pass
            else:
                # default case: query is a single vector in sparse gensim format
                query = matutils.sparse2full(query, self.num_features)
            query = np.asarray(query, dtype=self.index.dtype)
        
        # do a little transposition dance to stop numpy from making a copy of
        # self.index internally in numpy.dot (very slow).
        #result = numpy.dot(self.index, query.T).T  # return #queries x #index
        dist2 = euclidean_distances(self.index, query.reshape(1,-1), squared=True).reshape((1,-1))
        gamma = 1./(2.*self.index.shape[1]*0.1)
        #gamma = 1.0
        result = np.exp(-gamma * dist2)
        return result  # XXX: removed casting the result from array to list; does anyone care?

class WordAndDocSimilarity(object):
    
    def __init__(self, wgt_row, row_dic, wgt_col, col_dic):
        self.wgt_row = wgt_row
        self.wgt_col = wgt_col
        self.row_dic = row_dic
        self.col_dic = col_dic
        self.num_features = wgt_row.shape[1]
        
        self.sim_row = Similarity(self.wgt_row, num_features=self.num_features)
        self.sim_col = Similarity(self.wgt_col, num_features=self.num_features)
    
    def _get_sim(self, sim, d, query, num_best=100):
        sim.num_best = None
        sim.normalize = False
        res = sim[query]
        res2 = gensim.matutils.scipy2sparse(scipy.sparse.csr_matrix(res))
        res2.sort(key=lambda x: x[1], reverse=True)
        if num_best is None:
            res3 = res2
        else:
            res3 = res2[:num_best]
        res4 = [(d[idx], v) for idx, v in res3]
        return res4
    
    def get_sim_bycol(self, query, num_best=100):
        return self._get_sim(self.sim_col, self.col_dic, query, num_best=num_best)
    
    def get_sim_byrow(self, query, num_best=100):
        return self._get_sim(self.sim_row, self.row_dic, query, num_best=num_best)

def get_sim(wgt_row, doc_dic, wgt_col, word_dic):
    sim = WordAndDocSimilarity(wgt_row, doc_dic, wgt_col, word_dic)
    return sim



