'''
Copyright (c) 2018 Norio Tamada
Released under the MIT license
https://github.com/darecophoenixx/wordroid.sblo.jp/blob/master/lib/feature_eng/LICENSE.md
'''

import itertools
import random
from collections import Mapping
import logging

import numpy as np
import scipy
import gensim
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
from keras.constraints import maxnorm, non_neg
from keras.utils import Sequence
from keras import backend as K

__all__ = ['WordAndDoc2vec', ]
           

SIGMA2 = 0.5**2


class MySparseMatrixSimilarity(gensim.similarities.docsim.SparseMatrixSimilarity):
    DTYPE = np.float32
    
    def __init__(self, corpus_csr, tfidf, num_features=None, num_terms=None, num_docs=None, num_nnz=None,
                 num_best=None, chunksize=500, dtype=np.float32, maintain_sparsity=False):
        super(MySparseMatrixSimilarity, self).__init__(None, num_features=num_features, num_terms=num_terms, num_docs=num_docs, num_nnz=num_nnz,
                 num_best=num_best, chunksize=chunksize, dtype=dtype, maintain_sparsity=maintain_sparsity)
        self.index = corpus_csr
        self.index_csc = corpus_csr.tocsc()
        self.tfidf = tfidf
        self.normalize = False
        self.method = None
        self.SLOPE = 0.2
        
        idfs = np.empty((num_features,1), dtype=self.DTYPE)
        for termid in tfidf.idfs:
            v = tfidf.idfs[termid]
            idfs[termid,0] = v
        self.idfs = idfs
        
        self.getDF()
    
    def getDF(self):
        ll = []
        for ii in range(self.index.shape[0]):
            ll.append(self.index.getrow(ii).nnz)
        self.DF = np.array(ll, dtype=self.DTYPE)
        
    def get_nonzero_idx2(self, ix):
        tmp = self.index_csc[:,ix]
        nonzero_idx2 = list(set(tmp.nonzero()[0]))
        nonzero_idx2.sort()
        return nonzero_idx2
    
    def _calc(self, naiseki, norm, nonzero_idx2):
        res = naiseki.multiply(norm.reshape((norm.shape[0],1)))
        res2 = scipy.sparse.csr_matrix((self.index.shape[0], 1), dtype=self.DTYPE)
        res2[nonzero_idx2,:] = res
        return res2
    
    def getCorpusByDoc(self, docid, method='WT_SMART'):
        query = self.index[docid]
        if method in ['WT_TF', 'WT_TFIDF', 'WT_SMART', 'WT_SMART2']:
            return self.calc_wq(query, method=method)
        else:
            raise Exception('no such method [%s]' % method)
    
    def calc_wq(self, query, method='WT_SMART'):
        if query.count_nonzero() == 0:
            raise ValueError('freq must be more than zero')
        if method == 'WT_SMART':
            wq = self.calc_tfn_mx(query)
            wq = wq.T.multiply(self.idfs) # tf_n(t|q) * idf(t)
            return wq.T
        elif method == 'WT_SMART2':
            wq = self.calc_tfn_mx2(query)
            wq = wq.T.multiply(self.idfs) # tf_n(t|q) * idf(t)
            return wq.T
        elif method == 'WT_TF':
            wq = query.tocsc()
            return wq
        elif method == 'WT_TFIDF':
            wq = query.tocsc()
            wq = wq.T.multiply(self.idfs)
            return wq.T
        else:
            raise Exception('no such method [%s]' % method)
    
    def calc_wd(self, mx, method='WT_SMART'):
        if method == 'WT_SMART':
            wd = self.calc_tfn_mx(mx)
            return wd
        elif method == 'WT_SMART2':
            wd = self.calc_tfn_mx2(mx)
            return wd
        elif method in ['WT_TF', 'WT_TFIDF']:
            return mx
        else:
            raise Exception('no such method [%s]' % method)
    
    def calc_norm(self, wq, wd, nonzero_idx2, method='WT_SMART'):
        if method == 'WT_SMART':
            norm = 1 / ((1-self.SLOPE)*self.DF.mean() + self.SLOPE*self.DF[nonzero_idx2])
            return norm
        elif method == 'WT_SMART2':
            norm = 1 / ((1-self.SLOPE)*self.DF.mean() + self.SLOPE*self.DF[nonzero_idx2])
            return norm
        elif method in ['WT_TF', 'WT_TFIDF']:
            ll = []
            for ii in range(wd.shape[0]):
                ll.append(wd.getrow(ii).sum())
            norm = np.array(ll, dtype=self.DTYPE)
            norm = 1 / norm
            return norm
        else:
            raise Exception('no such method [%s]' % method)
    
    def calc_tfn_mx(self, mx):
        '''
                    1 + log(TF(t|q))
        tf_n(t|q) = ---------------------
                    1 + log(ave(TF(.|q)))
        '''
        sums = mx.sum(axis=1)
        sums = np.array(sums.reshape(-1).tolist()[0])
        nnz = []
        for ii in range(mx.shape[0]):
            nnz.append(mx.getrow(ii).count_nonzero())
        nnz = np.array(nnz)
        means = 1 + np.log(sums / nnz)
        nonzero_idx = mx.nonzero()
        mx[nonzero_idx] = mx[nonzero_idx] - 1
        mx = mx.log1p()
        mx[nonzero_idx] = (1 + mx[nonzero_idx])
        vs = []
        for ii in range(mx.shape[0]):
            vs.append(mx[ii,].multiply(1 / means[ii]))
        mx2 = scipy.sparse.vstack(vs)
        return mx2
    
    def calc_tfn_mx2(self, mx):
        '''
                    log(1 + TF(t|q))
        tf_n(t|q) = ---------------------
                    ave(log(1 + TF(.|q)))
        '''
        nnz = []
        for ii in range(mx.shape[0]):
            nnz.append(mx.getrow(ii).count_nonzero())
        nnz = np.array(nnz)
        
        mx = mx.log1p()
        sums = mx.sum(axis=1)
        sums = np.array(sums.reshape(-1).tolist()[0])
        means = sums / nnz
        
        vs = []
        for ii in range(mx.shape[0]):
            vs.append(mx[ii,].multiply(1 / means[ii]))
        mx2 = scipy.sparse.vstack(vs)
        return mx2
    
    def calc_sim_WT_MINE(self, query):
        '''wq'''
        wq = query.tocsc().multiply(self.idfs)
        '''wd'''
        wd = self.index
        '''inner product'''
        naiseki = wd * wq
        '''norm'''
        ones = np.ones(self.index.shape[1])
        norm = 1 / self.index.dot(ones)
        res = naiseki.multiply(norm.reshape((norm.shape[0],1)))
        return res
    
    def calc_sim_WT_TF(self, query):
        '''
        WQ
        wq(t|q) = TF(t|q)
        '''
        #wq = query.tocsc()
        wq = self.calc_wq(query.T, method='WT_TF').T
        nonzero_idx = wq.nonzero()
        
        '''
        WD
        wd(t|d) = TF(t|d)
        '''
        nonzero_idx2 = self.get_nonzero_idx2(nonzero_idx[0])
        wd = self.index[nonzero_idx2,:].copy()
        
        '''inner product'''
        naiseki = wd.dot(wq)
        
        '''
        norm
        norm(d) = TF(.|d)
        '''
        ll = []
        for ii in range(wd.shape[0]):
            ll.append(wd.getrow(ii).sum())
        norm = np.array(ll, dtype=self.DTYPE)
        norm = 1 / norm
        
        return self._calc(naiseki, norm, nonzero_idx2)
    
    def calc_sim_WT_TFIDF(self, query):
        '''
        WQ
        wq(t|q) = TF(t|q) * idf(t)
        '''
        wq = self.calc_wq(query.T, method='WT_TFIDF').T
        nonzero_idx = wq.nonzero()
        
        '''
        WD
        wd(t|d) = TF(t|d)
        '''
        nonzero_idx2 = self.get_nonzero_idx2(nonzero_idx[0])
        wd = self.index[nonzero_idx2,:].copy()
        
        '''inner product'''
        naiseki = wd.dot(wq)
        
        '''
        norm
        norm(d) = TF(.|d)
        '''
        ll = []
        for ii in range(wd.shape[0]):
            ll.append(wd.getrow(ii).sum())
        norm = np.array(ll, dtype=self.DTYPE)
        norm = 1 / norm
        
        return self._calc(naiseki, norm, nonzero_idx2)
    
    def calc_sim_WT_SMART(self, query):
        '''
        sim(d|q) = 1 / norm(d) * \sum_t { wq(t|q) * wd(t|d) }
        wq(t|q) = tf_n(t|q) * idf(t)
        wd(t|d) = tf_n(t|d)
        '''
        '''
        WQ
                    1 + log(TF(t|q))
        tf_n(t|q) = ---------------------
                    1 + log(ave(TF(.|q)))
        '''
        wq = self.calc_wq(query.T).T
        nonzero_idx = wq.nonzero()
        
        '''
        WD
        tf_n(t|d) = 1 + log(TF(t|d))
        '''
        nonzero_idx2 = self.get_nonzero_idx2(nonzero_idx[0])
        wd = self.index[nonzero_idx2,].astype(self.DTYPE)
        wd = self.calc_tfn_mx(wd)
        
        '''inner product'''
        naiseki = wd.dot(wq)
        
        '''
        norm
        norm(d) = ave(len(.)) + slope * (len(d) - ave(len(.)))
        '''
        norm = 1 / ((1-self.SLOPE)*self.DF.mean() + self.SLOPE*self.DF[nonzero_idx2])
        
        ret = self._calc(naiseki, norm, nonzero_idx2)
        return ret
    
    def calc_sim_WT(self, query, method='WT_SMART'):
        '''
        sim(d|q) = 1 / norm(d) * \sum_t { wq(t|q) * wd(t|d) }
        '''
        '''
        wq
        '''
        wq = self.calc_wq(query.T, method=method).T
        nonzero_idx = wq.nonzero()
        
        '''
        wd
        '''
        nonzero_idx2 = self.get_nonzero_idx2(nonzero_idx[0])
        wd = self.index[nonzero_idx2,].astype(self.DTYPE)
        wd = self.calc_wd(wd, method=method)
        
        '''inner product'''
        naiseki = wd.dot(wq)
        
        '''
        norm
        norm(d) = ave(len(.)) + slope * (len(d) - ave(len(.)))
        '''
        norm = self.calc_norm(wq, wd, nonzero_idx2, method=method)
        
        ret = self._calc(naiseki, norm, nonzero_idx2)
        return ret
    
    def calc_sim(self, query):
        if self.method is None:
            ret = self.calc_sim_WT(query, method='WT_SMART')
            return ret
        elif self.method in ['WT_TFIDF', 'WT_TF', 'WT_SMART', 'WT_SMART2']:
            ret = self.calc_sim_WT(query, method=self.method)
            return ret
        elif self.method == 'WT_MINE':
            return self.calc_sim_WT_MINE(query)
        else:
            raise Exception('no such method [%s]' % self.method)
    
    def get_similarities(self, query):
        is_corpus, query = gensim.utils.is_corpus(query)
        if is_corpus:
            query = gensim.matutils.corpus2csc(query, self.index.shape[1], dtype=self.index.dtype)
        else:
            if scipy.sparse.issparse(query):
                query = query.T  # convert documents=rows to documents=columns
            elif isinstance(query, np.ndarray):
                if query.ndim == 1:
                    query.shape = (1, len(query))
                query = scipy.sparse.csr_matrix(query, dtype=self.index.dtype).T
            else:
                # default case: query is a single vector, in sparse gensim format
                query = gensim.matutils.corpus2csc([query], self.index.shape[1], dtype=self.index.dtype)
        # compute cosine similarity against every other document in the collection
        #result = self.index * query.tocsc()  # N x T * T x C = N x C
        result = self.calc_sim(query)
        if result.shape[1] == 1 and not is_corpus:
            # for queries of one document, return a 1d array
            result = result.toarray().flatten()
        elif self.maintain_sparsity:
            # avoid converting to dense array if maintaining sparsity
            result = result.T
        else:
            # otherwise, return a 2d matrix (#queries x #index)
            result = result.toarray().T
        return result


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
        return self.mysim.index.shape[0]
    
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
    
    def __init__(self, dic4seq,
                 num_neg=3, batch_size=32, max_num_prod=5, shuffle=False, state=None):
        self.dic4seq = dic4seq
        self.shuffle = shuffle
        self.state = state
        self.max_num_prod = max_num_prod
        self.batch_size = batch_size
        self.num_neg = num_neg
        
        self.prod_pad = [0] * self.max_num_prod
        
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
        ret_user, ret_prods, ret_negs, ret_y = self.get_data(iuser)
        return ({'input_user': ret_user, 'input_prod': ret_prods, 'input_neg': ret_negs}, ret_y)
    
    def get_data(self, iuser):
        user_id = self.dic4seq.to_ids_row([iuser])[0]
        prods = self.dic4seq[iuser]
        prods_id = self.dic4seq.to_ids(prods)
        prods_id_group = [[ee1 if ee1 is not None else 0 for ee1 in ee] for ee in itertools.zip_longest(*[iter(prods_id)]*self.max_num_prod)]
        
        ret_y = [1]*self.max_num_prod + [0]*self.num_neg
        ret_prods_group = []
        ret_negs_group = []
        ret_y_group = []
        for iprods_id in prods_id_group:
            ret_y, ret_negs = self.get_neg(iprods_id)
            ret_prods_group.append(iprods_id)
            ret_negs_group.append(ret_negs)
            ret_y_group.append([1]*self.max_num_prod + ret_y)
        return ([user_id]*len(prods_id_group), ret_prods_group, ret_negs_group, ret_y_group)
    
    def get_neg(self, prods):
        neg = random.sample(self.smpl, self.num_neg*self.max_num_prod)
        neg = np.array(neg).reshape(self.max_num_prod, self.num_neg).tolist()
        ret_y = [0]*(self.num_neg*self.max_num_prod)
        return (ret_y, neg)
        
    def _neg(self, neg_smpl_num, prods):
        #random.seed()
        #return random.sample(self.smpl, neg_smpl_num)
        ret = []
        while 1:
            neg = random.sample(self.smpl, neg_smpl_num - len(ret))
            neg = set(neg).difference(prods)
            ret.extend(neg)
            if len(ret) == neg_smpl_num:
                break
        return ret
    
    def getpart(self, users_part):
        x_input_user = []
        x_input_prod = []
        x_input_neg = []
        y = []
        for iuser in users_part:
            x_train, y_train = self[iuser]
            x_input_prod.extend(x_train['input_prod'])
            x_input_neg.extend(x_train['input_neg'])
            x_input_user.extend(x_train['input_user'])
            y.extend(y_train)
        return ({
            'input_prod': np.array(x_input_prod),
            'input_neg': np.array(x_input_neg),
            'input_user': np.array(x_input_user)
            },
            np.array(y))


class Seq2(Sequence):
    
    def __init__(self, seq):
        self.seq = seq
    
    def __len__(self):
        return len(self.seq)
    
    def __getitem__(self, idx):
        bs = self.seq.batch_size
        user_part = self.seq.user_list[(idx*bs):((idx*bs+bs) if (idx*bs+bs)<len(self.seq.user_list) else len(self.seq.user_list))]
        res = self.seq.getpart(user_part)
        return res


def make_model(num_user=20, num_product=10, max_num_prod=5,
               num_neg=3, num_features=8, gamma=0.0,
               embeddings_val=0.5, sigma2=SIGMA2):

    user_embedding = Embedding(output_dim=num_features, input_dim=num_user,
                               embeddings_initializer=initializers.RandomUniform(minval=-embeddings_val, maxval=embeddings_val),
                               embeddings_regularizer=regularizers.l2(gamma),
                               name='user_embedding', trainable=True)
    prod_embedding = Embedding(output_dim=num_features, input_dim=num_product,
                               embeddings_initializer=initializers.RandomUniform(minval=-embeddings_val, maxval=embeddings_val),
                               embeddings_regularizer=regularizers.l2(gamma),
                               name='prod_embedding', trainable=True)

    input_user = Input(shape=(1,), name='input_user')
    input_prod = Input(shape=(max_num_prod,), name='input_prod')
    input_neg = Input(shape=(max_num_prod, num_neg), name='input_neg')

    embed_user = user_embedding(input_user)
    #print('embed_user >', K.int_shape(embed_user))
    embed_prod = prod_embedding(input_prod)
    #print('embed_prod >', K.int_shape(embed_prod))
    embed_neg = prod_embedding(input_neg)
    #print('embed_neg >', K.int_shape(embed_neg))

    model_user = Model(input_user, embed_user)
    model_prod = Model(input_prod, embed_prod)
    model_neg = Model(input_neg, embed_neg)

    input_user_vec = Input(shape=(1, num_features), name='input_user_vec')
    input_prod_vec = Input(shape=(max_num_prod, num_features), name='input_prod_vec')
    input_neg_vec = Input(shape=(max_num_prod, num_neg, num_features), name='input_neg_vec')

    def reshape_embed_prod(embed_prod):
        embed_prod2 = K.sum(K.square(embed_prod), axis=2)
        #print('embed_prod2 >', K.int_shape(embed_prod2))
        embed_prod2 = K.expand_dims(embed_prod2)
        #print('embed_prod2 >', K.int_shape(embed_prod2))
        return embed_prod2
    
    '''calc prob'''
    def calc_prob(x):
        embed_prod = x[0]
        embed_user = x[1]
        embed_prod2 = reshape_embed_prod(embed_prod)
        #print('embed_prod2 (in calc_prob) >', K.int_shape(embed_prod2))
        
        embed_user2 = K.sum(K.square(embed_user), axis=2)
        embed_user2 = K.expand_dims(embed_user2)
        embed_user2 = K.repeat_elements(embed_user2, max_num_prod, axis=1)
        #print('embed_user2 (in calc_prob) >', K.int_shape(embed_user2))
        
        embed_prod_x_embed_user = K.batch_dot(embed_prod, embed_user, axes=2)
        #print('embed_prod_x_embed_user (in calc_prob) >', K.int_shape(embed_prod_x_embed_user))
        d2 = embed_prod2 + embed_user2 - 2*embed_prod_x_embed_user
        d2 = K.squeeze(d2, axis=2)
        prob = K.exp(-1./(num_features*sigma2) * d2)
        return prob
    prob = Lambda(calc_prob, name='calc_prob')([embed_prod, embed_user])
    model_prob = Model([input_prod, input_user], prob)
    prob_cnfm = Lambda(calc_prob, name='prob_cnfm')([input_prod_vec, input_user_vec])
    model_prob_cnfm = Model([input_prod_vec, input_user_vec], prob_cnfm)
    
    '''calc prob2'''
    def calc_prob2(x):
        embed_prod = x[0]
        embed_neg = x[1]
        embed_prod2 = reshape_embed_prod(embed_prod)
        embed_prod2 = K.repeat_elements(embed_prod2, num_neg, axis=2)
        #print('embed_prod2 (in calc_prob2) >', K.int_shape(embed_prod2))
        
        embed_neg2 = K.sum(K.square(embed_neg), axis=3)
        #print('embed_neg2 (in calc_prob2) >', K.int_shape(embed_neg2))
        
        #print('embed_prod (in calc_prob2) >', K.int_shape(embed_prod))
        #print('embed_neg (in calc_prob2) >', K.int_shape(embed_neg))
        embed_prod_x_embed_neg = K.batch_dot(
            K.reshape(embed_prod, (-1, num_features)), 
            K.reshape(embed_neg, (-1, num_neg, num_features)), 
            axes=(1,2))
        #print('embed_prod_x_embed_neg (in calc_prob2) >', K.int_shape(embed_prod_x_embed_neg))
        embed_prod_x_embed_neg = K.reshape(embed_prod_x_embed_neg, (-1, max_num_prod, num_neg))
        #print('embed_prod_x_embed_neg (in calc_prob2) >', K.int_shape(embed_prod_x_embed_neg))
        d2 = embed_prod2 + embed_neg2 - 2*embed_prod_x_embed_neg
        prob2 = K.exp(-1./(num_features*sigma2) * d2)
        return prob2
    prob2 = Lambda(calc_prob2, name='calc_prob2')([embed_prod, embed_neg])
    model_prob2 = Model([input_neg, input_prod], prob2)
    prob2_cnfm = Lambda(calc_prob2, name='calc_prob2_cnfm')([input_prod_vec, input_neg_vec])
    model_prob2_cnfm = Model([input_prod_vec, input_neg_vec], prob2_cnfm)
    
    #print('prob >', K.int_shape(prob))
    #print('Flatten()(prob2) >', K.int_shape(Flatten()(prob2)))
    prob = concatenate([prob, Flatten()(prob2)])
    model = Model([input_user, input_prod, input_neg], prob)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    models = {
        'model': model,
        'model_neg': model_neg,
        'model_user': model_user,
        'model_prod': model_prod,
        'model_prob': model_prob,
        'model_prob2': model_prob2,
        'model_prob_cnfm': model_prob_cnfm,
        'model_prob2_cnfm': model_prob2_cnfm,
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
        
        self.create_tfidf()
        print('### creating MySparseMatrixSimilarity...')
        self.mysim = MySparseMatrixSimilarity(self.corpus_csr, num_features=num_features, tfidf=self.tfidf)
        print(self.mysim)
        
        print('### creating Dic4seq...')
        self.dic4seq = Dic4seq(self.mysim, self.doc_dic, self.word_dic)
        print(self.dic4seq)
#    def __init__(self,
#                 doc_seq, word_dic, doc_dic,
#                 logging=False,
#                 load=False
#                 ):
#        if load:
#            return
#        self.logging = logging
#        self.init()
#        
#        self.doc_seq = doc_seq
#        self.word_dic = word_dic
#        self.doc_dic = doc_dic
#        
#        print('len(doc_seq) >>>', len(doc_seq))
#        print('max(doc_dic.keys()) + 1 >>>', max(doc_dic.keys()) + 1)
#        if len(doc_seq) != (max(doc_dic.keys()) + 1):
#            raise IlligalDocIndexException('num of doc_seq is [%s]. But doc_dic is [%s].' % (len(doc_seq), max(doc_dic.keys()) + 1))
#        
#        num_features = max(self.word_dic.keys()) + 1
#        print('num_features >>>', num_features)
#        
#        print('### creating corpus_csr...')
#        #corpus_csr = gensim.matutils.corpus2csc((self.word_dic.doc2bow(ee) for ee in self.doc_seq), num_terms=len(self.word_dic)).T
#        corpus_csr = gensim.matutils.corpus2csc((self.word_dic.doc2bow(ee) for ee in self.doc_seq), num_terms=num_features).T
#        print('corpus_csr.shape >>>', corpus_csr.shape)
#        
#        self.create_tfidf()
#        print('### creating MySparseMatrixSimilarity...')
#        #self.mysim = MySparseMatrixSimilarity(corpus_csr, num_features=len(self.word_dic), tfidf=self.tfidf)
#        self.mysim = MySparseMatrixSimilarity(corpus_csr, num_features=num_features, tfidf=self.tfidf)
#        print(self.mysim)
#        
#        print('### creating Dic4seq...')
#        self.dic4seq = Dic4seq(self.mysim, self.doc_dic, self.word_dic)
#        print(self.dic4seq)
    
    def _create_path(self, path):
        path_doc_seq = path + '_doc_seq'
        path_word_dic = path + '_word_dic'
        path_doc_dic = path + '_doc_dic'
        path_mysim = path + '_mysim'
        return path_doc_seq, path_word_dic, path_doc_dic, path_mysim
    
#    def save(self, path):
#        path_doc_seq, path_word_dic, path_doc_dic, path_mysim = self._create_path(path)
#        with open(path_doc_seq, 'wb') as fw:
#            import pickle
#            pickle.dump(self.doc_seq, fw)
#        self.word_dic.save(path_word_dic)
#        self.doc_dic.save(path_doc_dic)
#        self.mysim.save(path_mysim)
#    
#    @classmethod
#    def load(cls, path):
#        path_doc_seq, path_word_dic, path_doc_dic, path_mysim = cls._create_path(cls, path)
#        with open(path_doc_seq, 'rb') as f:
#            import pickle
#            doc_seq = pickle.load(f)
#        mysim = MySparseMatrixSimilarity.load(path_mysim)
#        word_dic = gensim.corpora.dictionary.Dictionary.load(path_word_dic)
#        doc_dic = gensim.corpora.dictionary.Dictionary.load(path_doc_dic)
#        
#        wd2v = WordAndDoc2vec(doc_seq, word_dic, doc_dic, load=True)
#        wd2v.doc_seq = doc_seq
#        wd2v.word_dic = word_dic
#        wd2v.doc_dic = doc_dic
#        wd2v.mysim = mysim
#        wd2v.tfidf = mysim.tfidf
#        wd2v.dic4seq = Dic4seq(mysim, doc_dic, word_dic)
#        return wd2v
    
    def init(self):
        if self.logging:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    
    def create_tfidf(self):
        print('### creating tfidf...')
        tfidf = gensim.models.TfidfModel(
            (gensim.matutils.scipy2sparse(ee) for ee in self.corpus_csr),
            id2word=self.word_dic
        )
        self.tfidf = tfidf
#    def create_tfidf(self):
#        print('### creating tfidf...')
#        sentences = Sentences(self.doc_seq, self.word_dic)
#        
#        tfidf = gensim.models.TfidfModel((self.word_dic.doc2bow(ee) for ee in self.doc_seq), id2word=self.word_dic)
#        self.tfidf = tfidf
    
    def make_model(self, max_num_prod=5, num_neg=3, num_features=8,
                   gamma=0.0, embeddings_val=0.5):
        #self.num_user = len(self.doc_seq)
        self.num_user = self.corpus_csr.shape[0]
        self.num_product = self.mysim.index.shape[1]
        self.max_num_prod = max_num_prod
        self.num_neg = num_neg
        self.num_features = num_features
        
        models = make_model(num_user=self.num_user, num_product=self.num_product, max_num_prod=max_num_prod,
                   num_neg=num_neg, num_features=num_features, gamma=gamma,
                   embeddings_val=embeddings_val)
        self.models = models
        self.model = models['model']
        return models
    
    def get_seq(self, batch_size=32, shuffle=False, state=None):
        '''
        "shuffle" not implemented yet
        '''
        seq = Seq(self.dic4seq,
                  num_neg=self.num_neg,
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
