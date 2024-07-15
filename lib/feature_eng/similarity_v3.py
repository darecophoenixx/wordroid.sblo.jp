'''
Copyright (c) 2024 Norio Tamada
Released under the MIT license
https://github.com/darecophoenixx/wordroid.sblo.jp/blob/master/lib/feature_eng/LICENSE.md
'''




import sys, re
import logging

import numpy as np
import pandas as pd
import scipy
import gensim
from tqdm import tqdm
from scipy.sparse import csc_matrix, csr_matrix



class MySparseMatrixSimilarity3(gensim.similarities.docsim.SparseMatrixSimilarity):
    '''
    d_normを先に作用させておく
    '''
    def __init__(self, smart_csc, idfs, d_norm,
                 num_features=None, num_terms=None, num_docs=None, num_nnz=None,
                 num_best=None, chunksize=500, dtype=np.float32, maintain_sparsity=False):
        super(MySparseMatrixSimilarity3, self).__init__(None, num_features=num_features, num_terms=num_terms, num_docs=num_docs, num_nnz=num_nnz,
                 num_best=num_best, chunksize=chunksize, dtype=dtype, maintain_sparsity=maintain_sparsity)
        assert smart_csc.shape[1] == idfs.shape[1]
        assert smart_csc.shape[0] == d_norm.shape[0]
        self.index_csc = smart_csc
        self.idfs = idfs
        self.d_norm = d_norm
        self.method = None
        #self.SLOPE = 0.2
        self.DTYPE = dtype
        self.num_row, self.num_col = smart_csc.shape

        #self.getDF()

    # def getDF(self):
    #     '''異なり語数'''
    #     print('processing d_len')
    #     nonzero = self.index_csc.nonzero()
    #     tmp = pd.DataFrame(nonzero[0], columns=['idx'])
    #     d_len = tmp.groupby('idx').size().values
    #     print('processing idfs')
    #     tmp = pd.DataFrame(nonzero[1], columns=['idx'])
    #     self.idfs = np.log(self.num_row / tmp.groupby('idx').size().values)
    #     '''トータル語数'''
    #     print('processing d_tot')
    #     d_tot = self.index_csc.sum(axis=1).flatten()
    #     print('processing log avg') # １文書あたり１単語当たりの平均出現度数
    #     avg = 1 + np.log(d_tot / d_len)
    #     print('processing norm')
    #     d_norm = d_len.mean() + self.SLOPE * (d_len - d_len.mean()) # (1-SLOPE)*d_len.mean() + SLOPE*d_len
    #     #self.df_stats = pd.DataFrame(np.c_[d_len.reshape((-1,1)), d_tot.reshape((-1,1)), avg.reshape((-1,1)), d_norm.reshape((-1,1))], columns=['d_len','d_tot','logavg','d_norm'])
    #     #self.logavg = self.df_stats['logavg'].values.reshape((-1,1))
    #     self.logavg = avg.reshape((-1,1))
    #     #self.d_norm = self.df_stats['d_norm'].values.reshape((-1,1))
    #     self.d_norm = d_norm.reshape((-1,1))
    #     self.freq_max = self.index_csc.max(axis=1)
    #     print('re-calc index_csc')
    #     self.index_csc = self.calc_tfn_mx(self.index_csc, avg=False)
    #     print('re-calc index_csc 001')
    #     self.index_csc = self.index_csc.multiply(1 / self.logavg)
    #     print('re-calc index_csc 002')
    #     self.index_csc = self.index_csc.multiply(self.idfs) # tf_n(t|q) * idf(t)
    #     print('re-calc index_csc 003')
    #     max_index_csc = self.index_csc.max(axis=1).todense()
    #     print('re-calc index_csc 004')
    #     self.index_csc = self.index_csc.multiply(1.0/max_index_csc)
    #     print('re-calc index_csc 005')
    #     self.index_csc = self.index_csc.tocsc()

    def _calc(self, naiseki, norm):
        res = naiseki.multiply(1/norm)
        return res

    def getCorpusByDoc(self, docid):
        query = self.index_csc[docid].tocsr()
        return query

    def calc_idf(self, tgt_mat):
        #idf = np.log(self.num_row / np.array([len(tgt_mat[:,ii].nonzero()[0]) for ii in range(tgt_mat.shape[1])]))
        idf = self.idfs[self.idx_word]
        return idf

    def create_query_mat(self, wgt_list):
        return csr_matrix(np.array(wgt_list).reshape((1,-1)), dtype=self.DTYPE)

    def calc_wq(self, tgt_mat, wgt_list, method='WT_SMART'):
        mat_csr_q = self.create_query_mat(wgt_list)
        if method in ['WT_SMART']:
            wq = self.calc_tfn_mx(mat_csr_q.copy(), avg=True)
            idf = self.calc_idf(tgt_mat)
            wq = wq.multiply(idf) # tf_n(t|q) * idf(t)
            return wq
        elif method == 'WT_SMARTWA':
            wq = mat_csr_q
            return wq
        else:
            raise Exception('no such method [%s]' % method)

    def calc_wd(self, tgt_mat, method='WT_SMART'):
        if method in ['WT_SMART', 'WT_SMARTWA']:
            return tgt_mat
        else:
            raise Exception('no such method [%s]' % method)

    def calc_norm(self, wq, wd, method='WT_SMART'):
        if method in ['WT_SMART', 'WT_SMARTWA']:
            norm = self.d_norm
            return norm
        else:
            raise Exception('no such method [%s]' % method)

    def calc_tfn_mx(self, mx, avg=False):
        '''
        元の度数マトリックスを変換する
                    1 + log(TF(t|q))
        tf_n(t|q) = ---------------------
                    1 + log(ave(TF(.|q)))
        '''
        #mat = mx.copy()
        mat = mx
        nonzero_idx = mat.nonzero()
        mat[nonzero_idx] = 1+np.log(mat[nonzero_idx])
        if avg:
            m = 1 + np.log(mx.mean(axis=1))
            return mat.multiply(1 / m)
        else:
            return mat

    def calc_sim_WT(self, tgt_mat, idx_word, wgt_list, method='WT_SMART'):
        '''
        sim(d|q) = 1 / norm(d) * \sum_t { wq(t|q) * wd(t|d) }
        '''
        '''
        wq
        '''
        wq = self.calc_wq(tgt_mat, wgt_list, method=method)

        '''
        wd
        '''
        wd = self.calc_wd(tgt_mat, method=method)

        '''inner product'''
        naiseki = wd.dot(wq.T)

        '''
        norm
        norm(d) = ave(len(.)) + slope * (len(d) - ave(len(.)))
        '''
        # norm = self.calc_norm(wq, wd, method=method)
        # ret = self._calc(naiseki, norm)
        ret = naiseki
        return ret

    def calc_sim(self, query):
        self.idx_word = query.indices
        self.wgt_list = query.data
        self.tgt_mat = self.index_csc[:,self.idx_word]

        if self.method is None:
            ret = self.calc_sim_WT(self.tgt_mat, self.idx_word, self.wgt_list, method='WT_SMART')
            return ret
        elif self.method in ['WT_TFIDF', 'WT_TF', 'WT_SMART', 'WT_RAW',
                             'WT_SMARTAW', 'WT_SMARTAW2', 'WT_SMARTAW3', 'WT_SMARTAW4',
                             'WT_SMARTWA', 'WT_SMARTb']:
            ret = self.calc_sim_WT(self.tgt_mat, self.idx_word, self.wgt_list, method=self.method)
            return ret
        elif self.method == 'WT_MINE':
            return self.calc_sim_WT_MINE(query)
        else:
            raise Exception('no such method [%s]' % self.method)

    def get_similarities(self, query):
        is_corpus, query = gensim.utils.is_corpus(query)
        if is_corpus:
            query = gensim.matutils.corpus2csc(query, self.index_csc.shape[1], dtype=self.index_csc.dtype)
        else:
            if scipy.sparse.issparse(query):
                query = query.T  # convert documents=rows to documents=columns
            elif isinstance(query, np.ndarray):
                if query.ndim == 1:
                    query.shape = (1, len(query))
                query = scipy.sparse.csr_matrix(query, dtype=self.index_csc.dtype).T
            else:
                # default case: query is a single vector, in sparse gensim format
                query = gensim.matutils.corpus2csc([query], self.index_csc.shape[1], dtype=self.index_csc.dtype)
        # compute cosine similarity against every other document in the collection
        #result = self.index * query.tocsc()  # N x T * T x C = N x C
        self.query = query
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

def cut(res0, shresh=0.1):
    max_val = max(list(zip(*res0))[1])
    res = []
    for ee in res0:
        idx, w = ee
        if max_val*shresh <= w:
            res.append((int(ee[0]), float(ee[1])))
    return res

class Collaborative(object):

    def __init__(self, dic_user, dic_word, sim_user, sim_word):
        assert sim_user.index_csc.shape[0] == sim_word.index_csc.shape[1]
        assert sim_user.index_csc.shape[1] == sim_word.index_csc.shape[0]
        assert len(dic_user) == sim_user.index_csc.shape[0]
        assert len(dic_word) == sim_word.index_csc.shape[0]
        self.dic_user = dic_user
        self.dic_word = dic_word
        self.sim_user = sim_user
        self.sim_word = sim_word

    def get_query_by_word(self, word_list):
        query = self.dic_word.doc2bow(word_list)
        return query

    def get_query_by_user(self, user_list):
        query = self.dic_user.doc2bow(user_list)
        return query

    def get_sim_user(self, query, num_best=100000, method='WT_SMARTAW', shresh=0.1):
        self.sim_user.num_best = num_best
        self.sim_user.method = method
        res = self.sim_user[query]
        res1 = cut(res, shresh=shresh)
        return res1

    def get_sim_word(self, query, num_best=100000, method='WT_SMARTAW', shresh=0.1):
        self.sim_word.num_best = num_best
        self.sim_word.method = method
        res = self.sim_word[query]
        res1 = cut(res, shresh=shresh)
        return res1

    def transform_user(self, res):
        ret = []
        for user_id, wgt in res:
            ret.append((self.dic_user[user_id], wgt))
        return ret

    def transform_word(self, res):
        ret = []
        for word_id, wgt in res:
            ret.append((self.dic_word[word_id], wgt))
        return ret
