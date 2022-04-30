'''
Copyright (c) 2018 Norio Tamada
Released under the MIT license
https://github.com/darecophoenixx/wordroid.sblo.jp/blob/master/lib/feature_eng/LICENSE.md
'''




import itertools

import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix
import gensim






class MySparseMatrixSimilarity(gensim.similarities.docsim.SparseMatrixSimilarity):
    
    def __init__(self, corpus_csc, num_features=None, num_terms=None, num_docs=None, num_nnz=None,
                 num_best=None, chunksize=500, dtype=np.float32, maintain_sparsity=False, use_getCorpusByDoc=False):
        super(MySparseMatrixSimilarity, self).__init__(None, num_features=num_features, num_terms=num_terms, num_docs=num_docs, num_nnz=num_nnz,
                 num_best=num_best, chunksize=chunksize, dtype=dtype, maintain_sparsity=maintain_sparsity)
        self.index_csc = corpus_csc
        if use_getCorpusByDoc:
            self.index_csr = corpus_csc.tocsr()
        self.normalize = False
        self.method = None
        self.SLOPE = 0.2
        self.DTYPE = dtype
        self.num_row, self.num_col = corpus_csc.shape

        self.getDF()
    
    def getDF(self):
        '''異なり語数'''
        print('processing d_len')
        nonzero = self.index_csc.nonzero()
        tmp = pd.DataFrame(nonzero[0], columns=['idx'])
        d_len = tmp.groupby('idx').size().values
        print('processing idfs')
        tmp = pd.DataFrame(nonzero[1], columns=['idx'])
        self.idfs = np.log(self.num_row / tmp.groupby('idx').size().values)
        '''トータル語数'''
        print('processing d_tot')
        d_tot = self.index_csc.sum(axis=1).flatten()
        print('processing log avg')
        avg = 1 + np.log(d_tot / d_len)
        print('processing norm')
        d_norm = d_len.mean() + self.SLOPE * (d_len - d_len.mean())
        self.df_stats = pd.DataFrame(np.c_[d_len.reshape((-1,1)), d_tot.reshape((-1,1)), avg.reshape((-1,1)), d_norm.reshape((-1,1))], columns=['d_len','d_tot','logavg','d_norm'])
        
    def _calc(self, naiseki, norm):
        res = naiseki.multiply(1/norm)
        return res
    
    def getCorpusByDoc(self, docid, method='WT_SMART'):
        query = self.index_csr[docid].tocsr()
        self.idx_word = query.indices
        self.wgt_list = query.data
        self.tgt_mat = self.index_csc[:,self.idx_word]
        if method in ['WT_RAW', 'WT_TF', 'WT_TFIDF', 'WT_SMART',
                      'WT_SMARTAW', 'WT_SMARTAW2', 'WT_SMARTAW3', 'WT_SMARTAW4',
                      'WT_SMARTWA', 'WT_SMART2']:
            res = self.calc_wq(self.tgt_mat, self.idx_word, self.wgt_list, method=method)
            row_ind = np.zeros(shape=(res.data.shape[0],), dtype=int)
            ret = scipy.sparse.csr_matrix((res.data, (row_ind, self.idx_word)), shape=(1, self.index_csc.shape[1]))
            return ret
        else:
            raise Exception('no such method [%s]' % method)
    
    def calc_idf(self, idx_word):
        idf = self.idfs[idx_word]
        return idf
    
    def create_query_mat(self, wgt_list):
        return csr_matrix(np.array(wgt_list).reshape((1,-1)), dtype=self.DTYPE)
    
    def calc_wq(self, tgt_mat, idx_word, wgt_list, method='WT_SMART'):
        mat_csr_q = self.create_query_mat(wgt_list)
        if method in ['WT_SMART']:
            wq = self.calc_tfn_mx(mat_csr_q, avg=True)
            idf = self.calc_idf(idx_word)
            wq = wq.multiply(idf) # tf_n(t|q) * idf(t)
            return wq
        elif method == 'WT_SMARTAW':
            wq = self.calc_tfn_mx(mat_csr_q, avg=False)
            a = np.array(tgt_mat.sum(axis=0)).flatten()
            nonzero = tgt_mat.nonzero()[1]
            b = np.array([(nonzero==ii).sum() for ii in range(tgt_mat.shape[1])])
            m = 1 + np.log(a / b)
            wq = wq.multiply(1 / m)
            return wq
        elif method == 'WT_SMARTAW2':
            wq = self.calc_tfn_mx(mat_csr_q, avg=False)
            a = np.array(tgt_mat.sum(axis=0)).flatten()
            nonzero = tgt_mat.nonzero()[1]
            b = np.array([(nonzero==ii).sum() for ii in range(tgt_mat.shape[1])])
            m = a / b
            wq = wq.multiply(m)
            idf = self.calc_idf(idx_word)
            wq = wq.multiply(idf) # tf_n(t|q) * idf(t)
            return wq
        elif method in ['WT_SMARTAW3', 'WT_SMARTAW4']:
            wq = mat_csr_q.copy()
            nonzero_idx = wq.nonzero()
            wq[nonzero_idx] = 1+np.log(wq[nonzero_idx])
            return wq
        elif method == 'WT_SMARTWA':
            wq = mat_csr_q
            return wq
        elif method == 'WT_SMART2':
            wq = self.calc_tfn_mx2(mat_csr_q, avg=True)
            idf = self.calc_idf(idx_word)
            wq = wq.multiply(idf) # tf_n(t|q) * idf(t)
            return wq
        elif method in ['WT_RAW', 'WT_TF']:
            wq = mat_csr_q
            return wq
        elif method == 'WT_TFIDF':
            wq = mat_csr_q
            idf = self.calc_idf(idx_word)
            wq = wq.multiply(idf) # tf_n(t|q) * idf(t)
            return wq
        else:
            raise Exception('no such method [%s]' % method)
    
    def calc_wd(self, tgt_mat, method='WT_SMART'):
        if method in ['WT_SMART', 'WT_SMARTWA']:
            wd = self.calc_tfn_mx(tgt_mat, avg=False)
            m = self.df_stats['logavg'].values.reshape((-1,1))
            wd = wd.multiply(1 / m)
            return wd
        elif method in ['WT_SMARTAW']:
            wd = self.calc_tfn_mx(tgt_mat, avg=False)
            idf = np.log(self.num_row / self.df_stats['d_len'].values).reshape((-1,1))
            wd = wd.multiply(idf)
            return wd
        elif method in ['WT_SMARTAW2']:
            wd = self.calc_tfn_mx(tgt_mat, avg=False)
            return wd
        elif method in ['WT_SMARTAW3', 'WT_SMARTAW4']:
            wd = self.calc_tfn_mx(tgt_mat, avg=False)
            a = np.array(tgt_mat.sum(axis=0)).flatten()
            nonzero = tgt_mat.nonzero()[1]
            b = np.array([(nonzero==ii).sum() for ii in range(tgt_mat.shape[1])])
            m = (a / b).reshape((1,-1))
            m1 = (1 + np.log(m))
            wd = wd.multiply(1/m1)
            idf = np.log(self.num_col / self.df_stats['d_len'].values).reshape((-1,1))
            wd = wd.multiply(idf)
            return wd
        elif method == 'WT_SMART2':
            wd = self.calc_tfn_mx2(mx)
            return wd
        elif method in ['WT_RAW', 'WT_TF', 'WT_TFIDF']:
            return tgt_mat
        else:
            raise Exception('no such method [%s]' % method)
    
    def calc_norm(self, wq, wd, method='WT_SMART'):
        if method in ['WT_SMART', 'WT_SMARTWA']:
            norm = self.df_stats['d_norm'].values.reshape((-1,1))
            return norm
        elif method in ['WT_SMARTAW', 'WT_SMARTAW2', 'WT_SMARTAW3', 'WT_SMARTAW4']:
            norm = wq.shape[1]
            return norm
        elif method == 'WT_SMART2':
            norm = 1 / ((1-self.SLOPE)*self.DF.mean() + self.SLOPE*self.DF[nonzero_idx2])
            return norm
        elif method in ['WT_TF', 'WT_TFIDF']:
            return self.df_stats['d_tot'].values.reshape((-1,1))
        elif method in ['WT_RAW']:
            return 1.0
        else:
            raise Exception('no such method [%s]' % method)
    
    def calc_tfn_mx(self, mx, avg=False):
        '''
                    1 + log(TF(t|q))
        tf_n(t|q) = ---------------------
                    1 + log(ave(TF(.|q)))
        '''
        mat = mx.copy()
        nonzero_idx = mat.nonzero()
        mat[nonzero_idx] = 1+np.log(mat[nonzero_idx])
        if avg:
            m = 1 + np.log(mx.mean(axis=1))
            return mat.multiply(1 / m)
        else:
            return mat
    
    def calc_tfn_mx2(self, mx, avg=False):
        '''
                    log(1 + TF(t|q))
        tf_n(t|q) = ---------------------
                    ave(log(1 + TF(.|q)))
        '''
        mat = mx.copy()
        mat = mat.log1p()
        if avg:
            m = mat.mean(axis=1)
            return mat.multiply(1 / m)
        else:
            return mat
    
    #def calc_sim_WT(self, tgt_mat, idx_word, wgt_list, method='WT_SMART'):
    def calc_sim_WT(self, idx_word, wgt_list, method='WT_SMART'):
        '''
        sim(d|q) = 1 / norm(d) * \sum_t { wq(t|q) * wd(t|d) }
        '''
        tgt_mat = self.tgt_mat = self.index_csc[:,self.idx_word]

        '''
        wq
        '''
        wq = self.calc_wq(tgt_mat, idx_word, wgt_list, method=method)
        
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
        norm = self.calc_norm(wq, wd, method=method)
        ret = self._calc(naiseki, norm)
        return ret
    
    def calc_sim(self, query):
        self.idx_word = query.indices
        self.wgt_list = query.data
#        self.tgt_mat = self.index_csc[:,self.idx_word]

        if self.method is None:
            ret = self.calc_sim_WT(self.idx_word, self.wgt_list, method='WT_SMART')
            return ret
        elif self.method in ['WT_TFIDF', 'WT_TF', 'WT_SMART', 'WT_RAW',
                             'WT_SMARTAW', 'WT_SMARTAW2', 'WT_SMARTAW3', 'WT_SMARTAW4',
                             'WT_SMARTWA', 'WT_SMART2']:
            ret = self.calc_sim_WT(self.idx_word, self.wgt_list, method=self.method)
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


def cut(res0, thresh=0.1):
    max_val = max(list(zip(*res0))[1])
    res = []
    for ee in res0:
        idx, w = ee
        if max_val*thresh <= w:
            res.append((int(ee[0]), float(ee[1])))
    return res

class Collaborative(object):
    
    def __init__(self, dic_user, dic_word, sim_user, sim_word):
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
    
    def get_sim_user(self, query, num_best=100000, method='WT_SMARTAW', thresh=0.1):
        self.sim_user.num_best = num_best
        self.sim_user.method = method
        res = self.sim_user[query]
        res1 = cut(res, thresh=thresh)
        return res1
    
    def get_sim_word(self, query, num_best=100000, method='WT_SMARTAW', thresh=0.1):
        self.sim_word.num_best = num_best
        self.sim_word.method = method
        res = self.sim_word[query]
        res1 = cut(res, thresh=thresh)
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

