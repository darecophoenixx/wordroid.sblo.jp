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
                 num_best=None, chunksize=500, dtype=np.float32, maintain_sparsity=False):
        super(MySparseMatrixSimilarity, self).__init__(None, num_features=num_features, num_terms=num_terms, num_docs=num_docs, num_nnz=num_nnz,
                 num_best=num_best, chunksize=chunksize, dtype=dtype, maintain_sparsity=maintain_sparsity)
        self.index_csc = corpus_csc
        self.normalize = False
        self.method = None
        self.SLOPE = 0.2
        self.DTYPE = dtype
        self.num_row, self.num_col = corpus_csc.shape

        self.getDF()
    
    def getDF(self):
        '''異なり語数'''
        print('processing d_len')
        tmp = pd.DataFrame(self.index_csc.nonzero()[0], columns=['idx'])
        d_len = tmp.groupby('idx').size().values
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
        query = self.index_csc[docid].tocsr()
        self.idx_word = query.indices
        self.wgt_list = query.data
        self.tgt_mat = self.index_csc[:,self.idx_word]
        if method in ['WT_RAW', 'WT_TF', 'WT_TFIDF', 'WT_SMART',
                      'WT_SMARTAW', 'WT_SMARTAW2', 'WT_SMARTAW3', 'WT_SMARTAW4',
                      'WT_SMARTWA', 'WT_SMART2']:
            return self.calc_wq(self.tgt_mat, self.wgt_list, method=method)
        else:
            raise Exception('no such method [%s]' % method)
    
    def calc_idf(self, tgt_mat):
        idf = np.log(self.num_row / np.array([len(tgt_mat[:,ii].nonzero()[0]) for ii in range(tgt_mat.shape[1])]))
        return idf
    
    def create_query_mat(self, wgt_list):
        return csr_matrix(np.array(wgt_list).reshape((1,-1)), dtype=self.DTYPE)
    
    def calc_wq(self, tgt_mat, wgt_list, method='WT_SMART'):
        mat_csr_q = self.create_query_mat(wgt_list)
        if method in ['WT_SMART']:
            wq = self.calc_tfn_mx(mat_csr_q, avg=True)
            idf = self.calc_idf(tgt_mat)
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
            idf = self.calc_idf(tgt_mat)
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
            wq = self.calc_tfn_mx2(query)
            wq = wq.T.multiply(self.idfs) # tf_n(t|q) * idf(t)
            return wq.T
        elif method in ['WT_RAW', 'WT_TF']:
            wq = mat_csr_q
            return wq
        elif method == 'WT_TFIDF':
            wq = mat_csr_q
            idf = self.calc_idf(tgt_mat)
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
        norm = self.calc_norm(wq, wd, method=method)
        ret = self._calc(naiseki, norm)
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
                             'WT_SMARTWA', 'WT_SMART2']:
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

