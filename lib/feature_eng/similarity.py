'''
Copyright (c) 2018 Norio Tamada
Released under the MIT license
https://github.com/darecophoenixx/wordroid.sblo.jp/blob/master/lib/feature_eng/LICENSE.md
'''




import itertools
import time

import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix
import gensim






#class MySparseMatrixSimilarity(gensim.similarities.docsim.SparseMatrixSimilarity):
#    
#    def __init__(self, corpus_csc, num_features=None, num_terms=None, num_docs=None, num_nnz=None,
#                 num_best=None, chunksize=500, dtype=np.float32, maintain_sparsity=False, use_getCorpusByDoc=False):
#        super(MySparseMatrixSimilarity, self).__init__(None, num_features=num_features, num_terms=num_terms, num_docs=num_docs, num_nnz=num_nnz,
#                 num_best=num_best, chunksize=chunksize, dtype=dtype, maintain_sparsity=maintain_sparsity)
#        self.index_csc = corpus_csc
#        if use_getCorpusByDoc:
#            self.index_csr = corpus_csc.tocsr()
#        self.normalize = False
#        self.method = None
#        self.SLOPE = 0.2
#        self.DTYPE = dtype
#        self.num_row, self.num_col = corpus_csc.shape
#
#        self.getDF()
#    
#    def getDF(self):
#        '''異なり語数'''
#        print('processing d_len')
#        nonzero = self.index_csc.nonzero()
#        tmp = pd.DataFrame(nonzero[0], columns=['idx'])
#        d_len = tmp.groupby('idx').size().values
#        print('processing idfs')
#        tmp = pd.DataFrame(nonzero[1], columns=['idx'])
#        self.d_len_col = tmp.groupby('idx').size().values
#        #self.idfs = np.log(self.num_row / tmp.groupby('idx').size().values)
#        self.idfs = np.log(self.num_row / self.d_len_col)
#        '''トータル語数'''
#        print('processing d_tot')
#        d_tot = self.index_csc.sum(axis=1).flatten()
#        print('processing log avg')
#        avg = 1 + np.log(d_tot / d_len)
#        print('processing norm')
#        d_norm = d_len.mean() + self.SLOPE * (d_len - d_len.mean())
#        self.df_stats = pd.DataFrame(np.c_[d_len.reshape((-1,1)), d_tot.reshape((-1,1)), avg.reshape((-1,1)), d_norm.reshape((-1,1))], columns=['d_len','d_tot','logavg','d_norm'])
#        print('self.df_stats.shape >', self.df_stats.shape)
#        
#    def _calc(self, naiseki, norm):
#        res = naiseki.multiply(1/norm)
#        return res
#    
#    def getCorpusByDoc(self, docid, method='WT_SMART'):
#        query = self.index_csr[docid].tocsr()
#        self.idx_word = query.indices
#        self.wgt_list = query.data
#        self.tgt_mat = self.index_csc[:,self.idx_word]
#        if method in ['WT_RAW', 'WT_TF', 'WT_TFIDF', 'WT_SMART',
#                      'WT_SMARTAW', 'WT_SMARTAW2', 'WT_SMARTAW3', 'WT_SMARTAW4',
#                      'WT_SMARTWA', 'WT_SMART2']:
#            res = self.calc_wq(self.tgt_mat, self.idx_word, self.wgt_list, method=method)
#            row_ind = np.zeros(shape=(res.data.shape[0],), dtype=int)
#            ret = scipy.sparse.csr_matrix((res.data, (row_ind, self.idx_word)), shape=(1, self.index_csc.shape[1]))
#            return ret
#        else:
#            raise Exception('no such method [%s]' % method)
#    
#    def calc_idf(self, idx_word):
#        idf = self.idfs[idx_word]
#        return idf
#    
#    def create_query_mat(self, wgt_list):
#        return csr_matrix(np.array(wgt_list).reshape((1,-1)), dtype=self.DTYPE)
#    
#    def calc_smartaw_wq_bunbo(self, tgt_mat):
#        '''
#        a / b : 1単語あたりの平均度数
#        a : 各列の合計度数
#        b : 各列の0でない要素の数
#        '''
#        a = np.array(tgt_mat.sum(axis=0)).flatten() # 各列の合計度数
#        nonzero = tgt_mat.nonzero()[1]
#        b = np.array([(nonzero==ii).sum() for ii in range(tgt_mat.shape[1])]) # 各列の0でない要素の数
#        return a, b
#    
#    def calc_wq(self, tgt_mat, idx_word, wgt_list, method='WT_SMART'):
#        mat_csr_q = self.create_query_mat(wgt_list)
#        if method in ['WT_SMART']:
#            '''idfを考慮'''
#            wq = self.calc_tfn_mx(mat_csr_q, avg=True)
#            idf = self.calc_idf(idx_word)
#            wq = wq.multiply(idf) # tf_n(t|q) * idf(t)
#            return wq
#        elif method in ['WT_SMARTAW', 'WT_SMARTAWtama', 'WT_tama', 'WT_tama2', 'WT_tama3']:
#            wq = self.calc_tfn_mx(mat_csr_q, avg=False)
#            sum_freq_bycol, cnt_bycol = self.calc_smartaw_wq_bunbo(tgt_mat)
#            m = 1 + np.log(sum_freq_bycol / cnt_bycol)
#            wq = wq.multiply(1 / m)
#            return wq
#        elif method == 'WT_SMARTAW2':
#            '''idfを考慮'''
#            wq = self.calc_tfn_mx(mat_csr_q, avg=False)
#            sum_freq_bycol, cnt_bycol = self.calc_smartaw_wq_bunbo(tgt_mat)
#            m = sum_freq_bycol / cnt_bycol
#            wq = wq.multiply(m)
#            idf = self.calc_idf(idx_word)
#            wq = wq.multiply(idf) # tf_n(t|q) * idf(t)
#            return wq
#        elif method in ['WT_SMARTAW3', 'WT_SMARTAW4', 'WT_SMARTAW3tama']:
#            wq = mat_csr_q.copy()
#            nonzero_idx = wq.nonzero()
#            wq[nonzero_idx] = 1+np.log(wq[nonzero_idx])
#            return wq
#        elif method == 'WT_SMART2':
#            '''idfを考慮'''
#            wq = self.calc_tfn_mx2(mat_csr_q, avg=True)
#            idf = self.calc_idf(idx_word)
#            wq = wq.multiply(idf) # tf_n(t|q) * idf(t)
#            return wq
#        elif method in ['WT_RAW', 'WT_TF', 'WT_SMARTWA', 'WT_SMARTWAtama']:
#            wq = mat_csr_q
#            return wq
#        elif method == 'WT_TFIDF':
#            wq = mat_csr_q
#            idf = self.calc_idf(idx_word)
#            wq = wq.multiply(idf) # tf_n(t|q) * idf(t)
#            return wq
#        else:
#            raise Exception('no such method [%s]' % method)
#    
#    def calc_wd(self, tgt_mat, method='WT_SMART'):
#        if method in ['WT_SMART', 'WT_SMARTWA']:
#            wd = self.calc_tfn_mx(tgt_mat, avg=False)
#            m = self.df_stats['logavg'].values.reshape((-1,1))
#            wd = wd.multiply(1 / m)
#            return wd
#        elif method in ['WT_SMARTWAtama', 'WT_tama']:
#            '''WT_SMARTWA'''
#            wd = self.calc_tfn_mx(tgt_mat, avg=False)
#            m = self.df_stats['logavg'].values.reshape((-1,1))
#            wd = wd.multiply(1 / m)
#            '''最大重みを1にする'''
#            wd = wd.multiply(1.0/wd.max(axis=0).toarray())
#            return wd
#        elif method in ['WT_SMARTAW']:
#            '''idfを考慮'''
#            wd = self.calc_tfn_mx(tgt_mat, avg=False)
#            idf = np.log(self.num_row / self.df_stats['d_len'].values).reshape((-1,1))
#            wd = wd.multiply(idf)
#            return wd
#        elif method in ['WT_SMARTAWtama', 'WT_tama2']:
#            '''idfを考慮'''
#            wd = self.calc_tfn_mx(tgt_mat, avg=False)
#            idf = np.log(self.num_col / self.df_stats['d_len'].values).reshape((-1,1))
#            wd = wd.multiply(idf)
#            '''最大重みを1にする'''
#            wd = wd.multiply(1.0/wd.max(axis=0).toarray())
#            return wd
#        elif method in ['WT_SMARTAW2']:
#            wd = self.calc_tfn_mx(tgt_mat, avg=False)
#            return wd
#        elif method in ['WT_SMARTAW3', 'WT_SMARTAW4']:
#            '''idfを考慮'''
#            wd = self.calc_tfn_mx(tgt_mat, avg=False)
#            a, b = self.calc_smartaw_wq_bunbo(tgt_mat)
#            m = (a / b).reshape((1,-1))
#            m1 = (1 + np.log(m))
#            wd = wd.multiply(1/m1)
#            idf = np.log(self.num_col / self.df_stats['d_len'].values).reshape((-1,1))
#            wd = wd.multiply(idf)
#            return wd
#        elif method in ['WT_SMARTAW3tama']:
#            '''idfを考慮'''
#            wd = self.calc_tfn_mx(tgt_mat, avg=False)
#            a, b = self.calc_smartaw_wq_bunbo(tgt_mat)
#            m = (a / b).reshape((1,-1))
#            m1 = (1 + np.log(m))
#            wd = wd.multiply(1/m1)
#            idf = np.log(self.num_col / self.df_stats['d_len'].values).reshape((-1,1))
#            wd = wd.multiply(idf)
#            '''最大重みを1にする'''
#            wd = wd.multiply(1.0/wd.max(axis=0).toarray())
#            return wd
#        elif method in ['WT_tama3']:
#            '''idfを考慮'''
#            wd = self.calc_tfn_mx(tgt_mat, avg=False)
#            idf = np.log(self.num_col / self.df_stats['d_len'].values).reshape((-1,1))
#            wd = wd.multiply(idf)
#            m = self.df_stats['logavg'].values.reshape((-1,1))
#            wd = wd.multiply(1 / m)
#            '''最大重みを1にする'''
#            wd = wd.multiply(1.0/wd.max(axis=0).toarray())
#            return wd
#        elif method == 'WT_SMART2':
#            wd = self.calc_tfn_mx2(tgt_mat)
#            return wd
#        elif method in ['WT_RAW', 'WT_TF', 'WT_TFIDF']:
#            return tgt_mat
#        else:
#            raise Exception('no such method [%s]' % method)
#    
#    def calc_norm(self, wq, wd, method='WT_SMART'):
#        if method in ['WT_SMART', 'WT_SMART2', 'WT_SMARTWA']:
#            norm = self.df_stats['d_norm'].values.reshape((-1,1))
#            return norm
#        elif method in ['WT_SMARTAW', 'WT_SMARTAW2', 'WT_SMARTAW3', 'WT_SMARTAW4']:
#            '''クエリーの個数を返す'''
#            norm = wq.shape[1]
#            return norm
#        elif method in ['WT_tama', 'WT_tama2', 'WT_tama3', 'WT_SMARTAWtama', 'WT_SMARTAW3tama', 'WT_SMARTWAtama']:
#            '''クエリーの合計を返す'''
#            norm = wq.sum()
#            return norm
#        elif method in ['WT_TF', 'WT_TFIDF']:
#            return self.df_stats['d_tot'].values.reshape((-1,1))
#        elif method in ['WT_RAW']:
#            return 1.0
#        else:
#            raise Exception('no such method [%s]' % method)
#    
#    def calc_tfn_mx(self, mx, avg=False):
#        '''
#                    1 + log(TF(t|q))
#        tf_n(t|q) = ---------------------
#                    1 + log(ave(TF(.|q)))
#        '''
#        mat = mx.copy()
#        nonzero_idx = mat.nonzero()
#        mat[nonzero_idx] = 1+np.log(mat[nonzero_idx])
#        if avg:
#            m = 1 + np.log(mx.mean(axis=1))
#            return mat.multiply(1 / m)
#        else:
#            return mat
#    
#    def calc_tfn_mx2(self, mx, avg=False):
#        '''
#                    log(1 + TF(t|q))
#        tf_n(t|q) = ---------------------
#                    ave(log(1 + TF(.|q)))
#        '''
#        mat = mx.copy()
#        mat = mat.log1p()
#        if avg:
#            m = mat.mean(axis=1)
#            return mat.multiply(1 / m)
#        else:
#            return mat
#    
#    #def calc_sim_WT(self, tgt_mat, idx_word, wgt_list, method='WT_SMART'):
#    def calc_sim_WT(self, idx_word, wgt_list, method='WT_SMART'):
#        '''
#        sim(d|q) = 1 / norm(d) * \sum_t { wq(t|q) * wd(t|d) }
#        '''
#        tgt_mat = self.tgt_mat = self.index_csc[:,idx_word]
#
#        '''
#        wq
#        '''
#        wq = self.calc_wq(tgt_mat, idx_word, wgt_list, method=method)
#        
#        '''
#        wd
#        '''
#        wd = self.calc_wd(tgt_mat, method=method)
#        
#        '''inner product'''
#        naiseki = wd.dot(wq.T)
#        
#        '''
#        norm
#        norm(d) = ave(len(.)) + slope * (len(d) - ave(len(.)))
#        '''
#        norm = self.calc_norm(wq, wd, method=method)
#        ret = self._calc(naiseki, norm)
#        return ret
#    
#    def calc_sim(self, query):
#        self.idx_word = query.indices
#        self.wgt_list = query.data
#
#        if self.method is None:
#            ret = self.calc_sim_WT(self.idx_word, self.wgt_list, method='WT_SMART')
#            return ret
#        elif self.method in ['WT_RAW', 'WT_TFIDF', 'WT_TF', 'WT_SMART', 'WT_SMART2',
#                             'WT_SMARTAW', 'WT_SMARTAW2', 'WT_SMARTAW3', 'WT_SMARTAW4',
#                             'WT_SMARTAWtama', 'WT_SMARTAW3tama',
#                             'WT_SMARTWA', 'WT_SMARTWAtama',
#                             'WT_tama', 'WT_tama2', 'WT_tama3']:
#            ret = self.calc_sim_WT(self.idx_word, self.wgt_list, method=self.method)
#            return ret
#        elif self.method == 'WT_MINE':
#            return self.calc_sim_WT_MINE(query)
#        else:
#            raise Exception('no such method [%s]' % self.method)
#    
#    def get_similarities(self, query):
#        is_corpus, query = gensim.utils.is_corpus(query)
#        if is_corpus:
#            query = gensim.matutils.corpus2csc(query, self.index_csc.shape[1], dtype=self.index_csc.dtype)
#        else:
#            if scipy.sparse.issparse(query):
#                query = query.T  # convert documents=rows to documents=columns
#            elif isinstance(query, np.ndarray):
#                if query.ndim == 1:
#                    query.shape = (1, len(query))
#                query = scipy.sparse.csr_matrix(query, dtype=self.index_csc.dtype).T
#            else:
#                # default case: query is a single vector, in sparse gensim format
#                query = gensim.matutils.corpus2csc([query], self.index_csc.shape[1], dtype=self.index_csc.dtype)
#        # compute cosine similarity against every other document in the collection
#        #result = self.index * query.tocsc()  # N x T * T x C = N x C
#        self.query = query
#        result = self.calc_sim(query)
#        if result.shape[1] == 1 and not is_corpus:
#            # for queries of one document, return a 1d array
#            result = result.toarray().flatten()
#        elif self.maintain_sparsity:
#            # avoid converting to dense array if maintaining sparsity
#            result = result.T
#        else:
#            # otherwise, return a 2d matrix (#queries x #index)
#            result = result.toarray().T
#        return result
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
        self.tama = False
        self.SLOPE = 0.2
        self.DTYPE = dtype
        self.num_row, self.num_col = corpus_csc.shape

        self.getDF()
    
    def getDF(self):
        nonzero = self.index_csc.nonzero()
        '''列側'''
        d_tot_col = np.array(self.index_csc.sum(axis=0)).flatten() # 各列の合計度数
        print('processing idfs')
        tmp = pd.DataFrame(nonzero[1], columns=['idx'])
        d_len_col = tmp.groupby('idx').size().values
        self.idfs = np.log(self.num_row / d_len_col)
        print('self.idfs.shape >', self.idfs.shape)
        self.d_stat_col = {
            'd_len_col': d_len_col,
            'd_tot_col': d_tot_col,
            'idfs': self.idfs,
        }
        '''行側'''
        '''異なり語数'''
        print('processing d_len')
        tmp = pd.DataFrame(nonzero[0], columns=['idx'])
        d_len = tmp.groupby('idx').size().values
        '''トータル語数'''
        print('processing d_tot')
        d_tot = self.index_csc.sum(axis=1).flatten()
        print('processing log avg')
        avg = 1 + np.log(d_tot / d_len)
        print('processing norm')
        d_norm = d_len.mean() + self.SLOPE * (d_len - d_len.mean())
        self.df_stats = pd.DataFrame(np.c_[d_len.reshape((-1,1)), d_tot.reshape((-1,1)), avg.reshape((-1,1)), d_norm.reshape((-1,1))], columns=['d_len','d_tot','logavg','d_norm'])
        print('self.df_stats.shape >', self.df_stats.shape)
        
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
    
    def calc_smartaw_wq_bunbo(self, idx_word):
        '''
        a / b : 1単語あたりの平均度数
        a : 各列の合計度数
        b : 各列の0でない要素の数
        '''
        a = self.d_stat_col['d_tot_col'][idx_word] # 各列の合計度数
        b = self.d_stat_col['d_len_col'][idx_word] # 各列の0でない要素の数
        return a, b
    
    def calc_wq(self, tgt_mat, idx_word, wgt_list, method='WT_SMART'):
        mat_csr_q = self.create_query_mat(wgt_list)
        if method in ['WT_SMART']:
            '''idfを考慮'''
            wq = self.calc_tfn_mx(mat_csr_q, avg=True)
            idf = self.calc_idf(idx_word)
            wq = wq.multiply(idf) # tf_n(t|q) * idf(t)
            return wq
        elif method in ['WT_SMARTAW']:
            wq = self.calc_tfn_mx(mat_csr_q, avg=False)
            sum_freq_bycol, cnt_bycol = self.calc_smartaw_wq_bunbo(idx_word)
            m = 1 + np.log(sum_freq_bycol / cnt_bycol)
            wq = wq.multiply(1 / m)
            return wq
        elif method == 'WT_SMARTAW2':
            '''idfを考慮'''
            wq = self.calc_tfn_mx(mat_csr_q, avg=False)
            sum_freq_bycol, cnt_bycol = self.calc_smartaw_wq_bunbo(idx_word)
            #m = sum_freq_bycol / cnt_bycol
            m = 1 + np.log(sum_freq_bycol / cnt_bycol)
            wq = wq.multiply(m)
            idf = self.calc_idf(idx_word)
            wq = wq.multiply(idf) # tf_n(t|q) * idf(t)
            return wq
        elif method in ['WT_SMARTAW3']:
            wq = mat_csr_q.copy()
            nonzero_idx = wq.nonzero()
            wq[nonzero_idx] = 1+np.log(wq[nonzero_idx])
            a, b = self.calc_smartaw_wq_bunbo(idx_word)
            m = a / b
            m1 = (1 + np.log(m))
            wq = wq.multiply(1/m1)
            return wq
        elif method == 'WT_SMART2':
            '''idfを考慮'''
            wq = self.calc_tfn_mx2(mat_csr_q, avg=True)
            idf = self.calc_idf(idx_word)
            wq = wq.multiply(idf) # tf_n(t|q) * idf(t)
            return wq
        elif method in ['WT_RAW', 'WT_TF', 'WT_SMARTWA', 'WT_SMARTWA2']:
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
        if method in ['WT_SMART', 'WT_SMARTWA', 'WT_SMARTWA2', 'WT_SMARTAW', 'WT_SMARTAW2', 'WT_SMARTAW3']:
            wd = self.calc_tfn_mx(tgt_mat, avg=False)
            return wd
        elif method == 'WT_SMART2':
            wd = self.calc_tfn_mx2(tgt_mat, avg=False)
            return wd
        elif method in ['WT_RAW', 'WT_TF', 'WT_TFIDF']:
            return tgt_mat
        else:
            raise Exception('no such method [%s]' % method)
    
    def calc_norm(self, wq, wd, method='WT_SMART'):
        '''行にわたる数値'''
        if method in ['WT_SMART', 'WT_SMART2', 'WT_SMARTWA']:
            norm0 = self.df_stats['d_norm'].values.reshape((-1,1))
        else:
            norm0 = 1.0
        
        if method in ['WT_SMARTAW']:
            '''for WT_SMARTAW'''
            idf = np.log(self.num_row / self.df_stats['d_len'].values).reshape((-1,1))
        elif method in ['WT_SMARTAW2', 'WT_SMARTAW3', 'WT_SMARTWA2']:
            idf = np.log(self.num_col / self.df_stats['d_len'].values).reshape((-1,1))
        else:
            idf = 1.0
        
        if method in ['WT_SMART', 'WT_SMART2', 'WT_SMARTAW2', 'WT_SMARTWA', 'WT_SMARTWA2']:
            smart_m = self.df_stats['logavg'].values.reshape((-1,1))
        else:
            smart_m = 1.0
        
        if method in ['WT_TF', 'WT_TFIDF']:
            d_tot = self.df_stats['d_tot'].values.reshape((-1,1))
        else:
            d_tot = 1.0
        norm1 = norm0 * d_tot * smart_m / idf
        return norm1
        
        '''重み調整'''
        if method in ['WT_SMARTAW', 'WT_SMARTAW2', 'WT_SMARTAW3']:
            '''クエリーの個数を返す'''
            norm = wq.shape[1]
            return norm1 * norm
        elif method in ['WT_tama', 'WT_tama2', 'WT_tama3', 'WT_SMARTAWtama', 'WT_SMARTAW3tama', 'WT_SMARTWAtama']:
            '''クエリーの合計を返す'''
            norm = wq.sum()
            return norm1 * norm
#        elif method in ['WT_TF', 'WT_TFIDF']:
#            return self.df_stats['d_tot'].values.reshape((-1,1))
        else:
            return norm1
    
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
        tgt_mat = self.tgt_mat = self.index_csc[:,idx_word]

        '''
        wq
        '''
        now0 = time.perf_counter()
        wq = self.calc_wq(tgt_mat, idx_word, wgt_list, method=method)
        print('wq >', time.perf_counter() - now0)
        
        '''
        wd
        '''
        now0 = time.perf_counter()
        wd = self.calc_wd(tgt_mat, method=method)
        print('wd >', time.perf_counter() - now0)
        
        '''inner product'''
        #naiseki = wd.dot(wq.T)
        
        '''
        norm
        norm(d) = ave(len(.)) + slope * (len(d) - ave(len(.)))
        '''
        now0 = time.perf_counter()
        norm = self.calc_norm(wq, wd, method=method)
        print('norm >', time.perf_counter() - now0)
        #ret = self._calc(naiseki, norm)
        if self.tama:
            ret0 = wd.multiply(1/norm)
            ret0 = ret0.multiply(1.0/ret0.max(axis=0).toarray())
            ret = ret0.dot(wq.T) / wq.sum()
        else:
            ret = wd.dot(wq.T).multiply(1/norm)
        return ret
    
    def calc_sim(self, query):
        self.idx_word = query.indices
        self.wgt_list = query.data

        if self.method is None:
            ret = self.calc_sim_WT(self.idx_word, self.wgt_list, method='WT_SMART')
            return ret
        elif self.method in ['WT_RAW', 'WT_TFIDF', 'WT_TF', 'WT_SMART', 'WT_SMART2',
                             'WT_SMARTAW', 'WT_SMARTAW2', 'WT_SMARTAW3',
                             'WT_SMARTWA', 'WT_SMARTWA2',
                            ]:
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

class MySparseMatrixSimilarity2(gensim.similarities.docsim.SparseMatrixSimilarity):
    
    def __init__(self, corpus_csc, num_features=None, num_terms=None, num_docs=None, num_nnz=None,
                 num_best=None, chunksize=500, dtype=np.float32, maintain_sparsity=False, use_getCorpusByDoc=False):
        super().__init__(None, num_features=num_features, num_terms=num_terms, num_docs=num_docs, num_nnz=num_nnz,
                 num_best=num_best, chunksize=chunksize, dtype=dtype, maintain_sparsity=maintain_sparsity)
        self.index_csc = corpus_csc
        if use_getCorpusByDoc:
            self.index_csr = corpus_csc.tocsr()
        self.normalize = False
        self.method = None
        self.tama = False
        self.SLOPE = 0.2
        self.DTYPE = dtype
        self.num_row, self.num_col = corpus_csc.shape

        self.getDF()
    
    def getDF(self):
        nonzero = self.index_csc.nonzero()
        '''列側'''
        d_tot_col = np.array(self.index_csc.sum(axis=0)).flatten() # 各列の合計度数
        print('processing idfs')
        tmp = pd.DataFrame(nonzero[1], columns=['idx'])
        d_len_col = tmp.groupby('idx').size().values
        self.idfs = np.log(self.num_row / d_len_col)
        print('self.idfs.shape >', self.idfs.shape)
        self.d_stat_col = {
            'd_len_col': d_len_col,
            'd_tot_col': d_tot_col,
            'idfs': self.idfs,
        }
        '''行側'''
        '''異なり語数'''
        print('processing d_len')
        tmp = pd.DataFrame(nonzero[0], columns=['idx'])
        d_len = tmp.groupby('idx').size().values
        '''トータル語数'''
        print('processing d_tot')
        d_tot = self.index_csc.sum(axis=1).flatten()
        print('processing log avg')
        avg = 1 + np.log(d_tot / d_len)
        print('processing norm')
        d_norm = d_len.mean() + self.SLOPE * (d_len - d_len.mean())
        self.df_stats = pd.DataFrame(np.c_[d_len.reshape((-1,1)), d_tot.reshape((-1,1)), avg.reshape((-1,1)), d_norm.reshape((-1,1))], columns=['d_len','d_tot','logavg','d_norm'])
        print('self.df_stats.shape >', self.df_stats.shape)
        
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
    
    def calc_smartaw_wq_bunbo(self, idx_word):
        '''
        a / b : 1単語あたりの平均度数
        a : 各列の合計度数
        b : 各列の0でない要素の数
        '''
        a = self.d_stat_col['d_tot_col'][idx_word] # 各列の合計度数
        b = self.d_stat_col['d_len_col'][idx_word] # 各列の0でない要素の数
        return a, b
    
    def calc_wq(self, tgt_mat, idx_word, wgt_list,
                wq_method='tfn_mx',
                wq_norm_method='logavg',
                wq_idf_method='idf'):
        mat_csr_q = self.create_query_mat(wgt_list)
        if wq_method in ['tfn_mx']:
            wq = self.calc_tfn_mx(mat_csr_q, avg=True)
        elif wq_method in ['tfn_mx2']:
            wq = self.calc_tfn_mx2(mat_csr_q, avg=True)
        else:
            wq = mat_csr_q
        
        if wq_idf_method in ['idf']:
            idf = self.calc_idf(idx_word)
        else:
            idf = 1.0
        
        if wq_norm_method in ['logavg']:
            sum_freq_bycol, cnt_bycol = self.calc_smartaw_wq_bunbo(idx_word)
            m = 1 + np.log(sum_freq_bycol / cnt_bycol)
        elif wq_norm_method in ['logavg2']:
            sum_freq_bycol, cnt_bycol = self.calc_smartaw_wq_bunbo(idx_word)
            m = 1 / (1 + np.log(sum_freq_bycol / cnt_bycol))
        else:
            m = 1.0
        
        wq = wq.multiply(idf).multiply(1 / m)
        return wq
    
    def calc_wd(self, tgt_mat, wd_method='tfn_mx'):
        if wd_method in ['tfn_mx']:
            wd = self.calc_tfn_mx(tgt_mat, avg=False)
            return wd
        elif wd_method in ['tfn_mx2']:
            wd = self.calc_tfn_mx2(tgt_mat, avg=False)
            return wd
        else:
            return tgt_mat
    
    def calc_norm(self, wq, wd,
                  norm_method='d_norm',
                  norm_idf_method='idf',
                  norm_dtot_method='d_tot',
                  norm_smartm_method='logavg'):
        if norm_method in ['d_norm']:
            norm0 = self.df_stats['d_norm'].values.reshape((-1,1))
        else:
            norm0 = 1.0
        
        if norm_idf_method in ['idf2']:
            idf = np.log(self.num_row / self.df_stats['d_len'].values).reshape((-1,1))
        elif norm_idf_method in ['idf']:
            idf = np.log(self.num_col / self.df_stats['d_len'].values).reshape((-1,1))
        else:
            idf = 1.0
        
        if norm_smartm_method in ['logavg']:
            smart_m = self.df_stats['logavg'].values.reshape((-1,1))
        elif norm_smartm_method in ['logavg2']:
            smart_m = 1 / self.df_stats['logavg'].values.reshape((-1,1))
        else:
            smart_m = 1.0
        
        if norm_dtot_method in ['d_tot']:
            d_tot = self.df_stats['d_tot'].values.reshape((-1,1))
        else:
            d_tot = 1.0
        norm = norm0 * d_tot * smart_m / idf
        return norm
    
    def calc_tfn_mx(self, mx, avg=False):
        '''
                    1 + log(TF(t|q))
        tf_n(t|q) = ---------------------
                    1 + log(ave(TF(.|q)))
        '''
        mat = mx.copy()
        nonzero_idx = mat.nonzero()
        mat[nonzero_idx] = 1 + np.log(mat[nonzero_idx])
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
    
    def calc_sim_WT(self, idx_word, wgt_list, method='WT_SMART',
                    wq_method=None, wq_idf_method=None, wq_norm_method=None,
                    wd_method=None,
                    norm_method=None, norm_idf_method=None, norm_dtot_method=None, norm_smartm_method=None):
        '''
        sim(d|q) = 1 / norm(d) * \sum_t { wq(t|q) * wd(t|d) }
        '''
        tgt_mat = self.tgt_mat = self.index_csc[:,idx_word]

        '''
        ==============================
        === wq ===
        ==============================
        '''
        if method in ['WT_SMART']:
            wq_method = 'tfn_mx'
            wq_idf_method = 'idf'
            wq_norm_method = None
        elif method in ['WT_SMART2']:
            wq_method = 'tfn_mx2'
            wq_idf_method = 'idf'
            wq_norm_method = None
        elif method in ['WT_SMARTAW']:
            wq_method = 'tfn_mx'
            wq_idf_method = None
            wq_norm_method = 'logavg'
        elif method in ['WT_SMART2AW']:
            wq_method = 'tfn_mx2'
            wq_idf_method = None
            wq_norm_method = 'logavg'
        elif method in ['WT_SMARTAW2']:
            wq_method = 'tfn_mx'
            wq_idf_method = 'idf'
            wq_norm_method = 'logavg'
        elif method in ['WT_SMARTAW3']:
            wq_method = 'tfn_mx'
            wq_idf_method = 'idf'
            wq_norm_method = None
        elif method in ['WT_SMART2AW2']:
            wq_method = 'tfn_mx2'
            wq_idf_method = 'idf'
            wq_norm_method = 'logavg'
        elif method in ['WT_SMART2AW3', 'WT_SMART2AW4']:
            wq_method = 'tfn_mx2'
            wq_idf_method = 'idf'
            wq_norm_method = None
        elif method in ['WT_RAW', 'WT_TF', 'WT_SMARTWA', 'WT_SMART2WA', 'WT_SMARTWA2', 'WT_SMART2WA2']:
            wq_method = None
            wq_idf_method = None
            wq_norm_method = None
        elif method in ['WT_TFIDF']:
            wq_method = None
            wq_idf_method = 'idf'
            wq_norm_method = None
        elif method is None:
            pass
        else:
            raise Exception('no such method [%s]' % method)
        
        now0 = time.perf_counter()
        wq = self.calc_wq(tgt_mat, idx_word, wgt_list,
                          wq_method=wq_method, wq_idf_method=wq_idf_method, wq_norm_method=wq_norm_method)
        print('wq >', time.perf_counter() - now0)
        
        '''
        ==============================
        === wd ===
        ==============================
        '''
        if method in ['WT_SMART', 'WT_SMARTAW', 'WT_SMARTAW2', 'WT_SMARTAW3', 'WT_SMARTWA', 'WT_SMARTWA2']:
            wd_method = 'tfn_mx'
        elif method in ['WT_SMART2', 'WT_SMART2AW', 'WT_SMART2AW2', 'WT_SMART2AW3', 'WT_SMART2AW4', 'WT_SMART2WA', 'WT_SMART2WA2']:
            wd_method = 'tfn_mx2'
        elif method in ['WT_RAW', 'WT_TF', 'WT_TFIDF']:
            wd_method = None
        elif method is None:
            pass
        else:
            raise Exception('no such method [%s]' % method)
        
        now0 = time.perf_counter()
        wd = self.calc_wd(tgt_mat, wd_method=wd_method)
        print('wd >', time.perf_counter() - now0)
        
        '''
        ==============================
        === norm ===
        ==============================
        norm(d) = ave(len(.)) + slope * (len(d) - ave(len(.)))
        '''
        if method in ['WT_SMART', 'WT_SMART2', 'WT_SMARTWA', 'WT_SMART2WA', 'WT_SMART2AW4']:
            norm_method = 'd_norm'
            norm_idf_method = None
            norm_dtot_method = None
            norm_smartm_method = 'logavg'
        elif method in ['WT_SMARTAW', 'WT_SMART2AW']:
            norm_method = None
            #norm_idf_method = 'idf2'
            norm_idf_method = 'idf'
            norm_dtot_method = None
            norm_smartm_method = None
        elif method in ['WT_SMARTAW2', 'WT_SMART2AW2',
                        'WT_SMARTAW3', 'WT_SMART2AW3',
                        'WT_SMARTWA2', 'WT_SMART2WA2']:
            norm_method = None
            norm_idf_method = 'idf'
            norm_dtot_method = None
            norm_smartm_method = 'logavg'
        elif method in ['WT_RAW']:
            norm_method = None
            norm_idf_method = None
            norm_dtot_method = None
            norm_smartm_method = None
        elif method in ['WT_TF', 'WT_TFIDF']:
            norm_method = None
            norm_idf_method = None
            norm_dtot_method = 'd_tot'
            norm_smartm_method = None
        elif method is None:
            pass
        else:
            raise Exception('no such method [%s]' % method)
        
        now0 = time.perf_counter()
        norm = self.calc_norm(wq, wd,
                              norm_method=norm_method,
                              norm_idf_method=norm_idf_method,
                              norm_dtot_method=norm_dtot_method,
                              norm_smartm_method=norm_smartm_method)
        print('norm >', time.perf_counter() - now0)
        
        if self.tama:
            ret0 = wd.multiply(1/norm)
            ret0 = ret0.multiply(1.0/ret0.max(axis=0).toarray())
            ret = ret0.dot(wq.T) / wq.sum()
        else:
            ret = wd.dot(wq.T).multiply(1/norm)
        return ret
    
    def calc_sim(self, query):
        self.idx_word = query.indices
        self.wgt_list = query.data

        if self.method is None:
            ret = self.calc_sim_WT(self.idx_word, self.wgt_list, method='WT_SMART')
            return ret
        elif self.method in ['WT_RAW', 'WT_TFIDF', 'WT_TF', 'WT_SMART', 'WT_SMART2',
                             'WT_SMARTAW', 'WT_SMARTAW2', 'WT_SMARTAW3',
                             'WT_SMART2AW', 'WT_SMART2AW2', 'WT_SMART2AW3', 'WT_SMART2AW4',
                             'WT_SMARTWA', 'WT_SMART2WA', 'WT_SMARTWA2', 'WT_SMART2WA2',
                            ]:
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
    
    def get_sim_user(self, query, num_best=100000, method='WT_SMARTAW', thresh=0.1, tama=False):
        self.sim_user.num_best = num_best
        self.sim_user.method = method
        self.sim_user.tama = tama
        res = self.sim_user[query]
        res1 = cut(res, thresh=thresh)
        return res1
    
    def get_sim_word(self, query, num_best=100000, method='WT_SMARTAW', thresh=0.1, tama=False):
        self.sim_word.num_best = num_best
        self.sim_word.method = method
        self.sim_word.tama = tama
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

