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






def join_colab_wgt(df_som_xy, user_wgt):
    '''
    df_som_xyに、重みをジョインする
    最大重みを分母にして、基準化する
    最大重みが1になる
    '''
    max_wgt = max(list(zip(*user_wgt))[1])
    df_wgt = pd.DataFrame(user_wgt, columns=['user_id', 'wgt'])
    df_wgt.index = df_wgt['user_id']
    df_wgt['wgt'] = df_wgt['wgt'].values / max_wgt
    df_som_xy_tmp = pd.merge(df_som_xy, df_wgt, left_index=True, right_index=True, how='left')
    df_som_xy_tmp.drop(['user_id'], inplace=True, axis=1)
    df_som_xy_tmp = df_som_xy_tmp.fillna({'wgt': 0.0})
    return df_som_xy_tmp

class SomXYpCollab(object):
    
    THRESH = 0.0
    MAX_DEG = 20

    def __init__(self, word_list, collab, df_som_xy):
        self.word_list = word_list
        self.collab = collab
        self.df_som_xy = df_som_xy
        self.cls_cat = list(np.unique(df_som_xy['cls'].values)) # cls一覧
        self.cls_cat.sort(key=lambda x: int(x.replace('cls', '')))
        self.query = collab.get_query_by_word(word_list)
        print('self.query', self.query)

        self.calc_xy_uniq()
    
    def calc_xy_uniq(self):
        '''
        各クラスのxy座標をユニークにする
        '''
        l_res = []
        for ii in self.cls_cat:
            res = self.df_som_xy.query('cls=="{}"'.format(ii)).iloc[[0],:]
            l_res.append(res)
        self.df_som_xy_uni = pd.concat(l_res, axis=0)
        self.df_som_xy_uni['cls'] = pd.Categorical(self.df_som_xy_uni['cls'], categories=self.cls_cat)
        self.df_som_xy_uni.index = self.df_som_xy_uni.cls

    def get_users1(self, num_best=1000000, method='WT_SMART2', thresh=THRESH):
        if not hasattr(self, '_users1'):
            self._users1 = self.collab.get_sim_user(self.query, num_best=num_best, method=method, thresh=thresh)
            print('len(self._users1) >', len(self._users1))
        return self._users1
    
    @property
    def users1(self):
        return self.get_users1()
    
    def get_words1(self, num_best=1000000, method='WT_SMART2WA', thresh=THRESH):
        if not hasattr(self, '_words1'):
            self._words1 = self.collab.get_sim_word(self.users1, num_best=num_best, method=method, thresh=thresh)
            print('len(self._words1) >', len(self._words1))
        return self._words1
    
    @property
    def words1(self):
        return self.get_words1()
    
    def get_users2(self, num_best=1000000, method='WT_SMART2WA', thresh=THRESH, bunbo=1000):
        if not hasattr(self, '_users2'):
            self._users2 = self.collab.get_sim_user(self.words1[:int(len(self.words1)/bunbo)], num_best=num_best, method=method, thresh=thresh)
            print('len(self._users2) >', len(self._users2))
        return self._users2
    
    @property
    def users2(self):
        return self.get_users2()
    
    def get_somxy_user(self, users, df_som_xy, max_deg=MAX_DEG):
        '''no use'''
        max_wgt = max(list(zip(*users))[1])
        idx = [ii for ii, wgt in users for _ in range(int(wgt / max_wgt * max_deg))]
        # ユーザーがデュプリケイトされる。結果[各クラスの重み]が計算される
        print('len(idx) >', len(idx), idx[:5])
        self.df_som_xy2 = df_som_xy.iloc[idx,:].copy()
        print('df_som_xy2.shape >', self.df_som_xy2.shape)
        return self.df_som_xy2
    
    def get_df_wgt(self, users, drop=True):
        '''
        df_som_xyにユーザーの重みをジョインする
        '''
        self.df_som_xy_wgt = join_colab_wgt(self.df_som_xy, users)
        self.df_som_xy_wgt['cls'] = pd.Categorical(self.df_som_xy_wgt['cls'], categories=self.cls_cat)
        if drop:
            '''重み0のユーザーを削除'''
            self.df_som_xy_wgt = self.df_som_xy_wgt.iloc[self.df_som_xy_wgt['wgt'].values != 0,:]
        return self.df_som_xy_wgt
    
    def groupby_cls(self, df_som_xy_wgt):
        '''
        [wgt]カラムを[cls]カラムごとに合計する
        output:
        cls0, cls1, cls2, cls3 ・・・
        '''
        df_som_xy_wgt_size = pd.DataFrame(df_som_xy_wgt.groupby('cls')['wgt'].agg('sum'))
        df_som_xy_wgt_size_sorted = df_som_xy_wgt_size.sort_values(['cls'], key=lambda k: k.str.replace('cls', '').astype('int'))
        df_som_xy_wgt_size_sorted['cls'] = pd.Categorical(df_som_xy_wgt_size_sorted.index, categories=self.cls_cat)
        return df_som_xy_wgt_size_sorted

    def get_df_sumwgt_bycls(self, users):
        '''
        [wgt]カラムを[cls]カラムごとに合計する
        output:
        cls0, cls1, cls2, cls3 ・・・
        '''
        self.df_som_xy_wgt = self.get_df_wgt(users)
        self.df_sumwgt_bycls = self.groupby_cls(self.df_som_xy_wgt)
        df_som_xy_uni = self.df_som_xy_uni.copy()
        df_som_xy_uni.drop(['cls'], axis=1, inplace=True)
        self.df_sumwgt_bycls = pd.merge(self.df_sumwgt_bycls, df_som_xy_uni, left_index=True, right_index=True)
        self.total_point = self.df_sumwgt_bycls['wgt'].values.sum()
        print('total_point >', self.total_point)
        return self.df_sumwgt_bycls
    
    def _calc_cnt(self):
        '''
        cnt:
        self.df_som_xyの各クラスのユーザー数
        重み0のユーザーも含む
        cls0, cls1, cls2, cls3 ・・・
        '''
        cnt = self.df_som_xy.groupby('cls').size().sort_index(key=lambda k: k.str.replace('cls', '').astype(int)).values
        return cnt
    
    def _calc_cls_sorted(self):
        '''
        ランドマークのソートインデックスを返す
        平均重みの大きい順
        '''
        df_groupby_cls = self.df_sumwgt_bycls
        cnt = self._calc_cnt()
        df_ratio = df_groupby_cls['wgt'] / cnt # ランドマーク毎の平均重みを計算する
        df_ratio.fillna(0.0, inplace=True)
        df_ratio_sorted = df_ratio.sort_values(ascending=False)
        idx_sorted = df_ratio_sorted.index
        return idx_sorted

    def calc_som_seg(self, num_cls=3):
        '''
        ライト・ミドル・コアなどのセグメントを付与する
        seg-[0]は、ユーザーが存在しなかったセグメント
        数値が大きいほどコア
        '''
        df_som_xy_wgt = self.df_som_xy_wgt
        df_groupby_cls = self.df_sumwgt_bycls.copy()
        
        cls_sorted = self._calc_cls_sorted()
        df_groupby_cls_sorted = df_groupby_cls.loc[cls_sorted,:]
        cumsum = np.cumsum(df_groupby_cls_sorted['wgt'].values / df_groupby_cls_sorted['wgt'].values.sum())
        
        res = 1
        for ii in range(num_cls-1):
            res += cumsum < ((ii+1)/num_cls)
        df_seg = pd.DataFrame(res, index=cls_sorted, columns=['seg'])
        if num_cls not in df_seg['seg'].values:
            print('{} does not exists'.format(num_cls))
            top_idx = cls_sorted.values[0]
            print(top_idx)
            df_seg.loc[top_idx, 'seg'] = num_cls
        df_seg = df_seg.loc[self.cls_cat,:]
        '''重みのないクラスは０'''
        df_seg.iloc[df_groupby_cls['wgt'].values == 0,:] = 0
        
        df_seg['seg'] = pd.Categorical(df_seg['seg'].values, categories=np.arange(num_cls+1), ordered=True)
        self.df_seg = df_seg
        self.join_seg(self.df_seg)
        return self.df_seg
    
    def calc_seg_byuser(self, num_cls=3):
        '''
        ライト・ミドル・コアなどのセグメントを付与する
        seg-[0]は、ユーザーが存在しなかったセグメント
        数値が大きいほどコア
        '''
        wgt_sorted = self.df_som_xy_wgt.sort_values('wgt', ascending=False)['wgt'].values
        idx_sorted = self.df_som_xy_wgt.sort_values('wgt', ascending=False).index.values
        # wgt_sorted, idx_sorted
        cumsum = np.cumsum(wgt_sorted) / wgt_sorted.sum()
        # cumsum

        res = 1
        for ii in range(num_cls-1):
            res += cumsum < ((ii+1)/num_cls)
        res
        df_res = pd.DataFrame(res, columns=['seg'], index=idx_sorted)
        df_res['seg'] = pd.Categorical(df_res['seg'].values, categories=np.arange(num_cls+1), ordered=True)
        df_res.sort_index(inplace=True)
        return df_res
    
    def join_seg(self, df_seg):
        '''seg列をジョインする'''
        try:
            self.df_sumwgt_bycls.drop(['seg'], axis=1, inplace=True)
            self.df_som_xy_wgt.drop(['seg'], axis=1, inplace=True)
        except:
            pass
        self.df_sumwgt_bycls = pd.merge(self.df_sumwgt_bycls, df_seg, left_index=True, right_index=True, validate='1:1')
        self.df_som_xy_wgt = pd.merge(self.df_som_xy_wgt, df_seg, left_on='cls', right_index=True, how='left', validate='m:1')
        '''なぜかカテゴリカルの属性がなくなるので・・・'''
        self.df_som_xy_wgt['cls'] = pd.Categorical(self.df_som_xy_wgt['cls'], categories=self.cls_cat)
        return self.df_som_xy_wgt, self.df_sumwgt_bycls
        

