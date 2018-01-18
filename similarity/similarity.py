
import numpy as np
import scipy
import gensim



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
        
        idfs = np.empty((max(tfidf.idfs.keys())+1,1), dtype=self.DTYPE)
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
        '''内積'''
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
        wd = self.index[nonzero_idx2,:].copy() # 必要な文書のみを抽出
        
        '''内積'''
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
        wd = self.index[nonzero_idx2,:].copy() # 必要な文書のみを抽出
        
        '''内積'''
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
        wd = self.index[nonzero_idx2,].astype(self.DTYPE) # 必要な文書のみを抽出
        wd = self.calc_tfn_mx(wd)
        
        '''内積'''
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
        wd = self.index[nonzero_idx2,].astype(self.DTYPE) # 必要な文書のみを抽出
        wd = self.calc_wd(wd, method=method)
        
        '''内積'''
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
