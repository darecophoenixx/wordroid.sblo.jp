'''
Copyright (c) 2019 Norio Tamada
Released under the MIT license
https://github.com/darecophoenixx/wordroid.sblo.jp/blob/master/lib/keras_ex/gkernel/LICENSE.md
'''
import random
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

class SimpleSOM(object):
    
    def __init__(self, kshape, init=False, initialization_func=None, rand_stat=0,
                 X=None):
        '''
        Parameters
        ----------
        kshape : (int, int)
        initialization_func: callable or None
            if None, use Sample Init.
            if "linear", use PCA.
        rand_stat:
        '''
        self.kshape = np.array(kshape)
        self.initialization_func = initialization_func
        self.rand_stat = rand_stat
        if init:
            self._initialize(X)
    
    def _initialize(self, X):
        if self.initialization_func is None:
            self.K = self.sample_init(X)
        elif self.initialization_func == 'linear':
            self.K = self.linear_init(X)
        else:
            self.K = self.initialization_func()
        
        qd = [np.array((ii,jj)) for ii in range(self.kshape[0]) for jj in range(self.kshape[1])]
        qd = np.array([np.square(p1 - p2).sum() for p1 in qd for p2 in qd]).reshape(self.kshape.prod(), self.kshape.prod())
        self.qd = qd
    
    def sample_init(self, X):
        random.seed(self.rand_stat)
        ind = random.choices(range(X.shape[0]), k=self.kshape.prod())
        smpl = X[ind]
        return smpl
    
    def linear_init(self, X):
        pca = PCA(n_components=2)
        pca.fit(X)
        if self.kshape[1] < self.kshape[0]:
            x_tick = pca.components_[0] * np.sqrt(pca.explained_variance_)[0]
            y_tick = pca.components_[1] * np.sqrt(pca.explained_variance_)[1]
        else:
            y_tick = pca.components_[0] * np.sqrt(pca.explained_variance_)[0]
            x_tick = pca.components_[1] * np.sqrt(pca.explained_variance_)[1]
            
        l = []
        for xi in np.linspace(-2, 2, self.kshape[0]):
            tmp = np.linspace(-2, 2, self.kshape[1]).reshape(self.kshape[1],1) * y_tick
            tmp += xi * x_tick
            l.append(tmp)
        init_map = np.vstack(l)
        return init_map
    
    def update_once(self, X, K, r=1.5, gamma=0.01, alpha=0.05):
        delta = self.calc_delta(X, K, r, gamma, alpha)
        K += delta
        return K
    
    def update_iter(self, X, K, r=1.5, gamma=0.01, alpha=0.05, it=10):
        for _ in tqdm(range(it)):
            K = self.update_once(X, K, r, gamma, alpha)
        return K
    
#    def calc_delta(self, X, K, r=1.5, gamma=0.01, alpha=0.05):
#        '''
#        if r is provided, gamma is ignored.
#        '''
#        if r:
#            if r <= 0:
#                raise ValueError('r must be greater than zero.')
#            gamma = 1.0 / (2.0 * r**2)
#        else:
#            if gamma <= 0:
#                raise ValueError('gamma must be greater than zero.')
#        resp = []
#        for ii in range(X.shape[0]):
#            resp0 = np.square(K - X[ii]).sum(axis=1).argmin()
#            resp.append(resp0)
#        delta = np.zeros(K.shape)
#        for ii in range(X.shape[0]):
#            x = X[ii]
#            iqd = resp[ii]
#            h = (alpha * np.exp(-gamma * self.qd[iqd])).reshape((K.shape[0],1))
#            delta0 = h * (x - K)
#            delta += delta0
#        return delta
    def calc_delta(self, X, K, r=1.5, gamma=0.01, alpha=0.05):
        '''
        if r is provided, gamma is ignored.
        '''
        if r:
            if r <= 0:
                raise ValueError('r must be greater than zero.')
            gamma = 1.0 / (2.0 * r**2)
        else:
            if gamma <= 0:
                raise ValueError('gamma must be greater than zero.')
        self.gamma = gamma
        
        delta = np.zeros(K.shape)
        for ii in range(X.shape[0]):
            iqd = np.square(K - X[ii]).sum(axis=1).argmin()
            h = (alpha * np.exp(-self.gamma * self.qd[iqd])).reshape((K.shape[0],1))
            delta0 = h * (X[ii] - K)
            delta += delta0
            
        return delta


from sklearn.cluster import KMeans
class sksom(object):
    
    def __init__(self, kshape, init_K=None, rand_stat=0,
                 r=1.5, gamma=0.01, alpha=0.05, it=5):
        '''
        predict:
            use KMeans
        '''
        self.init_K = init_K
        self.r = r
        self.gamma = gamma
        self.alpha = alpha
        self.it = it
        
        '''
        init must be True
        for init qd
        '''
        self.som = SimpleSOM(
            kshape,
            init=True, initialization_func=None,
            rand_stat=rand_stat, X=self.init_K)
        self.landmarks_ = self.init_K
    
    def _kmeans(self):
        kmeans = KMeans(n_clusters=self.som.kshape.prod(), n_init=1, max_iter=1)
        kmeans.labels_ = np.arange(self.som.kshape.prod())
        kmeans.cluster_centers_ = self.landmarks_
        return kmeans
    
    def fit(self, X):
        self.landmarks_ = self.som.update_iter(X, self.landmarks_,
                            self.r, self.gamma, self.alpha, self.it)
        self.som.K = self.landmarks_
        self.kmeans = self._kmeans()
        self.labels_ = self.kmeans.labels_
    
    def predict(self, X):
        return self.kmeans.predict(X)
    
    def predict_proba(self, X):
        p_list = []
        for ii in self.labels_:
            d = np.square(X - self.landmarks_[ii]).sum(axis=1)
            p_list.append(np.exp(-self.som.gamma * d))
        return np.vstack(p_list).T
    





def conv2img(x, kshape, target=range(3)):
    num_feats = x.shape[1]
    x = x.copy().reshape((kshape[0], kshape[1], num_feats))
    codes = x[:,:,target]
    for ii in range(len(target)):
        v_max = codes[:,:,ii].max()
        v_min = codes[:,:,ii].min()
        codes[:,:,ii] = (codes[:,:,ii] - v_min) / (v_max - v_min)
    return codes
