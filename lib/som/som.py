'''
Copyright (c) 2019 Norio Tamada
Released under the MIT license
https://github.com/darecophoenixx/wordroid.sblo.jp/blob/master/lib/keras_ex/gkernel/LICENSE.md
'''
import sys
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm

class SimpleSOM(object):
    
    def __init__(self, kshape, init=False, initialization_func=None,
                 rand_stat=0, X=None, dtype=np.float64):
        '''
        Parameters
        ----------
        kshape : (int, int)
        initialization_func: callable or None
            if None, use Sample Init.
            if "linear", use PCA.
        rand_stat:
        '''
        self.dtype = dtype
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
        self._calc_qd(X)
    
    def _calc_qd(self, X):
        qd = [np.array((ii,jj)) for ii in range(self.kshape[0]) for jj in range(self.kshape[1])]
        qd = np.array([np.square(p1 - p2).sum() for p1 in qd for p2 in qd]).reshape(self.kshape.prod(), self.kshape.prod())
        self.qd = qd
    
    def sample_init(self, X):
        random.seed(self.rand_stat)
        ind = random.choices(range(X.shape[0]), k=self.kshape.prod())
        smpl = X[ind]
        return smpl
    
    def linear_init(self, X):
        sc = StandardScaler()
        sc.fit(X)
        x_sc = sc.transform(X)
        
        pca = PCA(n_components=2, random_state=self.rand_stat)
        pca.fit(x_sc)
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
        return init_map * np.sqrt(sc.var_) + sc.mean_
    
    def update_once(self, X, K, r=1.5, gamma=None, alpha=1.0):
        delta = self.calc_delta(X, K, r=r, gamma=gamma, alpha=alpha)
        K += delta
        return K
    
    def update_iter(self, X, K, r=1.5, gamma=None, alpha=1.0, it=10):
        X0 = X.astype(self.dtype)
        K0 = K.astype(self.dtype)
        #for _ in tqdm(range(it)):
        with tqdm(total=it, file=sys.stdout) as pbar:
            for _ in range(it):
                K0 = self.update_once(X0, K0, r=r, gamma=gamma, alpha=alpha)
                pbar.update(1)
        return K0
    
    def calc_delta(self, X, K, r=1.5, gamma=None, alpha=1.0):
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
        
        return self._calc_delta(X.astype(self.dtype),
                                (X**2).sum(axis=1).astype(self.dtype),
                                K,
                                delta=np.zeros(K.shape, dtype=self.dtype),
                                h=np.zeros((K.shape[0],1), dtype=self.dtype),
                                nn=X.shape[0],
                                idx=np.arange(X.shape[0]),
                                mean_dist=np.zeros((1,), dtype=self.dtype),
                                gamma=gamma, alpha=alpha)
    
    def _update_once(self, X, X2s1, K,
                     delta, h , nn, idx,
                     mean_dist,
                     gamma=None, alpha=1.0):
        delta = self._calc_delta(X, X2s1, K,
                                 delta, h , nn, idx,
                                 mean_dist,
                                 gamma, alpha)
        K += delta
        return K
    
    def _calc_delta(self, X, X2s1, K,
                    delta, h, nn, idx,
                    mean_dist,
                    gamma=None, alpha=1.0,
                    batch_size=100):
        self.gamma = gamma
        
        K2s1 = (K**2).sum(axis=1)
        delta[:,:] = 0
        distance2ClosestLM_list = []
        for ii in range(0, nn, batch_size):
            idx_p = idx[ii:((ii+batch_size) if (ii+batch_size)<nn else nn)]
            KdotX = K.dot(X[idx_p].T)
            for jj in range(len(idx_p)):
                res = K2s1 - 2*KdotX[:,jj] + X2s1[ii+jj]
                iqd = res.argmin()
                distance2ClosestLM = res[iqd]
                distance2ClosestLM_list.append(distance2ClosestLM)
                h[:,0] = (alpha * np.exp(-gamma * self.qd[iqd]))
                delta0 = h * (X[ii+jj] - K)
                delta += delta0
        mean_dist[0] = np.stack(distance2ClosestLM_list).mean()
        return delta / nn

class SphereSOM(SimpleSOM):
    
    def _calc_qd(self, X):
        longitude = [2*np.pi*ii/self.kshape[1] for ii in range(self.kshape[1])]
        latitude = np.linspace(-np.pi/2, np.pi/2, self.kshape[0]+1)[1:] - np.pi/self.kshape[0]/2
        pos0 = [np.array((np.cos(lo), np.sin(lo), np.sin(la))) for la in latitude for lo in longitude]
        pos = [np.array((ee[0]*np.sqrt(1-ee[2]**2), ee[1]*np.sqrt(1-ee[2]**2), ee[2])) for ee in pos0]
        qd = np.array([np.arccos(np.dot(p0, p1).clip(-1,1)) for p0 in pos for p1 in pos])
        qd = qd.reshape((np.array(self.kshape).prod(), np.array(self.kshape).prod()))
        #self.qd = qd * 15
        #self.qd = qd * 10
        self.qd = qd * (1 / (2*np.pi/self.kshape[1]))



from sklearn.cluster import KMeans
class sksom(object):
    
    def __init__(self, kshape, form=None,
                 init_K=None, rand_stat=0,
                 r=None, gamma=None, alpha=1.0, it=5,
                 early_stopping=(5, 1.0e-7),
                 verbose=0, dtype=np.float64):
        '''
        predict:
            use KMeans
        '''
        if early_stopping:
            if len(early_stopping) != 2:
                raise Exception('lenght of early_stopping must be 2...{}'.format(early_stopping))
        self.early_stopping = early_stopping
        self.kshape = kshape
        self.init_K = init_K
        self.r = r
        self.dtype = dtype
        self.gamma = gamma
        self.alpha = alpha
        self.it = it
        if it < 2:
            raise Exception('"it[{}]" must be greater than one...'.format(it))
        self.verbose = verbose
        
        '''
        init must be True
        for init qd
        '''
        if form == 'sphere':
            ssom_constructor = SphereSOM
        else:
            ssom_constructor = SimpleSOM
        self.som = ssom_constructor(
            kshape,
            init=True, initialization_func=None,
            rand_stat=rand_stat, X=self.init_K,
            dtype=self.dtype)
        self.landmarks_ = self.init_K.copy().astype(self.dtype)
        self.som.K = self.init_K.copy()
        self.kmeans = self._kmeans()
        self.labels_ = self.kmeans.labels_
    
    def _kmeans(self):
        kmeans = KMeans(n_clusters=self.som.kshape.prod(), n_init=1, max_iter=1)
        kmeans.labels_ = np.arange(self.som.kshape.prod())
        kmeans.cluster_centers_ = self.landmarks_
        return kmeans
    
    def fit(self, X, y=None):
        X0 = X.astype(self.dtype)
        X2s1 = (X**2).sum(axis=1).astype(self.dtype)
        
        delta = np.zeros(self.landmarks_.shape, dtype=self.dtype)
        h = np.zeros((self.landmarks_.shape[0],1), dtype=self.dtype)
        nn = X.shape[0]
        idx = np.arange(nn)
        meanDist = np.zeros((self.it,), dtype=self.dtype)
        meanDist0 = np.zeros((1,), dtype=self.dtype)

        if self.r:
            if isinstance(self.r, (float, int)):
                if self.r <= 0:
                    raise ValueError('r must be greater than zero.')
            else:
                if self.r[0] <= 0 or self.r[1] <= 0:
                    raise ValueError('r must be greater than zero.')
        if self.early_stopping:
            early_stopping = self.early_stopping[0]
            early_stopping_cnt = 0
            tol = self.early_stopping[1]
            flag_stopping = False
        
        if self.r is None:
            r0 = min(self.kshape)
            rfin = 1
        elif isinstance(self.r, (float, int)):
            r0 = self.r
            rfin = self.r
        else:
            '''expect length of r is 2'''
            r0 = self.r[0]
            rfin = self.r[1]
        
        with tqdm(total=self.it, file=sys.stdout,
                  disable=False if 0<self.verbose else True) as pbar:
            for ii in range(self.it):
#                if self.r:
#                    r = self.r
#                else:
#                    #r = min(self.kshape) - (min(self.kshape)-1)/(self.it-1)*ii
#                    r = min(self.kshape) - (min(self.kshape)-1)/np.sqrt(self.it-1)*np.sqrt(ii)
                
                r = r0 - (r0-rfin)/np.sqrt(self.it-1)*np.sqrt(ii)
                gamma = 1.0 / (2.0 * r**2)
                K = self.som._update_once(X0, X2s1,
                                         self.landmarks_,
                                         delta, h, nn, idx,
                                         mean_dist=meanDist0,
                                         gamma=gamma, alpha=self.alpha)
                meanDist[ii] = meanDist0[0]
                self.landmarks_ = self.som.K = K
                if 1 < self.verbose:
                    pbar.set_description('r: %f / gamma: %f / mean distance: %f' % (r, self.som.gamma, meanDist0[0]))
                pbar.update(1)
                if self.early_stopping:
                    if meanDist[ii-1]-meanDist[ii] < tol:
                        early_stopping_cnt += 1
                    else:
                        early_stopping_cnt = 0
                    if early_stopping <= early_stopping_cnt:
                        flag_stopping = True
                        meanDist = meanDist[:(ii+1)]
                        break
        
        if self.early_stopping:
            if flag_stopping:
                if self.verbose: print('early stopping...')
        if self.verbose:
            print('r:', r, 'gamma:', self.som.gamma, 'mean distance:', meanDist0[0])
        self.meanDist = meanDist
    
    def predict(self, X):
        return self.kmeans.predict(X)
    
    def predict_proba(self, X):
        p_list = []
        for ii in self.labels_:
            d = np.square(X - self.landmarks_[ii]).sum(axis=1)
            p_list.append(np.exp(-self.som.gamma * d))
        return np.vstack(p_list).T
    

from sklearn.base import ClassifierMixin
from sklearn.neighbors import NearestNeighbors
class SomClassifier(ClassifierMixin):
    
    def __init__(self, kshape, rand_stat=0,
                 r=1.5, gamma=None, alpha=1.0, it=(5,50),
                 verbose=0, early_stopping=(5, 1.0e-6),
                 knn=None, sksom=None,
                 dtype=np.float64):
        self.kshape = kshape
        self.rand_stat = rand_stat
        self.r = r
        self.dtype = dtype
        self.gamma = gamma
        self.verbose = verbose
        self.early_stopping = early_stopping
        
        if alpha <= 0:
            raise Exception('"alpha[{}]" must be greater than zero...'.format(alpha))
        self.alpha = alpha
        
        if len(it) < 1:
            raise Exception('Length of "it[{}]" must be greater than or equal 2...'.format(it))
        if it[0] < 0:
            raise Exception('"it[0]({})" must be greater than or equal zero...'.format(it[0]))
        if it[1] < 2:
            raise Exception('"it[1]({})" must be greater than one...'.format(it[1]))
        self.it = it
        
        if knn is None:
            self.knn = NearestNeighbors()
        else:
            self.knn = knn
        
        '''
        sksom provided, no fit
        '''
        if sksom:
            self.sksom = sksom
            self.no_fit = True
        else:
            self.sksom = None
            self.no_fit = False
    
    def _fit(self, X):
        '''phase 1'''
        if self.rand_stat:
            r = self.r
        else:
            r = None
        self.sksom = sksom(kshape=self.kshape, init_K=self.init_lm,
                           rand_stat=self.rand_stat,
                           r=r, gamma=None, alpha=self.alpha,
                           it=self.it[0] if 1<self.it[0] else 2,
                           early_stopping=False,
                           verbose=self.verbose, dtype=self.dtype)
        self.sksom.fit(X)
        '''phase 2'''
        self.sksom.r = self.r
        self.sksom.it = self.it[1]
        self.sksom.early_stopping = self.early_stopping
        self.sksom.fit(X)
    
    def fit(self, X, y):
        '''
        It is expected that X was standardized.
        encode y
        '''
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))
            self.single = True
        y_enc = np.zeros((y.shape), dtype=self.dtype)
        for ii in range(y.shape[1]):
            y_enc[:,ii] = np.array([1 if ee!=0 else -1 for ee in y[:,ii]], dtype=self.dtype)
        
        Xy = np.concatenate([y_enc, X], axis=1)
        if self.rand_stat:
            som4initLM = SimpleSOM(
                kshape=self.kshape, init=True, initialization_func=None,
                rand_stat=self.rand_stat,
                X=Xy, dtype=np.float64)
        else:
            som4initLM = SimpleSOM(
                kshape=self.kshape, init=True, initialization_func='linear',
                rand_stat=self.rand_stat,
                X=Xy, dtype=np.float64)
        self.init_lm = som4initLM.K.copy()
        del som4initLM
        if not self.no_fit:
            self._fit(Xy)
        else:
            if self.verbose:
                print('no fitting...')
    
    def predict(self, X, exclude=True):
        y_prob = self.predict_proba(X)
        if hasattr(self, 'single'):
            return (0.5<y_prob).astype(int)
        if exclude:
            ret = y_prob.argmax(axis=1)
            return ret
        return (0.5<y_prob).astype(int)
    
    def predict_proba(self, X):
        y_prob0_lm = self.sksom.landmarks_[:,:(self.sksom.landmarks_.shape[1]-X.shape[1])]
        y_prob_lm = 1 / (1+np.exp(-y_prob0_lm)) # sigmoid
        lmX = self.sksom.landmarks_[:,-X.shape[1]:]
        self.knn.fit(lmX)
        idx = self.knn.kneighbors(X, return_distance=False)
        y_prob_list = []
        for ii in range(idx.shape[1]):
            y_prob0 = y_prob_lm[idx[:,ii]]
            y_prob_list.append(y_prob0)
        y_prob = np.stack(y_prob_list, axis=2).mean(axis=2)
        if hasattr(self, 'single'):
            y_prob = y_prob.flatten()
        return y_prob
        
    




def conv2img(x, kshape, target=range(3)):
    num_feats = x.shape[1]
    x = x.copy().reshape((kshape[0], kshape[1], num_feats))
    codes = x[:,:,target]
    for ii in range(len(target)):
        v_max = codes[:,:,ii].max()
        v_min = codes[:,:,ii].min()
        codes[:,:,ii] = (codes[:,:,ii] - v_min) / (v_max - v_min)
    return codes

