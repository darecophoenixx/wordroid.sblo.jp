import random
from collections import defaultdict, Counter

import numpy as np
import scipy
from tqdm import tqdm
from scipy.stats import multivariate_normal
from sklearn import mixture
from sklearn.metrics import pairwise_distances_argmin
from sklearn import cluster

def stabilize_covariance(cov_k, min_eigval=1e-6):
    # 対称化（数値誤差対策）
    cov_k = 0.5 * (cov_k + cov_k.T)
    if not np.all(np.isfinite(cov_k)):
        cov_k = np.nan_to_num(cov_k, nan=0.0, posinf=1e6, neginf=-1e6)

    # 固有値安定化
    try:
        # 最小固有値による安定化
        eigvals = np.linalg.eigvalsh(cov_k)
        min_val = np.min(eigvals)
        if min_val < min_eigval:
            adjustment = (min_eigval - min_val)
            cov_k += adjustment * np.eye(cov_k.shape[0])
    except np.linalg.LinAlgError:
        # fallback: 正則化
        cov_k += min_eigval * np.eye(cov_k.shape[0])

    return cov_k

def em_with_fixed_means(X, means_init, weights_init=None, covariances_init=None,
                        max_iter=100, tol=1e-3, tol2=1.0e-7, stabilize_covariance_min_eigval=1e-6,
                        verbose=False):
    """
    EMアルゴリズムで means_init を固定して重みと共分散を推定
    :param X: データ行列 (n_samples, n_features)
    :param means_init: 初期化された平均 (n_components, n_features)
    :param max_iter: 最大反復回数
    :param tol: 収束の閾値（ログ尤度の変化）
    :param verbose: True の場合進捗出力
    :return: weights, covariances, responsibilities, log_likelihoods
    """
    n_samples, n_features = X.shape
    n_components = means_init.shape[0]
    
    means = means_init.copy()
    if weights_init is None:
        weights = np.ones(n_components) / n_components  # 初期は均等
    else:
        weights = weights_init.astype(np.float64) / weights_init.astype(np.float64).sum()
    # weights = weights_init.astype(np.float64) / weights_init.astype(np.float64).sum()

    if covariances_init is None:
        covariances = np.array([np.cov(X.T) + 1e-6 * np.eye(n_features) for _ in range(n_components)])
    else:
        covariances = [(cov.astype(np.float64) + cov.astype(np.float64).T) / 2.0 for cov in covariances_init]
        covariances = np.stack(covariances)
    # covariances = [(cov.astype(np.float64) + cov.astype(np.float64).T) / 2.0 for cov in covariances_init]
    # covariances = np.stack(covariances)
    if not np.all(np.isfinite(covariances)):
        raise Exception('not np.all(np.isfinite(cov_k))')

    log_likelihoods = []
    
    for iteration in range(max_iter):
        # E-step
        responsibilities = np.zeros((n_samples, n_components))
        for k in range(n_components):
            rv = multivariate_normal(mean=means[k], cov=covariances[k])
            responsibilities[:, k] = weights[k] * rv.pdf(X)
        total_responsibility = responsibilities.sum(axis=1, keepdims=True)
        total_responsibility[total_responsibility == 0] = 1e-10  # avoid division by zero
        responsibilities /= total_responsibility
        if not np.all(np.isfinite(responsibilities)):
            print('total_responsibility >', total_responsibility[~np.isfinite(total_responsibility)])
            #print(responsibilities)
            print(responsibilities[np.isfinite(responsibilities)])
            raise Exception('not np.all(np.isfinite(responsibilities))')

        # M-step (means固定)
        Nk = responsibilities.sum(axis=0)
        weights = Nk / n_samples
        
        for k in range(n_components):
            diff = X - means[k]
            cov_k = (responsibilities[:, k][:, np.newaxis] * diff).T @ diff
            cov_k /= Nk[k]
            # 安定化対策
            cov_k = stabilize_covariance(cov_k, min_eigval=stabilize_covariance_min_eigval)
            covariances[k] = cov_k

        # Log-likelihood
        log_prob = np.zeros((n_samples, n_components))
        for k in range(n_components):
            rv = multivariate_normal(mean=means[k], cov=covariances[k])
            log_prob[:, k] = np.log(weights[k] + 1e-10) + rv.logpdf(X)
        log_likelihood = np.sum(np.logaddexp.reduce(log_prob, axis=1))
        log_likelihoods.append(log_likelihood)

        if verbose:
            print(f"Iter {iteration + 1}, log-likelihood: {log_likelihood:.4f}")
        
        # 収束判定
        if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            if verbose:
                print("収束しました。")
            break
        if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) / log_likelihoods[-2] < tol2:
            if verbose:
                print("収束しました。")
            break

    return weights, covariances, responsibilities, log_likelihoods

class GreedyGMMSelector:
    '''
    n_stepをスケジュールする
    '''

    def __init__(self, init_means,
                 init_weights, init_covariances,
                 min_clusters=3,
                 n_step3=60, n_step2=60, n_step1=0,
                 stabilize_covariance_min_eigval=1e-6):
        self.stabilize_covariance_min_eigval = stabilize_covariance_min_eigval
        self.init_means = init_means
        self.init_weights = init_weights
        self.init_covariances = init_covariances
        self.min_clusters = min_clusters
        #self.n_step = n_step
        self.n_step3 = n_step3
        self.n_step2 = n_step2
        self.n_step1 = n_step1

        self.selected_means = None
        self.best_bic = np.inf
        self.bic_history = []

    def get_n_step(self, n_components):
        if self.n_step3 <= n_components:
            return 3
        elif self.n_step2 <= n_components:
            return 2
        elif self.n_step1 <= n_components:
            return 1
        else:
            return None
    
    def _compute_all_pdfs(self, X, means, covariances):
        return [multivariate_normal(mean=mean, cov=cov).pdf(X) for mean, cov in zip(means, covariances)]

    def _compute_bic(self, X, n_components, drop_index, weights, pdfs):
        idx_keep = [i for i in range(n_components) if i != drop_index]
        reduced_weights = weights[idx_keep] / weights[idx_keep].sum()

        total_pdf = np.zeros(X.shape[0])
        for i, w in zip(idx_keep, reduced_weights):
            total_pdf += w * pdfs[i]
        # 数値安定化
        total_pdf[total_pdf <= 1e-300] = 1e-300
        log_likelihood = np.log(total_pdf).sum()
        
        n_features = X.shape[1]
        k = len(idx_keep)
        n_params = k * (n_features + n_features * (n_features + 1) / 2) + (k - 1)
        bic = -2 * log_likelihood + n_params * np.log(X.shape[0])
        return bic

    def _compute_gmm_bic(self, X, means, weights, covariances):
        print('Start GMM')
        precisions = [np.linalg.inv(cov) for cov in covariances]

        gmm = mixture.GaussianMixture(
            n_components=len(means),
            weights_init=weights,
            means_init=means,
            precisions_init=precisions,
            covariance_type='full',
            random_state=42
        )
        gmm.fit(X)
        return gmm.bic(X), gmm

    def fit(self, X):
        current_means = self.init_means.copy()
        #weights, covariances = self._estimate_gmm_params(X, current_means)
        # means_init, weights_init=None, covariances_init
        print('processing em_with_fixed_means...')
        weights, covariances, _, _ = em_with_fixed_means(X,
                                                         means_init=current_means,
                                                         weights_init=self.init_weights.copy(),
                                                         covariances_init=self.init_covariances.copy(),
                                                         tol2=1.0e-5,
                                                         stabilize_covariance_min_eigval=self.stabilize_covariance_min_eigval,
                                                         verbose=False)

        nn = 0
        while len(current_means) > self.min_clusters:
            print(f"Current cluster count: {len(current_means)}")
            pdfs = self._compute_all_pdfs(X, current_means, covariances)

            best_bic = np.inf
            for i in range(len(current_means)):
                bic = self._compute_bic(X, len(current_means), i, weights, pdfs)
                if bic < best_bic:
                    best_bic = bic
                    best_indices = [j for j in range(len(current_means)) if j != i]

            best_means = current_means[best_indices]
            best_weights = weights[best_indices] / weights[best_indices].sum()
            best_covariances = covariances[best_indices]

            if best_bic < self.best_bic:
                self.best_bic = best_bic
                self.selected_means = best_means.copy()

            print(f"Best BIC (approx): {best_bic}")
            nn += 1
            if nn >= self.get_n_step(len(current_means)):
                best_covariances = [stabilize_covariance(cov, min_eigval=self.stabilize_covariance_min_eigval) for cov in best_covariances]
                try:
                    full_bic, gmm = self._compute_gmm_bic(X, best_means, best_weights, best_covariances)
                except ValueError as e:
                    print(e)
                    best_covariances = [stabilize_covariance(cov, min_eigval=self.stabilize_covariance_min_eigval) for cov in best_covariances]
                    full_bic, gmm = self._compute_gmm_bic(X, best_means, best_weights, best_covariances)
                print(f"Full GMM BIC: {full_bic}")
                current_means = gmm.means_
                weights = gmm.weights_
                covariances = gmm.covariances_
                self.bic_history.append((len(current_means), current_means, best_bic, full_bic, gmm))
                nn = 0
            else:
                current_means = best_means.copy()
                weights = best_weights.copy()
                covariances = best_covariances.copy()
                self.bic_history.append((len(current_means), current_means, best_bic, None, None))

        return self

    def predict(self, X):
        return pairwise_distances_argmin(X, self.selected_means)

def fn(res):
    bics = [ee[3] if ee[3] is not None else np.inf for ee in res['bic_history']]
    return min(bics)

def run_greedy_gmm_trials(X, 
                          n_trials=20, 
                          init_clusters=30,
                          min_clusters=1,
                          random_state_seed=42,
                          nn=5, # 作成したい組み合わせの数
                          n_step3=60, n_step2=60, n_step1=0,
                          stabilize_covariance_min_eigval=1.0e-6):
    '''
    Iterative Refinement
    '''
    rng = np.random.RandomState(random_state_seed)
    kmeans = cluster.KMeans(n_clusters=init_clusters, random_state=rng, n_init=1)
    init_means = kmeans.fit(X).cluster_centers_
    init_weights = np.ones(kmeans.n_clusters, dtype=np.float64) / kmeans.n_clusters
    #init_covariances = np.array([np.cov(X.T) * np.eye(X.shape[1]) for _ in range(kmeans.n_clusters)])
    init_covariances = np.array([np.cov(X.T) + 1e-6 * np.eye(X.shape[1]) for _ in range(kmeans.n_clusters)])
    
    results = []
    best_means = None
    best_weights = None
    best_covariances = None
    for trial in tqdm(range(n_trials), desc="GreedyGMM Trials"):
        
        if best_weights is None:
            pass
        else:
            best_weights_avg = best_weights.mean()
            init_weights = np.array([e for e in best_weights]+[best_weights_avg for _ in range(nn)], dtype=np.float64)
            init_weights = init_weights / init_weights.sum()

        if best_covariances is None:
            pass
        else:
            #best_covariances_avg = best_covariances.mean(axis=0)
            #covs0 = np.stack([best_covariances_avg for _ in range(nn)])
            covs0 = np.stack([np.cov(X.T) * np.eye(X.shape[1]) for _ in range(nn)])
            init_covariances = np.vstack([best_covariances, covs0])
        
        # ランダム初期化された初期クラスタ中心（KMeans）
        if best_means is None:
            pass
        else:
            print('追加のクラスタ中心を選択')
            my_list = np.arange(best_means.shape[0]).tolist()
            combinations = []
            for _ in range(nn):
                # my_listから重複なく2つの要素をランダムに選ぶ
                selected_pair = tuple(random.sample(my_list, 2))
                combinations.append(selected_pair)
            
            new_centers = []
            for icomb in combinations:
                new_center = (best_means[icomb[0]] + best_means[icomb[1]]) / 2
                #new_center = best_means[icomb[0]] + (best_means[icomb[1]] - best_means[icomb[0]]) * 0.3
                new_centers.append(new_center)
            new_centers = np.vstack(new_centers)
            init_means = np.vstack([best_means, new_centers])

        print('GreedyGMMSelector の実行')
        selector = GreedyGMMSelector(init_means=init_means,
                                     init_weights=init_weights,
                                     init_covariances=init_covariances,
                                     min_clusters=min_clusters,
                                     n_step3=n_step3, n_step2=n_step2, n_step1=n_step1,
                                     stabilize_covariance_min_eigval=stabilize_covariance_min_eigval)
        selector.fit(X)

        # 結果を保存
        result = {
            'trial': trial,
            'bic': selector.best_bic,
            'n_clusters': selector.selected_means.shape[0],
            'means': selector.selected_means,
            'bic_history': selector.bic_history,
            'selector': selector,
        }
        results.append(result)

        # 最良の結果を選択
        #best_result = min(results, key=lambda r: r['bic'])
        best_result = min(results, key=fn)
        best_means = min(best_result['bic_history'], key=lambda x: x[3] if x[3] is not None else np.inf)[1]
        best_gmm = min(best_result['bic_history'], key=lambda x: x[3] if x[3] is not None else np.inf)[4]
        best_weights = best_gmm.weights_
        best_covariances = best_gmm.covariances_

    return best_result, results
