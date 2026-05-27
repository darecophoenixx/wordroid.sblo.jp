import numpy as np
import scipy
from sklearn.base import BaseEstimator, DensityMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_random_state

class SOMAdaptiveKDE(BaseEstimator, DensityMixin):
    '''
    A Kernel Density Estimator where the bandwidth and kernel centers are adaptively determined by a pre-trained Self-Organizing Map.
    SOM_kernelやXは適切に次元削減されたデータを入力してください
    '''
    
    def __init__(self, som_kernel, tgt_prob=0.005):
        self.som_kernel = som_kernel
        self.tgt_prob = tgt_prob
        # ... 必要なパラメータの初期化 ...
        self.K_cells, self.D = self.som_kernel.shape

    def fit(self, X, y=None):
        self.n_train = X.shape[0]
        dist_matrix = scipy.spatial.distance.cdist(self.som_kernel, X, metric='sqeuclidean')

        # これが各セルの「データ密度を反映したバンド幅」になる！
        average_distances = np.percentile(dist_matrix, self.tgt_prob * 100, axis=1)
        self.gammas = 0.5 / (average_distances / self.D + 1e-14)
        self.log_norm_constants = (self.D / 2.0) * (np.log(self.gammas) - np.log(np.pi))
        return self

    def score_samples(self, X):
        # fitが呼ばれたかどうかをチェック
        check_is_fitted(self, ['gammas', 'log_norm_constants'])
        # ... 対数尤度の算出 ...
        dist_matrix = scipy.spatial.distance.cdist(self.som_kernel, X, metric='sqeuclidean')
        log_densities = self.log_norm_constants - self.gammas * dist_matrix.T
        pointwise_log_likelihood = scipy.special.logsumexp(log_densities, axis=1) - np.log(self.K_cells)
        return pointwise_log_likelihood

    def score(self, X, y=None):
        return np.sum(self.score_samples(X))

    def sample(self, n_samples=1, random_state=None):
        """Generate random samples from the model.

        Currently, this is implemented only for gaussian and tophat kernels.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.

        random_state : int, RandomState instance or None, default=None
            Determines random number generation used to generate
            random samples. Pass an int for reproducible results
            across multiple function calls.
            See :term:`Glossary <random_state>`.

        Returns
        -------
        X : array-like of shape (n_samples, n_features)
            List of samples.
        """
        """Generate random samples from the model."""
        check_is_fitted(self, ['gammas', 'som_kernel'])
        rng = check_random_state(random_state)
        
        # 1. 各セルは等確率 (1/K) で選択される
        # (score_samplesで - np.log(self.K_cells) としているため)
        component_indices = rng.choice(self.K_cells, size=n_samples)
        
        # 2. 各セルに対応する平均(mu)と標準偏差(sigma)を取得
        means = self.som_kernel[component_indices]  # shape: (n_samples, D)
        
        # gamma = 1 / (2 * sigma^2) なので、sigma = sqrt(1 / (2 * gamma))
        # gammasの次元を考慮
        stds = np.sqrt(1 / (2 * self.gammas[component_indices]))  # shape: (n_samples,)
        
        # 3. 各ガウス分布からサンプリング
        # loc: 平均, scale: 標準偏差
        # stds[:, np.newaxis] でブロードキャストして(n_samples, 1)にする
        samples = rng.normal(loc=means, scale=stds[:, np.newaxis])
        
        return samples

    def bic(self, X):
        """
        BIC を計算します。
        k (パラメータ数) をランドマーク数 K と定義します。
        """
        # 1. 対数尤度の合計
        log_likelihood = np.sum(self.score_samples(X))
        
        # 2. パラメータ数 k の定義
        # SOMのランドマーク配置は固定（定数）とみなすため、
        # 自由度は「カーネルの数 K」に依存すると考えます。
        k = self.K_cells 
        
        # 3. データ数 n
        n = self.n_train
        
        # 4. BICの計算
        return -2 * log_likelihood + k * np.log(n)




