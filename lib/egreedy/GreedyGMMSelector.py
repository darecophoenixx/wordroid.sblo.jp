class GreedyGMMSelector3:
    '''
    n_stepをスケジュールする
    '''

    def __init__(self, init_means, min_clusters=3,
                 n_step3=60, n_step2=60, n_step1=0):
        self.init_means = init_means
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
        weights, covariances, _, _ = em_with_fixed_means(X, current_means, verbose=False)

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
                best_covariances = [stabilize_covariance(cov) for cov in best_covariances]
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
                          n_step3=60, n_step2=60, n_step1=0):
    '''
    Iterative Refinement
    '''
    rng = np.random.RandomState(random_state_seed)
    kmeans = cluster.KMeans(n_clusters=init_clusters, random_state=rng, n_init=1)
    init_means = kmeans.fit(X).cluster_centers_
    
    results = []
    best_means = None
    for trial in tqdm(range(n_trials), desc="GreedyGMM Trials"):
        
        # ランダム初期化された初期クラスタ中心（KMeans）
        if best_means is None:
            pass
        else:
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

        # GreedyGMMSelector の実行
        selector = GreedyGMMSelector3(init_means=init_means, min_clusters=min_clusters,
                                      n_step3=n_step3, n_step2=n_step2, n_step1=n_step1)
        selector.fit(X)

        # 結果を保存
        result = {
            'trial': trial,
            'bic': selector.best_bic,
            'n_clusters': selector.selected_means.shape[0],
            'means': selector.selected_means,
            'bic_history': selector.bic_history
        }
        results.append(result)

        # 最良の結果を選択
        #best_result = min(results, key=lambda r: r['bic'])
        best_result = min(results, key=fn)
        best_means = min(best_result['bic_history'], key=lambda x: x[3] if x[3] is not None else np.inf)[1]

    return best_result, results

