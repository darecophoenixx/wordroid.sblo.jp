from scipy.stats import multivariate_normal

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

def em_with_fixed_means(X, means_init, max_iter=100, tol=1e-4, verbose=False):
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
    weights = np.ones(n_components) / n_components  # 初期は均等
    covariances = np.array([np.cov(X.T) + 1e-6 * np.eye(n_features) for _ in range(n_components)])
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
            cov_k = stabilize_covariance(cov_k, min_eigval=1e-6)
            # 動的正則化（スケールに応じて）
            if 100 <= n_features:
                eps = 1e-1
            else:
                eps = 1e-2 * np.trace(cov_k) / n_features
            cov_k += eps * np.eye(n_features)
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

    return weights, covariances, responsibilities, log_likelihoods
