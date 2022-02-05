'''
Copyright (c) 2018 Norio Tamada
Released under the MIT license
https://github.com/darecophoenixx/wordroid.sblo.jp/blob/master/lib/feature_eng/LICENSE.md
'''

import sys
import itertools
import random
from collections import Mapping
import logging

import numpy as np
import scipy
import gensim
import pandas as pd
from tqdm import tqdm

from sklearn import mixture
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, classification_report, confusion_matrix, log_loss, pairwise
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import seaborn as sns
           
from feature_eng.lowcols2 import (
    WD2vec
)
from feature_eng import lowcols3





N_INIT = 30
G_RANGE = np.arange(3,16)
cov_type_list = ['full', 'tied', 'diag', 'spherical']


class M01F(object):
    """Calculate feature vector using lowcols2.
    
    Parameters
    ----------
    .
        X_df (pandas.DataFrame of shape (n_samples, n_features))
            training instances
    
    Attributes
    ----------
    
    wgt_row : ndarray of shape (n_samples, n_features)
        feature vector of row side
    
    wgt_col : ndarray of shape (n_features, n_features)
        feature vector of col side
    
    gamma : float
        RBF parameter gamma
    """
    
    def __init__(self, X_df):
        self.X_df = X_df
        self.wd2v = WD2vec(X_df)
    
    def make_model(self, num_features=None, seed=10001,
                   num_neg=2,
                   gamma=0.0, embeddings_val=0.1, loss_wgt_neg=0.05,
                   rscore=None, cscore=None):
        """create WordAndDoc2vec model
        
        Parameters
        ----------
        .
            num_features (int)
                number of features, if None, num_features = self.n_sig
            
            seed (int, or None)
                RandomState for embeddings_initializer
            
            embeddings_val (float)
                embeddings_initializer=initializers.RandomUniform(
                minval=-embeddings_val, maxval=embeddings_val, seed=seed)
            
            num_neg (int, default=1)
                number of negative samping (row direction)
            
            gamma (float)
                tf.keras Embedding layer regularizers.l2(gamma)
            
            loss_wgt_neg (float)
                loss_weights={'y': 1.0, 'neg_y': loss_wgt_neg, 'cor_y': 1.0}
            
            rscore
                user_embedding weights
            
            cscore
                prod_embedding weights
        
        Returns
        -------
        model dict
            models created
        """
        if num_features is None:
            num_features = self.X_df.shape[1]
        self.num_features = num_features
        self.models = self.wd2v.make_model(num_features=num_features, seed=seed,
                                           num_neg=num_neg,
                                           gamma=gamma, embeddings_val=embeddings_val, loss_wgt_neg=loss_wgt_neg,
                                           rscore=rscore, cscore=cscore)
        return self.models
    
    def train(self, epochs=512, batch_size=128, verbose=0,
              use_multiprocessing=False, workers=1, shuffle=True,
              callbacks=None, callbacks_add=None, lr0=0.02, flag_early_stopping=True,
              base=8):
        """compute feature vector
        
        Parameters
        ----------
            epochs : int, default=500
                number of epochs to train the model
            
            batch_size : int, default=1024
                number of samples per gradient update
            
            verbose : int, default=0
                verbosity mode
        
        Returns
        -------
        self
            Fitted estimator
        """
        self.hst, self.hst2 = self.wd2v.train(epochs=epochs, batch_size=batch_size, verbose=verbose, lr0=lr0,
                                    shuffle=shuffle, flag_early_stopping=flag_early_stopping, base=base,
                                    callbacks=callbacks, callbacks_add=callbacks_add,
                                    use_multiprocessing=use_multiprocessing, workers=workers)
        
        self.df_wgt_col = pd.DataFrame(self.wgt_col, index=self.X_df.columns, columns=['fet'+str(ee) for ee in range(self.num_features)])
        self.df_wgt_row = pd.DataFrame(self.wgt_row, index=self.X_df.index, columns=['fet'+str(ee) for ee in range(self.num_features)])
        self.calc_cor()
        self.eig()
        return self
    
    def plot_hst(self, figsize=(20,5)):
        """plot loss history of 1st phase
        """
        hst_history = self.hst.history
        fig, ax = plt.subplots(1, 3, figsize=figsize)
        ax[0].set_title('loss')
        ax[0].plot(list(range(len(hst_history["loss"]))), hst_history["loss"], label="Train loss")
        ax[1].set_title('acc')
        ax[1].plot(list(range(len(hst_history["loss"]))), hst_history["binary_accuracy"], label="accuracy")
        ax[2].set_title('learning rate')
        ax[2].plot(list(range(len(hst_history["loss"]))), hst_history["lr"], label="learning rate")
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        return ax
    
    def plot_hst2(self, figsize=(20,5)):
        """plot loss history of 2nd phase
        """
        hst_history = self.hst2.history
        fig, ax = plt.subplots(1, 3, figsize=figsize)
        ax[0].set_title('loss')
        ax[0].plot(list(range(len(hst_history["loss"]))), hst_history["loss"], label="Train loss")
        ax[0].plot(list(range(len(hst_history["loss"]))), hst_history["y_loss"], label="Train y loss")
        ax[0].plot(list(range(len(hst_history["loss"]))), hst_history["y_neg_user_loss"], label="Train y neg_user loss")
        ax[1].set_title('acc')
        ax[1].plot(list(range(len(hst_history["loss"]))), hst_history["y_binary_accuracy"], label="y accuracy")
        ax[1].plot(list(range(len(hst_history["loss"]))), hst_history["y_neg_user_binary_accuracy"], label="y neg_user accuracy")
        ax[2].set_title('learning rate')
        ax[2].plot(list(range(len(hst_history["loss"]))), hst_history["lr"], label="learning rate")
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        return ax
    
    '''距離類似度算出の関数を定義'''
    def _calc(self, v, df, gamma=1, sort=True):
        res = []
        d2 = ((df.values - v)**2).sum(axis=1)
        prob = np.exp(-gamma * d2)
        df_out = pd.DataFrame(df.index.values, index=df.index.values, columns=['title'])
        df_out['prob'] = prob
        if sort:
            df_out.sort_values('prob', inplace=True, ascending=False)
        return df_out
    
    def calc_cor(self):
        '''Evaluate row and column features with column features
        '''
        res_list = []
        for icol in self.df_wgt_col.index.values:
            res = self._calc(self.df_wgt_col.loc[icol,:].values, self.df_wgt_col, gamma=self.gamma, sort=False)
            res2 = res
            res_list.append(res2['prob'].values)
        self.df_cor = pd.DataFrame(res_list, index=self.df_wgt_col.index, columns=self.df_wgt_col.index)
        
        res_list = []
        for icol in self.df_wgt_col.index.values:
            res = self._calc(self.df_wgt_col.loc[icol,:].values, self.df_wgt_row, gamma=self.gamma, sort=False)
            res2 = res
            res_list.append(res2['prob'].values)
        fet_row_cnvt = pd.DataFrame(res_list).T
        fet_row_cnvt.index, fet_row_cnvt.columns = self.df_wgt_row.index, self.df_wgt_col.index
        self.fet_row_cnvt = fet_row_cnvt
    
    def eig(self):
        '''quasi PCA
        '''
        sdev2, loadings = np.linalg.eigh(self.df_cor.values)
        idx = np.argsort(sdev2)[::-1]
        sdev2, self.loadings = sdev2[idx], loadings[:,idx]
        self.sdev = np.sqrt(sdev2)
        return self.sdev, self.loadings
    
    def plot_eig(self, figsize=(10, 10), marker='-o'):
        """plot sdev
        """
        plt.figure(figsize=figsize)
        plt.plot(np.arange(len(self.sdev))+1, self.sdev, marker)
        plt.grid()
    
    def set_npca(self, n_pca=3):
        """set n_pca
        """
        self.n_pca = n_pca
        self.loadings_selected = self.loadings[:,:n_pca]
        self.fet_col_aggr = self.df_cor.values.dot(self.loadings_selected)
        self.fet_row_aggr = self.fet_row_cnvt.values.dot(self.loadings_selected)

    def mclust(self, df, n_init=N_INIT, g_range=G_RANGE, cov_type_list=cov_type_list,
               gm=mixture.GaussianMixture(init_params='kmeans')):
        """Simulate with Gaussian Mixture
        """
        self.g_range = g_range
        mclust_res = mclust(df, n_init=n_init, g_range=g_range, cov_type_list=cov_type_list, gm=gm)
        return mclust_res
    
    def mclust_col(self, n_pca=3, n_init=N_INIT, g_range=G_RANGE, cov_type_list=cov_type_list,
               gm=mixture.GaussianMixture(init_params='kmeans')):
        """Simulate with Gaussian Mixture using column features
        """
        self.set_npca(n_pca)
        self.mclust_col_res = self.mclust(pd.DataFrame(self.fet_col_aggr), n_init=n_init, g_range=g_range,
                                          cov_type_list=cov_type_list, gm=gm)
        return self.mclust_col_res
    
    def mclust_row(self, n_pca=None, n_init=N_INIT, g_range=G_RANGE, cov_type_list=cov_type_list,
               gm=mixture.GaussianMixture(init_params='kmeans')):
        """Simulate with Gaussian Mixture using row features
        
        self.n_pca must be set
        """
        if n_pca is None:
            assert self.n_pca
            n_pca = self.n_pca
        self.set_npca(n_pca)
        self.mclust_row_res = self.mclust(pd.DataFrame(self.fet_row_aggr), n_init=n_init, g_range=g_range,
                                          cov_type_list=cov_type_list, gm=gm)
        return self.mclust_row_res
    
    def plot_mclust(self, res, figsize=(15, 15), lw=2):
        """plot results of mclust
        """
        return plot_mclust(res, g_range=self.g_range, figsize=figsize, lw=lw)
    
    def plot_mclust_col(self, figsize=(15, 15), lw=2):
        """plot results of mclust
        """
        self.plot_mclust(self.mclust_col_res, figsize, lw)
    
    def plot_mclust_row(self, figsize=(15, 15), lw=2):
        """plot results of mclust
        """
        self.plot_mclust(self.mclust_row_res, figsize, lw)
    
    def plot_pca_fa_cv(self, df=None, n_range=np.arange(2, 16), n_init=10,
                       cv=KFold(n_splits=5, shuffle=True),
                       figsize=(10,10)):
        if df is not None:
            x_mat = df.values
            n_obs = df.shape[0]
        else:
            x_mat = self.df_cor.values
            n_obs = self.X_df.shape[0]
        self.cv_pca_scores, self.cv_fa_scores = plot_pca_fa_cv(x_mat, n_obs,
                                                               n_range=n_range, n_init=n_init, cv=cv,
                                                               figsize=figsize)
    
    def _create_mat_selected_cor(self, selected_cor, df):
        r_calced = create_mat_from_cor(selected_cor, df.shape[0])
        return r_calced
    
    def find_ncomponents(self, df=None, type='pca', n_iter=10,
                        figsize=(10,10)):
        """investigate the number of significant components 
        """
        if df is None:
            w, res = find_ncomponents_pca(self.df_cor.values, n_obs=self.X_df.shape[0], n_iter=n_iter, figsize=figsize)
        else:
            w, res = find_ncomponents_pca(df.values, n_iter=n_iter, figsize=figsize)
        return w, res
    
    def get_wgt_byrow(self):
        wgt_row = self.wd2v.get_wgt_byrow()
        return wgt_row
    wgt_row = property(get_wgt_byrow)
    
    def get_wgt_bycol(self):
        wgt_col = self.wd2v.get_wgt_bycol()
        return wgt_col
    wgt_col = property(get_wgt_bycol)
    
    def get_gamma(self):
        gamma = self.wd2v.get_gamma()
        return gamma
    gamma = property(get_gamma)



class M01Fv2(M01F):
    """Calculate feature vector using lowcols3.
    
    The correlation coefficient (distance similarity) of 
    the feature vector on the column side matches the 
    correlation coefficient of original data.
    
    Parameters
    ----------
    .
        X_df (pandas.DataFrame of shape (n_samples, n_features))
            training instances
        
        n_sig (int)
            number of significant components
        
        n_iter (int)
            using find_ncomponents_pca
        
        figsize (sequence of length 2)
            using find_ncomponents_pca
    
    Attributes
    ----------
    
    wgt_row : ndarray of shape (n_samples, n_features)
        feature vector of row side
    
    wgt_col : ndarray of shape (n_features, n_features)
        feature vector of col side
    
    gamma : float
        RBF parameter gamma
    """
    
    def __init__(self, X_df, n_sig=None, n_iter=10, figsize=(10,10)):
        if n_sig is None:
            cor_mat = cosine_similarity(X_df.values.T)
            w, res = find_ncomponents_pca(cor_mat, n_obs=X_df.shape[0], n_iter=n_iter, figsize=figsize)
            self.n_sig = ((res[:,1,:].mean(axis=0) + res[:,1,:].std(axis=0)) < w).sum()
            print('suggests number of significant component ->', self.n_sig)
        else:
            self.n_sig = n_sig
        self.X_df = X_df
        self.cor_mat = cosine_similarity(X_df.values.T)
        self.wd2v = lowcols3.WD2vec(X_df, self.cor_mat)
    
    def make_model(self, num_features=None, seed=10001,
                   num_neg=1,
                   gamma=0.0, embeddings_val=0.1, loss_wgt_neg=0.001,
                   rscore=None, cscore=None):
        """create WordAndDoc2vec model
        
        Parameters
        ----------
        .
            num_features (int)
                number of features, if None, num_features = self.n_sig
            
            seed (int, or None)
                RandomState for embeddings_initializer
            
            embeddings_val (float)
                embeddings_initializer=initializers.RandomUniform(
                minval=-embeddings_val, maxval=embeddings_val, seed=seed)
            
            num_neg (int, default=1)
                number of negative samping (row direction)
            
            gamma (float)
                tf.keras Embedding layer regularizers.l2(gamma)
            
            loss_wgt_neg (float)
                loss_weights={'y': 1.0, 'neg_y': loss_wgt_neg, 'cor_y': 1.0}
            
            rscore
                user_embedding weights
            
            cscore
                prod_embedding weights
        
        Returns
        -------
        model dict
            models created
        """
        if num_features is None:
            num_features = self.n_sig
        self.num_features = num_features
        self.models = self.wd2v.make_model(num_features=num_features, seed=seed,
                                           num_neg=num_neg,
                                           gamma=gamma, embeddings_val=embeddings_val, loss_wgt_neg=loss_wgt_neg,
                                           rscore=rscore, cscore=cscore)
        return self.models
    
    def plot_hst(self, figsize=(20,5)):
        """plot loss history of 1st phase
        """
        hst_history = self.hst.history
        fig, ax = plt.subplots(1, 3, figsize=figsize)
        ax[0].set_title('loss')
        ax[0].plot(list(range(len(hst_history["loss"]))), hst_history["loss"], label="Train loss", lw=2, color='navy')
        ax[0].plot(list(range(len(hst_history["loss"]))), hst_history["y_loss"], label="Train y loss", color='darkorange')
        ax[0].plot(list(range(len(hst_history["loss"]))), hst_history["neg_y_loss"], label="Train y neg loss", color='darkgreen')
        ax[0].plot(list(range(len(hst_history["loss"]))), hst_history["cor_y_loss"], label="Train y cor loss", color='darkred')
        ax[1].set_title('acc')
        ax[1].plot(list(range(len(hst_history["loss"]))), hst_history["y_binary_accuracy"], label="y accuracy", color='darkorange')
        ax[1].plot(list(range(len(hst_history["loss"]))), hst_history["neg_y_binary_accuracy"], label="y neg accuracy", color='darkgreen')
        ax[1].plot(list(range(len(hst_history["loss"]))), hst_history["cor_y_binary_accuracy"], label="y cor accuracy", color='darkred')
        ax[2].set_title('learning rate')
        ax[2].plot(list(range(len(hst_history["loss"]))), hst_history["lr"], label="learning rate")
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        return ax
    
    def plot_hst2(self, figsize=(20,5)):
        """plot loss history of 2nd phase
        """
        hst_history = self.hst2.history
        fig, ax = plt.subplots(1, 3, figsize=figsize)
        ax[0].set_title('loss')
        ax[0].plot(list(range(len(hst_history["loss"]))), hst_history["loss"], label="Train loss", lw=2, color='navy')
        ax[0].plot(list(range(len(hst_history["loss"]))), hst_history["y_loss"], label="Train y loss", color='darkorange')
        ax[0].plot(list(range(len(hst_history["loss"]))), hst_history["neg_y_loss"], label="Train y neg loss", color='darkgreen')
        ax[0].plot(list(range(len(hst_history["loss"]))), hst_history["cor_y_loss"], label="Train y cor loss", color='darkred')
        ax[1].set_title('acc')
        ax[1].plot(list(range(len(hst_history["loss"]))), hst_history["y_binary_accuracy"], label="y accuracy", color='darkorange')
        ax[1].plot(list(range(len(hst_history["loss"]))), hst_history["neg_y_binary_accuracy"], label="y neg_user accuracy", color='darkgreen')
        ax[1].plot(list(range(len(hst_history["loss"]))), hst_history["cor_y_binary_accuracy"], label="y cor accuracy", color='darkred')
        ax[2].set_title('learning rate')
        ax[2].plot(list(range(len(hst_history["loss"]))), hst_history["lr"], label="learning rate")
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        return ax
    
    def mclust_col(self, n_pca=None, n_init=N_INIT, g_range=G_RANGE, cov_type_list=cov_type_list,
               gm=mixture.GaussianMixture(init_params='kmeans')):
        """Simulate with Gaussian Mixture
        """
        if n_pca is None:
            self.n_pca = self.n_sig
            print('set n_pca to ', self.n_sig)
        else:
            self.n_pca = n_pca
        mclust_col_res = super().mclust_col(n_pca=self.n_pca, 
                                            n_init=n_init, g_range=g_range,
                                            cov_type_list=cov_type_list, gm=gm)
        return mclust_col_res



    


class M01F_li(object):
    """Calculate feature vector using noise reduction
    
    Parameters
    ----------
    .
        X_df (pandas.DataFrame of shape (n_samples, n_features))
            training instances
    """
    
    def __init__(self, X_df):
        self.X_df = X_df
        w, res = self.find_ncomponents(type='pca', n_iter=10, figsize=(10,10))
        
        self.n_sig = ((res[:,1,:].mean(axis=0) + res[:,1,:].std(axis=0)) < w).sum()
        print('suggests number of significant component ->', self.n_sig)
        self.set_n_sig(self.n_sig)
    
    def find_ncomponents(self, type='pca', n_iter=10, figsize=(10,10)):
        """investigate the number of significant components 
        """
        w, res = find_ncomponents_pca(self.X_df.values, n_iter=n_iter, figsize=figsize)
        return w, res
    
    def set_n_sig(self, n_sig=3):
        self.n_sig = n_sig
        self.calc_cor_nonoise()
        self.calc_col_score_nonoise()
        self.calc_X_df_nonoise()
        self.calc_row_fet()
    
    def calc_cor_nonoise(self):
        c = self.X_df.corr()
        m = calc_cor_nonoise(c, n_sig=self.n_sig)
        self.cor_nonoise = m
        return m
    
    def calc_col_score_nonoise(self):
        v = calc_col_score(self.cor_nonoise, n_sig=self.n_sig)
        self.col_fet = v
        return v
    
    def calc_X_df_nonoise(self):
        x_sc2 = calc_mat_nonoise(self.X_df.values, self.n_sig)
        self.X_df_nonoise = pd.DataFrame(x_sc2, index=self.X_df.index, columns=self.X_df.columns)
        return self.X_df_nonoise
    
    def calc_row_fet(self):
        m = cosine_similarity(self.X_df_nonoise.values, self.col_fet.T)
        self.row_fet = m
        return m
    
    def mclust_col(self, n_init=N_INIT, g_range=G_RANGE, cov_type_list=cov_type_list,
               gm=mixture.GaussianMixture(init_params='kmeans')):
        self.mclust_col_res = mclust(self.col_fet, n_init=n_init, g_range=g_range,
                                          cov_type_list=cov_type_list, gm=gm)
        return self.mclust_col_res
    
    def mclust_row(self, n_init=N_INIT, g_range=G_RANGE, cov_type_list=cov_type_list,
               gm=mixture.GaussianMixture(init_params='kmeans')):
        self.mclust_row_res = mclust(self.row_fet, n_init=n_init, g_range=g_range,
                                          cov_type_list=cov_type_list, gm=gm)
        return self.mclust_row_res









def calc_mat_nonoise(mat, n_sig=3):
    """calc matrix using noise reduction
    
    Parameters
    ----------
    .
        mat (ndarray of shape (n_samples, n_features))
            matrix
        
        n_sig (int)
            number of significant components
        
    Returns
    -------
    ndarray of shape(n_samples, n_features)
        matrix created
    """
    ss = StandardScaler()
    ss.fit(mat)
    x_sc = ss.transform(mat)
    u, s, vh = np.linalg.svd(x_sc)
    x_sc2 = u[:,:n_sig].dot(np.diag(s[:n_sig] * s.sum() / s[:n_sig].sum())).dot(vh[:n_sig])
    return x_sc2


def calc_cor_nonoise(c, n_sig=3):
    """calc correlation matrix using noise reduction
    
    Parameters
    ----------
    .
        c (ndarray of shape (n_features, n_features))
            correlation matrix
        
        n_sig (int)
            number of significant components
        
    Returns
    -------
    ndarray of shape(n_features, n_features)
        matrix created
    """
    try:
        w, v = np.linalg.eigh(c)
    except Exception as e:
        print(e)
        c1 = cor_smooth(c)
        w, v = np.linalg.eigh(c1)
        print('"Matrix was not positive definite, smoothing was done"')
    idx = np.argsort(w)[::-1]
    w, v = w[idx], v[:,idx]
    m = v[:,:n_sig].dot(np.diag(w[:n_sig])).dot(v[:,:n_sig].T)
    d = np.sqrt(np.diag(m)).reshape((v.shape[0],1))
    m = m / d / d.T
    return m


def calc_col_score(c, n_sig=None):
    """calc scores of columns side
    
    Parameters
    ----------
    .
        c (ndarray of shape (n_features, n_features))
            correlation matrix
        
        n_sig (int)
            number of significant components
        
    Returns
    -------
    ndarray of shape(n_features, n_sig)
        scores created
    """
    w, v = np.linalg.eigh(c)
    idx = np.argsort(w)[::-1]
    w, v = w[idx], v[:,idx]
    m = v[:,:n_sig].dot(np.diag(np.sqrt(w[:n_sig])))
    return m
    
calc_loadings_nonoise = calc_col_score


def eig(cor_mat):
    """quasi PCA
    """
    sdev2, loadings = np.linalg.eigh(cor_mat)
    idx = np.argsort(sdev2)[::-1]
    sdev2, loadings = sdev2[idx], loadings[:,idx]
    return sdev2, loadings


def create_mat_from_cor(selected_cor, n_samples=100, state=None):
    """create simulated observation matrix using correlation matrix
    of shape (n_samples, selected_cor.shape[0])
    
    Parameters
    ----------
    .
        selected_cor (ndarray of shape (n_features, n_features))
            correlation matrix
        
        n_samples (int)
            number of samples
        
    Returns
    -------
    ndarray of shape(n_samples, n_features)
        created matrix with input correlation
    """
    rs = np.random.RandomState(state)
    r = rs.normal(size=(n_samples, selected_cor.shape[1]))
    cor_r = np.corrcoef(r, rowvar=False)
    ss = StandardScaler()
    r_scaled = ss.fit_transform(r)
    r_scaled = r_scaled.dot(np.linalg.inv(np.linalg.cholesky(cor_r).T))
    np.corrcoef(r_scaled, rowvar=False)
    try:
        r_calced = r_scaled.dot(np.linalg.cholesky(selected_cor).T)
    except:
        r_calced = r_scaled.dot(np.linalg.cholesky(cor_smooth(selected_cor)).T)
    return r_calced


def mclust(df_target, n_init=3, g_range=G_RANGE,
           cov_type_list=cov_type_list,
           gm=mixture.GaussianMixture(init_params='kmeans')):
    """Simulate with Gaussian Mixture
    
    Parameters
    ----------
    .
        df_target (pandas.DataFrame or ndarray of shape (n_samples, n_features))
            data
        
        n_init (int)
            Number of initializations
        
        g_range (int)
            Number of groups to investigate
        
        gm (GaussianMixture instance)
            mixture.GaussianMixture(init_params='kmeans')
    
    Returns
    -------
    res
        Results for each cov_type
    """
    res = {}
    for cov_type in cov_type_list:
        print(cov_type)
        res.setdefault(cov_type)
        res1 = []
        with tqdm(total=len(g_range), file=sys.stdout) as pbar:
            for n_components in g_range:
                res0 = []
                for ii in range(n_init):
                    gm.set_params(n_components=n_components)
                    gm.fit(df_target)
                    # -1 x BIC
                    res0.append(-gm.bic(df_target))
                res1.append(res0)
                pbar.update(1)
        res[cov_type] = np.array(res1)
    return res


def plot_mclust(res, g_range=G_RANGE, figsize=(15, 15), lw=2):
    """plot mclust results
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    
    for ii, k in enumerate(cov_type_list):
        i, j = divmod(ii, 2)
        m = res[k].mean(axis=1)
        std = np.std(res[k], axis=1)
        axes[i,j].plot(g_range, m, label=k, marker='o')
        axes[i,j].fill_between(g_range, m - std,
                         m + std, alpha=0.2, color="darkorange", lw=lw)
        axes[i,j].grid()
        axes[i,j].legend(loc="best")
    return fig, axes


def plot_pca_fa_cv(x_mat, n_obs=None, 
                   n_range=np.arange(2, 16), n_init=10,
                   cv=KFold(n_splits=5, shuffle=True),
                   figsize=(10,10)):
    isCorrelation, c, n_obs = check_cor(x_mat, n_obs)
    
    pca = PCA(svd_solver='full')
    fa = FactorAnalysis()
    pca_scores, fa_scores = [], []
    with tqdm(total=n_init, file=sys.stdout) as pbar:
        for _ in range(n_init):
            if isCorrelation:
                r_calced = create_mat_from_cor(c, n_obs, state=None)
            else:
                r_calced = x_mat
            for n in n_range:
                pca.n_components = n
                fa.n_components = n
                pca_scores.append(np.mean(cross_val_score(pca, r_calced, cv=cv)))
                fa_scores.append(np.mean(cross_val_score(fa, r_calced, cv=cv)))
            pbar.update(1)
    
    pca_scores = np.array(pca_scores).reshape((n_init, -1))
    fa_scores = np.array(fa_scores).reshape((n_init, -1))
    
    '''plot'''
    plt.figure(figsize=figsize)
    pca_m = pca_scores.mean(axis=0)
    pca_std = np.std(pca_scores, axis=0)
    plt.plot(n_range, pca_m, '-o', label='PCA')
    if n_init != 1:
        plt.fill_between(n_range, pca_m-pca_std, pca_m+pca_std, alpha=0.2, color="darkorange")
    
    fa_m = fa_scores.mean(axis=0)
    fa_std = np.std(fa_scores, axis=0)
    plt.plot(n_range, fa_m, '-o', label='FA')
    if n_init != 1:
        plt.fill_between(n_range, fa_m-fa_std, fa_m+fa_std, alpha=0.2, color="darkorange")
    plt.grid()
    plt.legend()
    return pca_scores, fa_scores


def check_cor(x, n_obs=None):
    """check if input x is a correlation matrix
    
    Parameters
    ----------
    .
        x (ndarray of shape (n_samples or n_features, n_features))
            data to check (data or correlation matrix)
        
        n_obs (int or None)
            number of samples
    
    Returns
    -------
    isCorrelation (boolean)
        input x is a correlation matrix or not
    
    c (ndarray)
        correlation matrix
    
    n_obs
        number of samples
    """
    isCorrelation = False
    if x.shape[0] == x.shape[1] and int(np.diag(x).sum().round()) == x.shape[0]:
        isCorrelation = True
        c = x
        if n_obs is None:
            raise Exception('if x is correlation matrix, n_obs must be specified.')
    else:
        c = np.corrcoef(x, rowvar=False)
        n_obs = x.shape[0]
    return isCorrelation, c, n_obs


def find_ncomponents_pca(x, n_obs=None, n_iter=10,
                         figsize=(10,10)):
    """investigate the number of significant components 
    using PCA
    
    fa.parallel https://www.rdocumentation.org/packages/psych/versions/2.1.9/topics/fa.parallel
    see References
    
    Parameters
    ----------
    .
        x (ndarray of shape (n_samples or n_features, n_features))
            data or correlation matrix
        
        n_obs (int or None)
            n_obs must be set if x is a correlation matrix
    """
    n_features = x.shape[1]
    isCorrelation, c, n_obs = check_cor(x, n_obs)
    if isCorrelation:
        x = create_mat_from_cor(c, n_samples=n_obs, state=None)
    
    '''calc PCA eigen values'''
    w = get_eigval(x, n_features)
    
    def func(mat):
        '''resampling'''
        resampling_w = np.array([0.0] * n_features)
        if not isCorrelation:
            new_mat = np.apply_along_axis(np.random.choice, 0, mat, size=mat.shape[0])
            w = get_eigval(new_mat, n_features)
            resampling_w = np.sort(w)[::-1]
        '''sim'''
        new_mat = np.random.normal(size=(n_obs, n_features))
        sim_w = get_eigval(new_mat, n_features)
        return resampling_w, sim_w
    res = []
    for ii in range(n_iter):
        res0 = func(x)
        res.append(res0)
    res = np.array(res)
    
    '''plot'''
    plt.figure(figsize=figsize)
    plt.plot(np.arange(x.shape[1])+1, w, '-o', label='PCA', color='blue')
    m = res.mean(axis=0)[0]
    if m.sum() != 0:
        std = res.std(axis=0)[0]
        plt.plot(np.arange(x.shape[1])+1, m, '-o', label='resampling', color='darkblue')
        plt.fill_between(np.arange(x.shape[1])+1, m-std, m+std, alpha=0.2, color="darkorange")
    m = res.mean(axis=0)[1]
    std = res.std(axis=0)[1]
    plt.plot(np.arange(x.shape[1])+1, m, '-o', label='sim', color='darkorange')
    plt.fill_between(np.arange(x.shape[1])+1, m-std, m+std, alpha=0.2, color="darkblue")
    plt.legend(loc='best')
    plt.grid()
    return w, res


def get_eigval(mat, n_features):
    """calc eigenvalues using scikit-learn PCA
    mainly used in find_ncomponents_pca
    """
    pca = PCA(n_components=n_features)
    pca.fit(mat)
    cv = pca.get_covariance()
    cv = cv / np.sqrt(np.diag(cv)) / np.sqrt(np.diag(cv)).T
    w, _ = np.linalg.eig(cv)
    w = np.sort(w)[::-1]
    return w


def cor_smooth(cor_mat, tol=10e-12):
    """smooth a non-positive definite correlation matrix 
    to make it positive definite
    
    Parameters
    ----------
    .
        cor_mat (ndarray of shape (n_features, n_features))
            correlation matirx to process
        
        tol (float)
            the minimum acceptable eigenvalue
        
    Returns
    -------
    ndarray of shape (n_features, n_features)
        the smoothed correlation matrix if smoothing was 
        in fact necessary
    """
    w, v = np.linalg.eigh(cor_mat)
    if w.min() < np.finfo(float).eps:
        w[w < tol] = tol * 100
        nvar = cor_mat.shape[0]
        tot = w.sum()
        w2 = w * nvar / tot
        new_cor = v.dot(np.diag(w2)).dot(v.T)
        d = np.sqrt(np.diag(new_cor)).reshape((nvar,1))
        new_cor = new_cor / d / d.T
        return new_cor
    return cor_mat





def create_random_map2(mat, n_std=2, kshape_start=12, noise=True):
    kshape = np.arange(2, kshape_start+1)[::-1]
    
    sc = StandardScaler()
    sc.fit(mat)
    mat_sc = sc.transform(mat)
    
    pca = PCA(n_components=mat_sc.shape[1], svd_solver='full')
    pca.fit(mat_sc)
    
    tmp = np.linspace(-n_std, n_std, kshape[0]).reshape(kshape[0], 1) * pca.components_[0] * np.sqrt(pca.explained_variance_)[0]
    
    for ii in range(1, mat_sc.shape[1]):
        idx = np.arange(ii+2)
        idx[0], idx[ii] = idx[ii], idx[0]
        if len(kshape) <= ii:
            ii_kshape = 2
        else:
            ii_kshape = kshape[ii]
#        ii_kshape = kshape[ii]
#        if ii_kshape < 3:
#            ii_kshape = 3
        tmp = np.transpose(np.tile(tmp, [ii_kshape]+[1]*tmp.ndim), idx) + \
            np.linspace(-n_std, n_std, ii_kshape).reshape(ii_kshape, 1) * pca.components_[ii] * np.sqrt(pca.explained_variance_)[ii]
    
    init_map = tmp.reshape((-1, mat_sc.shape[1]))
    init_map = sc.inverse_transform(init_map)
    return init_map

def create_random_map(mat):
    ss = StandardScaler()
    mat_sc = ss.fit_transform(mat)

    u, s, vh = np.linalg.svd(mat_sc)
    x1 = mat_sc.dot(vh)
    n_samples = x1.shape[0]
    ll = []
    for ii in range(x1.shape[1]):
        min_x = x1[:,ii].min()
        max_x = x1[:,ii].max()
        ll.append((max_x - min_x) * np.random.random_sample(n_samples) + min_x)
    x2 = np.c_[ll].T.dot(vh.T)
    mat_ret = ss.inverse_transform(x2)
    return mat_ret

def linear_init(X, shape=(3, 3), n_std=2):
    sc = StandardScaler()
    sc.fit(X)
    x_sc = sc.transform(X)
    
    pca = PCA(n_components=2, random_state=0)
    pca.fit(x_sc)
    x_tick = pca.components_[0] * np.sqrt(pca.explained_variance_)[0]
    y_tick = pca.components_[1] * np.sqrt(pca.explained_variance_)[1]
        
    l = []
    for xi in np.linspace(-n_std, n_std, shape[0]):
        tmp = np.linspace(-n_std, n_std, shape[1]).reshape(shape[1],1) * y_tick
        tmp += xi * x_tick
        l.append(tmp)
    init_map = np.vstack(l)
    return sc.inverse_transform(init_map)

def find_nclusters_gap(mat, kshape_start=12, n_range=np.arange(2, 31), nn=20,
                       figsize=(20,8)):
    #rand_map0 = create_random_map(mat, noise=False, kshape_start=kshape_start)
    
    def score(mat, cls):
        cls_unique = np.unique(cls)
        sc = 0.0
        for icls in cls_unique:
            mat_part = mat[cls == icls,:]
            sc += pairwise.euclidean_distances(mat_part).sum()
        return sc / mat.shape[0]**2
    
    scores = {'data': [], 'rand': []}
    with tqdm(total=nn, file=sys.stdout) as pbar:
        for jj in range(nn):
            #idx = np.random.choice(np.arange(rand_map0.shape[0]), size=mat.shape[0], replace=False)
            #rand_map = rand_map0[idx,:]
            rand_map = create_random_map(mat)
            for ii in n_range:
                kmeans = KMeans(n_clusters=ii, n_init=1)
                kmeans.fit(mat)
                # scores['data'].append(-kmeans.score(mdl.fet_row_aggr))
                cls = kmeans.predict(mat)
                scores['data'].append(score(mat, cls))
        
                #kmeans = KMeans(n_clusters=ii, n_init=1)
                #kmeans.fit(rand_map)
                # scores['rand'].append(-kmeans.score(rand_map))
                #scores['rand'].append(score(rand_map, kmeans.predict(rand_map)))
                scores['rand'].append(score(rand_map, cls))
            pbar.update(1)
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    axes[0].plot(n_range, np.array(scores['data']).reshape((nn, -1)).T.mean(axis=1), color='darkblue', marker='o', label='data')
    axes[0].plot(n_range, np.array(scores['rand']).reshape((nn, -1)).T.mean(axis=1), color='darkorange', marker='o', label='rand')
    axes[0].grid()
    axes[0].legend()
    
    axes[1].plot(n_range, -np.array(scores['data']).reshape((nn, -1)).T.mean(axis=1)/np.array(scores['rand']).reshape((nn, -1)).T.mean(axis=1), marker='o')
    axes[1].grid()
    return scores





