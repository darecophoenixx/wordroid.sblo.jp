'''
Copyright (c) 2018 Norio Tamada
Released under the MIT license
https://github.com/darecophoenixx/wordroid.sblo.jp/blob/master/lib/feature_eng/LICENSE.md
'''

import itertools
import random
from collections import Mapping
import logging

import numpy as np
import scipy
import gensim
import pandas as pd

from sklearn import mixture
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from keras.layers import Input, Embedding, LSTM, Dense, Dropout, Lambda, \
    Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Conv2DTranspose, \
    GlobalAveragePooling1D, MaxPooling1D, MaxPooling2D, \
    concatenate, Flatten, Average, Activation, \
    RepeatVector, Permute, Reshape, Dot, \
    multiply, dot, add
from keras.models import Model, Sequential
from keras import losses
from keras.callbacks import BaseLogger, ProgbarLogger, Callback, History
from keras.wrappers.scikit_learn import KerasClassifier
from keras import regularizers
from keras import initializers
from keras.metrics import categorical_accuracy
from keras.constraints import maxnorm, non_neg
from keras import backend as K

import matplotlib.pyplot as plt
import seaborn as sns
           
from feature_eng.lowcols2 import (
    WD2vec
)



N_INIT = 30
G_RANGE = np.arange(3,16)
cov_type_list = ['full', 'tied', 'diag', 'spherical']


class M01F(object):
    
    def __init__(self, X_df):
        self.X_df = X_df
        self.wd2v = WD2vec(X_df)
    
    def make_model(self, num_features=None, seed=10001,
                   num_neg=2,
                   gamma=0.0, embeddings_val=0.1, loss_wgt_neg=0.05,
                   rscore=None, cscore=None):
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
        self.hst, self.hst2 = self.wd2v.train(epochs=epochs, batch_size=batch_size, verbose=verbose, lr0=lr0,
                                    shuffle=shuffle, flag_early_stopping=flag_early_stopping, base=base,
                                    callbacks=callbacks, callbacks_add=callbacks_add,
                                    use_multiprocessing=use_multiprocessing, workers=workers)
        
        self.df_wgt_col = pd.DataFrame(self.wgt_col, index=self.X_df.columns, columns=['fet'+str(ee) for ee in range(self.num_features)])
        self.df_wgt_row = pd.DataFrame(self.wgt_row, index=self.X_df.index, columns=['fet'+str(ee) for ee in range(self.num_features)])
        self.calc_cor()
        self.eig()
        return self
    
    def plot_hst(self):
        hst_history = self.hst.history
        fig, ax = plt.subplots(1, 3, figsize=(20,5))
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
    
    def plot_hst2(self):
        hst_history = self.hst2.history
        fig, ax = plt.subplots(1, 3, figsize=(20,5))
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
        '''行と列の特徴量を列の特徴量で評価する'''
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
        '''quasi PCA'''
        sdev2, self.loadings = np.linalg.eig(self.df_cor.values)
        self.sdev = np.sqrt(sdev2)
        return self.sdev, self.loadings
    
    def plot_eig(self, figsize=(10, 10), marker='-o'):
        plt.figure(figsize=figsize)
        plt.plot(np.arange(len(self.sdev))+1, self.sdev, marker)
        plt.grid()
    
    def set_npca(self, n_pca=3):
        self.n_pca = n_pca
        self.loadings_selected = self.loadings[:,:n_pca]
        self.fet_col_aggr = self.df_cor.values.dot(self.loadings_selected)
        self.fet_row_aggr = self.fet_row_cnvt.values.dot(self.loadings_selected)

    def mclust(self, df, n_init=N_INIT, g_range=G_RANGE, cov_type_list=cov_type_list,
               gm=mixture.GaussianMixture(init_params='kmeans')):
        self.g_range = g_range
        mclust_res = mclust(df, n_init=n_init, g_range=g_range, cov_type_list=cov_type_list, gm=gm)
        return mclust_res
    
    def mclust_col(self, n_pca=3, n_init=N_INIT, g_range=G_RANGE, cov_type_list=cov_type_list,
               gm=mixture.GaussianMixture(init_params='kmeans')):
        self.set_npca(n_pca)
        self.mclust_col_res = self.mclust(pd.DataFrame(self.fet_col_aggr), n_init=n_init, g_range=g_range,
                                          cov_type_list=cov_type_list, gm=gm)
        return self.mclust_col_res
    
    def mclust_row(self, n_pca=None, n_init=N_INIT, g_range=G_RANGE, cov_type_list=cov_type_list,
               gm=mixture.GaussianMixture(init_params='kmeans')):
        '''self.n_pca must be set'''
        assert self.n_pca
        if n_pca is None:
            n_pca = self.n_pca
        self.set_npca(n_pca)
        self.mclust_row_res = self.mclust(pd.DataFrame(self.fet_row_aggr), n_init=n_init, g_range=g_range,
                                          cov_type_list=cov_type_list, gm=gm)
        return self.mclust_row_res
    
    def plot_mclust(self, res, figsize=(15, 15), lw=2):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        
        for ii, k in enumerate(cov_type_list):
            i, j = divmod(ii, 2)
            m = res[k].mean(axis=1)
            std = np.std(res[k], axis=1)
            axes[i,j].plot(self.g_range, m, label=k, marker='o')
            axes[i,j].fill_between(self.g_range, m - std,
                             m + std, alpha=0.2, color="darkorange", lw=lw)
            axes[i,j].grid()
            axes[i,j].legend(loc="best")
        return fig, axes
    
    def plot_mclust_col(self, figsize=(15, 15), lw=2):
        self.plot_mclust(self.mclust_col_res, figsize, lw)
    
    def plot_mclust_row(self, figsize=(15, 15), lw=2):
        self.plot_mclust(self.mclust_row_res, figsize, lw)
    
    def plot_pca_fa_cv(self, df=None, n_range=np.arange(2, 16), n_init=10,
                       figsize=(10,10)):
        if df is None:
            c = self.df_cor.values
        pca = PCA(svd_solver='full')
        fa = FactorAnalysis()
        pca_scores, fa_scores = [], []
        for _ in range(n_init):
            if df is None:
                r_calced = self._create_mat_selected_cor(c, self.X_df)
            else:
                r_calced = df.values
            for n in n_range:
                pca.n_components = n
                fa.n_components = n
                pca_scores.append(np.mean(cross_val_score(pca, r_calced)))
                fa_scores.append(np.mean(cross_val_score(fa, r_calced)))
        
        pca_scores = np.array(pca_scores).reshape((n_init, -1))
        fa_scores = np.array(fa_scores).reshape((n_init, -1))
        self.cv_pca_scores = pca_scores
        self.cv_fa_scores = fa_scores
        
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
    
    def _create_mat_selected_cor(self, selected_cor, df):
        r = np.random.normal(size=(df.shape))
        cor_r = np.corrcoef(r, rowvar=False)
        ss = StandardScaler()
        r_scaled = ss.fit_transform(r)
        r_scaled = r_scaled.dot(np.linalg.inv(np.linalg.cholesky(cor_r).T))
        np.corrcoef(r_scaled, rowvar=False)
        r_calced = r_scaled.dot(np.linalg.cholesky(selected_cor).T)
        return r_calced
    
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









def mclust(df_target, n_init=3, g_range=G_RANGE,
           cov_type_list=cov_type_list,
           gm=mixture.GaussianMixture(init_params='kmeans')):
    res = {}
    for cov_type in cov_type_list:
        print(cov_type)
        res.setdefault(cov_type)
        res1 = []
        for n_components in g_range:
            res0 = []
            for ii in range(n_init):
                gm.set_params(n_components=n_components)
                gm.fit(df_target)
                # -1 x BIC
                res0.append(-gm.bic(df_target))
            res1.append(res0)
        res[cov_type] = np.array(res1)
    return res















