import numpy as np
import pandas as pd
import Config as cfg

#
# def mahalanobis(x=None, data=None, cov=None):
#     """Compute the Mahalanobis Distance between each row of x and the data
#     x    : vector or matrix of data with, say, p columns.
#     data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
#     cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
#     """
#     x_minus_mu = x - np.mean(data)
#     if not cov:
#         cov = np.cov(data.values.T)
#     inv_covmat = sp.linalg.inv(cov)
#     left_term = np.dot(x_minus_mu, inv_covmat)
#     mahal = np.dot(left_term, x_minus_mu.T)
#     return mahal.diagonal()
#
#
# class MahalanobisBinaryClassifier():
#     def __init__(self, xtrain, ytrain):
#         pos_indices = np.where(ytrain == 1)
#         neg_indices = np.where(ytrain == -1)
#         self.xtrain_pos = xtrain[pos_indices]
#         self.xtrain_neg = xtrain[neg_indices]
#
#     def predict_proba(self, xtest):
#         pos_neg_dists = [(p,n) for p, n in zip(mahalanobis(xtest, self.xtrain_pos), mahalanobis(xtest, self.xtrain_neg))]
#         return np.array([(1-n/(p+n), 1-p/(p+n)) for p,n in pos_neg_dists])
#
#     def predict(self, xtest):
#         return np.array([np.argmax(row) for row in self.predict_proba(xtest)])

import numpy as np
import scipy as sp


class MahalanobisClassifier:
    def __init__(self, samples, labels):
        self.clusters = {}
        for lbl in np.unique(labels):
            indices = np.concatenate(np.where(labels == lbl))
            samples = pd.DataFrame(samples)
            self.clusters[lbl] = samples.loc[indices, :]

    def mahalanobis(self, x, data, all_ski=True, cov=None):
        """Compute the Mahalanobis Distance between each row of x and the data
        x    : vector or matrix of data with, say, p columns.
        data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
        cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
        """

        data = data.iloc[:, :-1]

        if cfg.TRAIN_MODELS:
            x = x.iloc[:, :-1]

        x_minus_mu = x - np.mean(data)
        if not cov:
            cov = np.cov(data.values.T)
        if all_ski is False:
            inv_covmat = cov
        else:
            inv_covmat = sp.linalg.inv(cov)
        left_term = np.dot(x_minus_mu, inv_covmat)
        mahal = np.dot(left_term, x_minus_mu.T)
        if not cfg.TRAIN_MODELS:
            mahal = np.array([np.array([mahal])])
        return mahal.diagonal()

    def predict_probability(self, unlabeled_samples, all_ski=True):
        dists = np.array([])

        def dist2prob(D):
            row_sums = D.sum(axis=1)
            D_norm = (D / row_sums[:, np.newaxis])
            S = 1 - D_norm
            row_sums = S.sum(axis=1)
            S_norm = (S / row_sums[:, np.newaxis])
            return S_norm

            # Distance of each sample from all clusters

        for lbl in self.clusters:
            tmp_dists = self.mahalanobis(unlabeled_samples, self.clusters[lbl], all_ski)
            if len(dists) != 0:
                dists = np.column_stack((dists, tmp_dists))
            else:
                dists = tmp_dists

        return dist2prob(dists)

    def predict_class(self, unlabeled_sample, ind2label, all_ski=True):
        return np.array([ind2label[np.argmax(row)] for row in self.predict_probability(unlabeled_sample, all_ski)])