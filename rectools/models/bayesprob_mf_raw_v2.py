from six.moves import xrange
from functools import partial
import numpy as np
from numpy.random import RandomState
import scipy.sparse as sparse


def make_ratings(n_users, n_items, min_rating_per_user, max_rating_per_user,
                 rating_choices, seed=None, shuffle=True):
    """Randomly generate a (user_id, item_id, rating) array

    Return
    ------
        ndarray with shape (n_samples, 3)

    """
    if not (isinstance(rating_choices, list) or
            isinstance(rating_choices, tuple)):
        raise ValueError("'rating_choices' must be a list or tuple")
    if min_rating_per_user < 0 or min_rating_per_user >= n_items:
        raise ValueError("invalid 'min_rating_per_user' invalid")
    if (min_rating_per_user > max_rating_per_user) or \
       (max_rating_per_user >= n_items):
        raise ValueError("invalid 'max_rating_per_user' invalid")

    rs = RandomState(seed=seed)
    user_arrs = []
    for user_id in xrange(n_users):
        item_count = rs.randint(min_rating_per_user, max_rating_per_user)
        item_ids = rs.choice(n_items, item_count, replace=False)
        ratings = rs.choice(rating_choices, item_count)
        arr = np.stack(
            [np.repeat(user_id, item_count), item_ids, ratings], axis=1)
        user_arrs.append(arr)

    ratings = np.array(np.vstack(user_arrs))
    ratings[:, 2] = ratings[:, 2].astype('float')
    if shuffle:
        rs.shuffle(ratings)
    return ratings


def load_movielens_ratings(ratings_file, separator):
    with open(ratings_file) as f:
        ratings = []
        for line in f:
            line = line.split(separator)[:3]
            line = [int(l) for l in line]
            ratings.append(line)
        ratings = np.array(ratings)
    return ratings


def build_user_item_matrix(n_users, n_items, ratings):
    """Build user-item matrix

    Return
    ------
        sparse matrix with shape (n_users, n_items)
    """
    data = ratings[:, 2]
    row_ind = ratings[:, 0]
    col_ind = ratings[:, 1]
    shape = (n_users, n_items)
    return sparse.csr_matrix((data, (row_ind, col_ind)), shape=shape)

"""
Reference paper: "Bayesian Probabilistic Matrix Factorization using MCMC"
                 R. Salakhutdinov and A.Mnih.
                 25th International Conference on Machine Learning (ICML-2008)

Reference Matlab code: http://www.cs.toronto.edu/~rsalakhu/BPMF.html
"""

import logging
from six.moves import xrange
import numpy as np
from numpy.linalg import inv, cholesky
from numpy.random import RandomState
from scipy.stats import wishart

# from .base import ModelBase
# from .exceptions import NotFittedError
# from .utils.datasets import build_user_item_matrix
# from .utils.validation import check_ratings
# from .utils.evaluation import RMSE

def RMSE(estimation, truth):
    """Root Mean Square Error"""
    estimation = np.float64(estimation)
    truth = np.float64(truth)
    num_sample = estimation.shape[0]

    sse = np.sum(np.square(truth - estimation))
    return np.sqrt(np.divide(sse, num_sample - 1))


def check_ratings(ratings, n_user, n_item, max_rating=None, min_rating=None):
    """Check rating array

    'ratings' must be a matrix with shape (n_sample, 3)
    and each row is (user_id, item_id, rating).
    Both user_id and item_id start from 0.

    Return
    ------
        None

    """
    if ratings.shape[1] != 3:
        print("Invalid rating format. number of column must be 3")

    if not np.all(ratings[:, :2] >= 0):
        raise ValueError("negative user_id or item_id")

    max_user_id = ratings[:, 0].max()
    if max_user_id >= n_user:
        print(max_user_id, n_user)
        raise ValueError("max user_id >= %d", n_user)

    max_item_id = ratings[:, 1].max()
    if max_item_id >= n_item:
        raise ValueError("max n_item >= %d", n_item)

    if max_rating is not None and np.any(ratings[:, 2] > max_rating):
        raise ValueError("max rating >= %d", max_rating)

    if min_rating is not None and np.any(ratings[:, 2] < min_rating):
        raise ValueError("min rating >= %d", min_rating)


class BPMF_V2():
    """Bayesian Probabilistic Matrix Factorization
    """

    def __init__(self, n_user, n_item, n_feature, beta=2.0, beta_user=2.0,
                 df_user=None, mu0_user=0., beta_item=2.0, df_item=None,
                 mu0_item=0., converge=1e-5, seed=None, max_rating=None,
                 min_rating=None):

        self.n_user = n_user
        self.n_item = n_item
        self.n_feature = n_feature
        self.rand_state = RandomState(seed)
        self.max_rating = float(max_rating) if max_rating is not None else None
        self.min_rating = float(min_rating) if min_rating is not None else None
        self.converge = converge

        # Hyper Parameter
        self.beta = beta

        # Inv-Whishart (User features)
        self.WI_user = np.eye(n_feature, dtype='float64')
        self.beta_user = beta_user
        self.df_user = int(df_user) if df_user is not None else n_feature
        self.mu0_user = np.repeat(mu0_user, n_feature).reshape(n_feature, 1)  # a vector

        # Inv-Whishart (item features)
        self.WI_item = np.eye(n_feature, dtype='float64')
        self.beta_item = beta_item
        self.df_item = int(df_item) if df_item is not None else n_feature
        self.mu0_item = np.repeat(mu0_item, n_feature).reshape(n_feature, 1)

        # Latent Variables
        self.mu_user = np.zeros((n_feature, 1), dtype='float64')
        self.mu_item = np.zeros((n_feature, 1), dtype='float64')

        self.alpha_user = np.eye(n_feature, dtype='float64')
        self.alpha_item = np.eye(n_feature, dtype='float64')

        # initializes the user features randomly.
        # (There is no special reason to use 0.3)
        self.user_features_ = 0.3 * self.rand_state.rand(n_user, n_feature)
        self.item_features_ = 0.3 * self.rand_state.rand(n_item, n_feature)

        # average user/item features
        self.avg_user_features_ = np.zeros((n_user, n_feature))
        self.avg_item_features_ = np.zeros((n_item, n_feature))

        # data state
        self.iter_ = 0
        self.mean_rating_ = None
        self.ratings_csr_ = None
        self.ratings_csc_ = None

    def _update_average_features(self, iteration):
        self.avg_user_features_ *= (iteration / (iteration + 1.))
        self.avg_user_features_ += (self.user_features_ / (iteration + 1.))
        self.avg_item_features_ *= (iteration / (iteration + 1.))
        self.avg_item_features_ += (self.item_features_ / (iteration + 1.))

    def fit(self, ratings, n_iters=50):

        check_ratings(ratings, self.n_user, self.n_item,
                      self.max_rating, self.min_rating)

        self.mean_rating_ = np.mean(ratings[:, 2])

        # only two different ways of building the matrix. 
        # csr user-item matrix for fast row access (user update)
        self.ratings_csr_ = build_user_item_matrix(
            self.n_user, self.n_item, ratings)
        # keep a csc matrix for fast col access (item update)
        self.ratings_csc_ = self.ratings_csr_.tocsc() 

        last_rmse = None
        for iteration in xrange(n_iters):
            # THE FOLLOWING ARE THE GIBBS SAMPLING STEP 
            # update item & user parameter
            self._update_item_params()
            self._update_user_params()

            # update item & user features
            self._udpate_item_features()
            self._update_user_features()

            # in order to make a meaningful MCMC
            # we need to sample from the correct distribution
            self._update_average_features(self.iter_)
            self.iter_ += 1

            # compute RMSE
            train_preds = self.predict(ratings[:, :2])
            train_rmse = RMSE(train_preds, ratings[:, 2])
            print(f"iteration: {self.iter_}, train RMSE: {train_rmse}")
            # stop when converge
            if last_rmse and abs(train_rmse - last_rmse) < self.converge:
                print(f'converges at iteration {self.iter_}. stop.')
                break
            else:
                last_rmse = train_rmse

        return self.user_features_, self.item_features_

    def predict(self, data):
        if not self.mean_rating_:
            print('NotFittedError')
            # raise NotFittedError()

        u_features = self.avg_user_features_.take(data.take(0, axis=1), axis=0)
        i_features = self.avg_item_features_.take(data.take(1, axis=1), axis=0)
        preds = np.sum(u_features * i_features, 1) + self.mean_rating_

        if self.max_rating:  # cut the prediction rate. 
            preds[preds > self.max_rating] = self.max_rating

        if self.min_rating:
            preds[preds < self.min_rating] = self.min_rating
        return preds

    def _update_item_params(self):
        N = self.n_item
        X_bar = np.mean(self.item_features_, 0).reshape((self.n_feature, 1))
        S_bar = np.cov(self.item_features_.T)

        diff_X_bar = self.mu0_item - X_bar

        # W_{0}_star
        WI_post = inv(inv(self.WI_item) +
                      N * S_bar +
                      np.dot(diff_X_bar, diff_X_bar.T) *
                      (N * self.beta_item) / (self.beta_item + N))

        # Note: WI_post and WI_post.T should be the same.
        #       Just make sure it is symmertic here
        WI_post = (WI_post + WI_post.T) / 2.0

        # update alpha_item
        df_post = self.df_item + N
        self.alpha_item = wishart.rvs(df_post, WI_post, 1, self.rand_state)

        # update mu_item
        mu_mean = (self.beta_item * self.mu0_item + N * X_bar) / \
            (self.beta_item + N)
        mu_var = cholesky(inv(np.dot(self.beta_item + N, self.alpha_item)))
        # print 'lam', lam.shape
        self.mu_item = mu_mean + np.dot(
            mu_var, self.rand_state.randn(self.n_feature, 1))
        # print 'mu_item', self.mu_item.shape

    def _update_user_params(self):
        # same as _update_user_params
        N = self.n_user
        X_bar = np.mean(self.user_features_, 0).reshape((self.n_feature, 1))
        S_bar = np.cov(self.user_features_.T)

        # mu_{0} - U_bar
        diff_X_bar = self.mu0_user - X_bar

        # W_{0}_star
        WI_post = inv(inv(self.WI_user) +
                      N * S_bar +
                      np.dot(diff_X_bar, diff_X_bar.T) *
                      (N * self.beta_user) / (self.beta_user + N))
        # Note: WI_post and WI_post.T should be the same.
        #       Just make sure it is symmertic here
        WI_post = (WI_post + WI_post.T) / 2.0

        # update alpha_user
        df_post = self.df_user + N
        # LAMBDA_{U} ~ W(W{0}_star, df_post)
        self.alpha_user = wishart.rvs(df_post, WI_post, 1, self.rand_state)

        # update mu_user
        # mu_{0}_star = (beta_{0} * mu_{0} + N * U_bar) / (beta_{0} + N)
        mu_mean = (self.beta_user * self.mu0_user + N * X_bar) / \
                  (self.beta_user + N)

        # decomposed inv(beta_{0}_star * LAMBDA_{U})
        mu_var = cholesky(inv(np.dot(self.beta_user + N, self.alpha_user)))
        # sample multivariate gaussian
        self.mu_user = mu_mean + np.dot(
            mu_var, self.rand_state.randn(self.n_feature, 1))

    def _udpate_item_features(self):
        # Gibbs sampling for item features
        for item_id in xrange(self.n_item):
            indices = self.ratings_csc_[:, item_id].indices
            features = self.user_features_[indices, :]
            rating = self.ratings_csc_[:, item_id].data - self.mean_rating_
            rating = np.reshape(rating, (rating.shape[0], 1))

            covar = inv(self.alpha_item +
                        self.beta * np.dot(features.T, features))
            lam = cholesky(covar)

            temp = (self.beta * np.dot(features.T, rating) +
                    np.dot(self.alpha_item, self.mu_item))

            mean = np.dot(covar, temp)
            temp_feature = mean + np.dot(
                lam, self.rand_state.randn(self.n_feature, 1))
            self.item_features_[item_id, :] = temp_feature.ravel()

    def _update_user_features(self):
        # Gibbs sampling for user features
        for user_id in xrange(self.n_user):
            indices = self.ratings_csr_[user_id, :].indices
            features = self.item_features_[indices, :]
            rating = self.ratings_csr_[user_id, :].data - self.mean_rating_
            rating = np.reshape(rating, (rating.shape[0], 1))

            covar = inv(
                self.alpha_user + self.beta * np.dot(features.T, features))
            lam = cholesky(covar)
            # aplha * sum(V_j * R_ij) + LAMBDA_U * mu_u
            temp = (self.beta * np.dot(features.T, rating) +
                    np.dot(self.alpha_user, self.mu_user))
            # mu_i_star
            mean = np.dot(covar, temp)
            temp_feature = mean + np.dot(
                lam, self.rand_state.randn(self.n_feature, 1))
            self.user_features_[user_id, :] = temp_feature.ravel()
