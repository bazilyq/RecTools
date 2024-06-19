from six.moves import xrange
import numpy as np
from numpy.linalg import inv, cholesky
from numpy.random import RandomState
from scipy.stats import wishart
import scipy.sparse as sparse
import time
from numba import njit


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


def RMSE(estimation, truth):
    """Root Mean Square Error"""
    estimation = np.float64(estimation)
    truth = np.float64(truth)
    num_sample = estimation.shape[0]

    sse = np.sum(np.square(truth - estimation))
    return np.sqrt(np.divide(sse, num_sample - 1))


def _update_item_params(n_item, item_features_, n_feature, mu0_item, WI_item, beta_item, df_item, rand_state):
    N = n_item
    X_bar = np.mean(item_features_, 0).reshape((n_feature, 1))
    S_bar = np.cov(item_features_.T)

    diff_X_bar = mu0_item - X_bar

    WI_post = inv(inv(WI_item) +
                  N * S_bar +
                  np.dot(diff_X_bar, diff_X_bar.T) *
                  (N * beta_item) / (beta_item + N))

    WI_post = (WI_post + WI_post.T) / 2.0

    df_post = df_item + N
    alpha_item = wishart.rvs(df_post, WI_post, 1, rand_state)

    mu_mean = (beta_item * mu0_item + N * X_bar) / \
        (beta_item + N)
    mu_var = cholesky(inv(np.dot(beta_item + N, alpha_item)))

    mu_item = mu_mean + np.dot(
        mu_var, rand_state.randn(n_feature, 1))
    
    return mu_item, alpha_item
    

def _update_user_params(n_user, user_features_, n_feature, mu0_user, WI_user, beta_user, df_user, rand_state):
    N = n_user
    X_bar = np.mean(user_features_, 0).reshape((n_feature, 1))
    S_bar = np.cov(user_features_.T)

    diff_X_bar = mu0_user - X_bar

    WI_post = inv(inv(WI_user) +
                    N * S_bar +
                    np.dot(diff_X_bar, diff_X_bar.T) *
                    (N * beta_user) / (beta_user + N))

    WI_post = (WI_post + WI_post.T) / 2.0

    df_post = df_user + N
    alpha_user = wishart.rvs(df_post, WI_post, 1, rand_state)

    mu_mean = (beta_user * mu0_user + N * X_bar) / \
                (beta_user + N)
    mu_var = cholesky(inv(np.dot(beta_user + N, alpha_user)))

    mu_user = mu_mean + np.dot(
        mu_var, rand_state.randn(n_feature, 1))
    
    return mu_user, alpha_user

@njit
def fun(dotdot):
    covar = inv(dotdot)
    return covar

def _udpate_item_features(n_item, ratings_csc_, user_features_, mean_rating_, alpha_item, beta, mu_item, rand_state, n_feature, item_features_):
    # Gibbs sampling for item features
    for item_id in xrange(n_item):
        indices = ratings_csc_[:, item_id].indices
        features = user_features_[indices, :]
        rating = ratings_csc_[:, item_id].data - mean_rating_
        rating = np.reshape(rating, (rating.shape[0], 1))

        dotdot = alpha_item + beta * np.dot(features.T, features)
        covar = fun(dotdot)
        # covar = inv(dotdot) # самая тяжелая операция
        lam = cholesky(covar)

        temp = (beta * np.dot(features.T, rating) +
                np.dot(alpha_item, mu_item))

        mean = np.dot(covar, temp)
        temp_feature = mean + np.dot(
            lam, rand_state.randn(n_feature, 1))
        item_features_[item_id, :] = temp_feature.ravel()

def _update_user_features(n_user, ratings_csr_, item_features_, mean_rating_, alpha_user, beta, mu_user, rand_state, n_feature, user_features_):
    # Gibbs sampling for user features
    for user_id in xrange(n_user):
        indices = ratings_csr_[user_id, :].indices
        features = item_features_[indices, :]
        rating = ratings_csr_[user_id, :].data - mean_rating_
        rating = np.reshape(rating, (rating.shape[0], 1))

        dotdot = alpha_user + beta * np.dot(features.T, features)
        covar = fun(dotdot)
        # covar = inv(dotdot) # самая тяжелая операция
        lam = cholesky(covar)

        temp = (beta * np.dot(features.T, rating) +
                np.dot(alpha_user, mu_user))

        mean = np.dot(covar, temp)
        temp_feature = mean + np.dot(
            lam, rand_state.randn(n_feature, 1))
        user_features_[user_id, :] = temp_feature.ravel()


def _update_average_features(iteration, avg_user_features_, user_features_, avg_item_features_, item_features_):
    avg_user_features_ *= (iteration / (iteration + 1.))
    avg_user_features_ += (user_features_ / (iteration + 1.))
    avg_item_features_ *= (iteration / (iteration + 1.))
    avg_item_features_ += (item_features_ / (iteration + 1.))

def predict(data, mean_rating_, avg_user_features_, avg_item_features_, max_rating, min_rating):
    if not mean_rating_:
        print('NotFittedError')
        # raise NotFittedError()

    u_features = avg_user_features_.take(data.take(0, axis=1), axis=0)
    i_features = avg_item_features_.take(data.take(1, axis=1), axis=0)
    preds = np.sum(u_features * i_features, 1) + mean_rating_

    if max_rating:  # cut the prediction rate. 
        preds[preds > max_rating] = max_rating

    if min_rating:
        preds[preds < min_rating] = min_rating
    return preds

def fit(ratings, n_user, n_item, n_feature, beta=2.0, beta_user=2.0,
        df_user=None, mu0_user=0., beta_item=2.0, df_item=None,
        mu0_item=0., converge=1e-5, seed=None, max_rating=None,
        min_rating=None,
        n_iters=50
        ):
    

    print(seed)
    rand_state = RandomState(seed)
    max_rating = float(max_rating) if max_rating is not None else None
    min_rating = float(min_rating) if min_rating is not None else None

    WI_user = np.eye(n_feature, dtype='float64')
    beta_user = beta_user
    df_user = int(df_user) if df_user is not None else n_feature
    mu0_user = np.repeat(mu0_user, n_feature).reshape(n_feature, 1)  # a vector

    WI_item = np.eye(n_feature, dtype='float64')
    beta_item = beta_item
    df_item = int(df_item) if df_item is not None else n_feature
    mu0_item = np.repeat(mu0_item, n_feature).reshape(n_feature, 1)

    # Latent Variables
    mu_user = np.zeros((n_feature, 1), dtype='float64')
    mu_item = np.zeros((n_feature, 1), dtype='float64')

    alpha_user = np.eye(n_feature, dtype='float64')
    alpha_item = np.eye(n_feature, dtype='float64')

    # initializes the user features randomly. (There is no special reason to use 0.3)
    user_features_ = 0.3 * rand_state.rand(n_user, n_feature)
    item_features_ = 0.3 * rand_state.rand(n_item, n_feature)

    # average user/item features
    avg_user_features_ = np.zeros((n_user, n_feature))
    avg_item_features_ = np.zeros((n_item, n_feature))

    iter_ = 0

    mean_rating_ = np.mean(ratings[:, 2])

    ratings_csr_ = build_user_item_matrix(n_user, n_item, ratings)

    ratings_csc_ = ratings_csr_.tocsc()

    last_rmse = None
    for iteration in xrange(n_iters):
        start_time = time.time()
        mu_item, alpha_item = _update_item_params(n_item, item_features_, n_feature, mu0_item, WI_item, beta_item, df_item, rand_state)
        # print("_update_item_params --- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        mu_user, alpha_user = _update_user_params(n_user, user_features_, n_feature, mu0_user, WI_user, beta_user, df_user, rand_state)
        # print("_update_user_params --- %s seconds ---" % (time.time() - start_time))

        start_time = time.time()
        _udpate_item_features(n_item, ratings_csc_, user_features_, mean_rating_, alpha_item, beta, mu_item, rand_state, n_feature, item_features_)
        # print("_udpate_item_features --- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        _update_user_features(n_user, ratings_csr_, item_features_, mean_rating_, alpha_user, beta, mu_user, rand_state, n_feature, user_features_)
        # print("_update_user_features --- %s seconds ---" % (time.time() - start_time))

        start_time = time.time()
        _update_average_features(iter_, avg_user_features_, user_features_, avg_item_features_, item_features_)
        # print("_update_average_features --- %s seconds ---" % (time.time() - start_time))
        iter_ += 1

        # compute RMSE
        train_preds = predict(ratings[:, :2], mean_rating_, avg_user_features_, avg_item_features_, max_rating, min_rating)
        train_rmse = RMSE(train_preds, ratings[:, 2])
        print(f"iteration: {iter_}, train RMSE: {train_rmse}")

        if last_rmse and abs(train_rmse - last_rmse) < converge:
            print(f'converges at iteration {iter_}. stop.')
            break
        else:
            last_rmse = train_rmse
        last_rmse = train_rmse

    return user_features_, item_features_
