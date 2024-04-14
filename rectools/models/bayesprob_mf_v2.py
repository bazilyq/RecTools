#  Copyright 2022 MTS (Mobile Telesystems)
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


"""BayesProbMF Model."""

import typing as tp

import numpy as np
from scipy.sparse.linalg import svds

from rectools.dataset import Dataset
from rectools.exceptions import NotFittedError
from rectools.models.vector import Distance, Factors, VectorModel
from rectools.models.bayesprob_mf_raw_v2 import BPMF_V2


class BayesProbMF_V2(VectorModel):
    """
    BayesProbMF matrix factorization model.

    See https://dl.acm.org/doi/10.1145/1864708.1864721

    Parameters
    ----------
    factors : int, default ``10``
        The number of latent factors to compute.
    verbose : int, default ``0``
        Degree of verbose output. If ``0``, no output will be provided.
    """

    u2i_dist = Distance.DOT
    i2i_dist = Distance.COSINE

    def __init__(self, factors: int = 10, verbose: int = 0, n_iters: int = 10,
                n_feature: int = 10, beta: float = 2.0, beta_user: float = 2.0,
                df_user = None, mu0_user: float = 0., beta_item: float = 2.0, df_item = None,
                mu0_item: float = 0., converge: float = 1e-5, seed = None, max_rating = None,
                min_rating = None):

        super().__init__(verbose=verbose)

        self.factors = factors
        self.user_factors: np.ndarray
        self.item_factors: np.ndarray

        self.n_iters = n_iters
        self.n_feature = n_feature
        self.beta = beta
        self.beta_user = beta_user
        self.df_user = df_user
        self.mu0_user = mu0_user
        self.beta_item = beta_item
        self.df_item = df_item
        self.mu0_item = mu0_item
        self.converge = converge
        self.seed = seed
        self.max_rating = max_rating
        self.min_rating = min_rating

    def _fit(self, dataset: Dataset) -> None:  # type: ignore
        # ui_csr = dataset.get_user_item_matrix(include_weights=True)

        ratings = dataset.interactions.df.to_numpy()[:, :3].copy()
        ratings = ratings.astype('int64')

        n_user = np.unique(ratings[:, 0]).shape[0]
        n_item = np.unique(ratings[:, 1]).shape[0]

        bpmf = BPMF_V2(n_user=n_user, n_item=n_item, 
                       n_feature = self.n_feature, beta = self.beta,
                       beta_user = self.beta_user, df_user = self.df_user, 
                       mu0_user = self.mu0_user, beta_item = self.beta_item,
                       df_item = self.df_item, mu0_item = self.mu0_item,
                       converge = self.converge, seed = self.seed, 
                       max_rating = self.max_rating, min_rating = self.min_rating
                       )

        u, v = bpmf.fit(ratings, n_iters=self.n_iters)

        self.user_factors = u
        self.item_factors = v

    def _get_users_factors(self, dataset: Dataset) -> Factors:
        return Factors(self.user_factors)

    def _get_items_factors(self, dataset: Dataset) -> Factors:
        return Factors(self.item_factors)

    def get_vectors(self) -> tp.Tuple[np.ndarray, np.ndarray]:
        """
        Return user and item vector representations from fitted model.

        Returns
        -------
        (np.ndarray, np.ndarray)
            User and item embeddings.
            Shapes are (n_users, n_factors) and (n_items, n_factors).
        """
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)
        return self.user_factors, self.item_factors
