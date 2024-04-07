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
from rectools.models.bayesprob_mf_raw import BPMF


class BayesProbMF(VectorModel):
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

    def __init__(self, factors: int = 10, verbose: int = 0,
                T: int = 10, D: int = 10, initial_cutoff: int = 0, lowest_rating: int = 1, highest_rating: int = 100):
        super().__init__(verbose=verbose)

        self.factors = factors
        self.user_factors: np.ndarray
        self.item_factors: np.ndarray

        self.T = T
        self.D = D
        self.initial_cutoff = initial_cutoff
        self.lowest_rating = lowest_rating
        self.highest_rating = highest_rating

    def _fit(self, dataset: Dataset) -> None:  # type: ignore
        ui_csr = dataset.get_user_item_matrix(include_weights=True)

        u, v = BPMF(ui_csr, self.T, self.D, self.initial_cutoff, self.lowest_rating, self.highest_rating)

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
