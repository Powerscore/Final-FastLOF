# -*- coding: utf-8 -*-
"""Ranged Local Outlier Factor (LOF). Implemented on scikit-learn library.
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause


import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted


# noinspection PyProtectedMember


def invert_order(scores):
    """Invert the order of outlier scores. Outliers have higher scores.
    
    Parameters
    ----------
    scores : numpy array of shape (n_samples,)
        The outlier scores to invert.
        
    Returns
    -------
    inverted_scores : numpy array of shape (n_samples,)
        The inverted outlier scores.
    """
    return -scores


class OriginalLOF(LocalOutlierFactor):
    """Wrapper of scikit-learn LOF Class with ranged neighbor functionality.
    Unsupervised Outlier Detection using Local Outlier Factor (LOF).

    The anomaly score of each sample is called Local Outlier Factor.
    It measures the local deviation of density of a given sample with
    respect to its neighbors.
    It is local in that the anomaly score depends on how isolated the object
    is with respect to the surrounding neighborhood.
    More precisely, locality is given by k-nearest neighbors, whose distance
    is used to estimate the local density.
    By comparing the local density of a sample to the local densities of
    its neighbors, one can identify samples that have a substantially lower
    density than their neighbors. These are considered outliers.
    
    This implementation extends the standard LOF by computing LOF scores
    across a range of neighborhood sizes and taking the most outlying score
    (minimum negative_outlier_factor) as recommended in the original LOF paper.
    
    See :cite:`breunig2000lof` for details.

    Parameters
    ----------
    n_neighbors : int, optional (default=20)
        Number of neighbors to use by default for `kneighbors` queries.
        If n_neighbors is larger than the number of samples provided,
        all samples will be used.

    n_neighbors_lb : int, optional (default=-1)
        Minimum number of neighbors to consider (lower bound of k-range).
        If ``-1``, it is set to ``n_neighbors`` (single-k behavior).

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use BallTree
        - 'kd_tree' will use KDTree
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, optional (default=30)
        Leaf size passed to `BallTree` or `KDTree`. This can
        affect the speed of the construction and query, as well as the memory
        required to store the tree. The optimal value depends on the
        nature of the problem.

    metric : string or callable, default 'minkowski'
        metric used for the distance computation. Any metric from scikit-learn
        or scipy.spatial.distance can be used.

        If 'precomputed', the training input X is expected to be a distance
        matrix.

        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

        Valid values for metric are:

        - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
          'manhattan']

        - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
          'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
          'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
          'sqeuclidean', 'yule']

        See the documentation for scipy.spatial.distance for details on these
        metrics:
        http://docs.scipy.org/doc/scipy/reference/spatial.distance.html

    p : integer, optional (default = 2)
        Parameter for the Minkowski metric from
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
        See http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_distances

    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. When fitting this is used to define the
        threshold on the decision function.

    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run for neighbors search.
        If ``-1``, then the number of jobs is set to the number of CPU cores.
        Affects only kneighbors and kneighbors_graph methods.

    novelty : bool (default=False)
        By default, LocalOutlierFactor is only meant to be used for outlier
        detection (novelty=False). Set novelty to True if you want to use
        LocalOutlierFactor for novelty detection. In this case be aware that
        that you should only use predict, decision_function and score_samples
        on new unseen data and not on the training set.

    Attributes
    ----------
    n_neighbors_ : int
        The actual number of neighbors used for `kneighbors` queries.

    n_neighbors_lb_ : int
        The actual minimum number of neighbors used for ranged computation.

    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.

    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    def __init__(self, n_neighbors=20, n_neighbors_lb=-1, algorithm='auto',
                 leaf_size=30, metric='minkowski', p=2, metric_params=None,
                 contamination=0.1, n_jobs=1, novelty=True):
        super(OriginalLOF, self).__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            contamination=contamination,
            n_jobs=n_jobs,
            novelty=novelty
        )
        self.n_neighbors_lb = n_neighbors_lb

    def _set_n_classes(self, y=None):
        """Set the number of classes.
        
        Parameters
        ----------
        y : Ignored
            Not used, present for API consistency by convention.
        """
        self.n_classes_ = 2

    def _process_decision_scores(self):
        """Process decision scores to determine threshold and labels.
        
        This method computes the threshold based on the contamination parameter
        and assigns binary labels to the training data.
        
        Raises
        ------
        ValueError
            If decision_scores_ is not set.
        """
        if getattr(self, "decision_scores_", None) is None:
            raise ValueError("decision_scores_ must be set before processing.")

        self.threshold_ = np.percentile(
            self.decision_scores_,
            100.0 * (1.0 - float(self.contamination)),
        )
        self.labels_ = (self.decision_scores_ > self.threshold_).astype(int)

    # noinspection PyIncorrectDocstring
    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = check_array(X)
        self._set_n_classes(y)

        self._fit(X)

        n_samples = self.n_samples_fit_
        if self.n_neighbors > n_samples:
            self.n_neighbors_ = n_samples - 1
        else:
            self.n_neighbors_ = self.n_neighbors

        if self.n_neighbors_lb == -1:
            self.n_neighbors_lb_ = self.n_neighbors_
        else:
            self.n_neighbors_lb_ = int(self.n_neighbors_lb)
            if self.n_neighbors_lb_ > self.n_neighbors_:
                self.n_neighbors_lb_ = self.n_neighbors_

        self._distances_fit_X_, self._neighbors_indices_fit_X_ = self.kneighbors(
            n_neighbors=self.n_neighbors_
        )

        if self._fit_X.dtype == np.float32:
            self._distances_fit_X_ = self._distances_fit_X_.astype(
                self._fit_X.dtype,
                copy=False,
            )

        if self.n_neighbors_lb_ < self.n_neighbors_:
            self.negative_outlier_factor_ = np.zeros(n_samples)

            for n_neighbors_ix in range(self.n_neighbors_, self.n_neighbors_lb_ - 1, -1):
                lrd = self._local_reachability_density(
                    self._distances_fit_X_[:, :n_neighbors_ix],
                    self._neighbors_indices_fit_X_[:, :n_neighbors_ix],
                    n_neighbors_ix
                )

                lrd_ratios_array = (
                    lrd[self._neighbors_indices_fit_X_[:, :n_neighbors_ix]] / lrd[:, np.newaxis]
                )
                negative_outlier_factor_tmp = -np.mean(lrd_ratios_array, axis=1)

                if n_neighbors_ix == self.n_neighbors_:
                    self.negative_outlier_factor_ = negative_outlier_factor_tmp
                else:
                    self.negative_outlier_factor_ = np.minimum(
                        self.negative_outlier_factor_,
                        negative_outlier_factor_tmp
                    )

            self._lrd = self._local_reachability_density(
                self._distances_fit_X_,
                self._neighbors_indices_fit_X_,
                self.n_neighbors_
            )
        else:
            self._lrd = self._local_reachability_density(
                self._distances_fit_X_, self._neighbors_indices_fit_X_, self.n_neighbors_
            )

            lrd_ratios_array = (
                self._lrd[self._neighbors_indices_fit_X_] / self._lrd[:, np.newaxis]
            )
            self.negative_outlier_factor_ = -np.mean(lrd_ratios_array, axis=1)

        if self.contamination == "auto":
            self.offset_ = -1.5
        else:
            self.offset_ = np.percentile(
                self.negative_outlier_factor_, 100.0 * self.contamination
            )

        self.decision_scores_ = invert_order(self.negative_outlier_factor_)
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])

        # noinspection PyProtectedMember
        try:
            return invert_order(self._score_samples(X))
        except AttributeError:
            try:
                return invert_order(self._decision_function(X))
            except AttributeError:
                return invert_order(self.score_samples(X))

    def _local_reachability_density(self, distances_X, neighbors_indices, n_neighbors_ix=None):
        """The local reachability density (LRD)

        The LRD of a sample is the inverse of the average reachability
        distance of its k-nearest neighbors.

        Parameters
        ----------
        distances_X : ndarray of shape (n_queries, n_neighbors_ix)
            Distances to the neighbors (in the training samples `self._fit_X`)
            of each query point to compute the LRD.

        neighbors_indices : ndarray of shape (n_queries, n_neighbors_ix)
            Neighbors indices (of each query point) among training samples
            self._fit_X.

        n_neighbors_ix : int, optional
            Number of neighbors being used for this computation.
            If None, uses the number of neighbors in distances_X.

        Returns
        -------
        local_reachability_density : ndarray of shape (n_queries,)
            The local reachability density of each sample.
        """
        if n_neighbors_ix is None:
            n_neighbors_ix = distances_X.shape[1]

        dist_k = self._distances_fit_X_[neighbors_indices, n_neighbors_ix - 1]
        reach_dist_array = np.maximum(distances_X, dist_k)

        return 1.0 / (np.mean(reach_dist_array, axis=1) + 1e-10)