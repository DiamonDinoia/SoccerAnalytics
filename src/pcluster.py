# /usr/local/bin/python
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import gaussian_kde
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, silhouette_samples


class PClustering(BaseEstimator, ClusterMixin):
    """Performance clustering

    Parameters
    ----------
    k_range: tuple (pair)
        the minimum and the maximum $k$ to try when choosing the best value of $k$
        (the one having the best silhouette score)

    border_threshold: float
        the threshold to use for selecting the borderline.
        When method="gaussian" it indicates the percentile of the distribution.
        When method="silhouette" it indicates the max silhouette for a borderline point.

    method: str
        the method to use when selecting and clustering the borderline performances
        options: "gaussian", "silhouette"
        default: "gaussian"

    verbose: boolean
        verbosity mode.
        default: False

    random_state : int
        RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.


    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers
    n_clusters_: int
        number of clusters found by the algorithm
    labels_ :
        Labels of each point
    k_range: tuple
        minimum and maximum number of clusters to try
    verbose: boolean
        whether or not to show details of the execution
    random_state: int
        RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by 'np.random'.
    kmeans: scikit-learn KMeans object
    """

    def __init__(self, k_range=(2, 15), border_threshold=1, method='gaussian',
                 verbose=False,
                 random_state=42):
        self.k_range = k_range
        self.border_threshold = border_threshold
        self.method = method
        self.verbose = verbose

        # initialize attributes
        self.labels_ = []
        self.random_state = random_state

    def _find_clusters(self, X):
        if self.verbose:
            print
            'fitting kmeans...\n'
            print
            'n_clust\t|silhouette'
            print
            '---------------------'

        kmin, kmax = self.k_range
        range_n_clusters = range(kmin, kmax + 1)
        best_silhouette, best_k = 0, 0

        for k in range_n_clusters:

            # computation
            kmeans = MiniBatchKMeans(n_clusters=k, n_init=100,
                                     random_state=self.random_state)
            kmeans.fit(X)
            cluster_labels = kmeans.labels_

            silhouette = silhouette_score(X, cluster_labels)
            if self.verbose:
                print
                '%s\t|%s' % (k, round(silhouette, 2))

            if silhouette >= best_silhouette:
                best_silhouette = silhouette
                best_k = k

        kmeans = MiniBatchKMeans(n_clusters=best_k, n_init=100,
                                 random_state=self.random_state)
        kmeans.fit(X)
        self.kmeans_ = kmeans
        self.n_clusters_ = best_k
        self.cluster_centers_ = kmeans.cluster_centers_

        if self.verbose:
            print
            'Best: n_clust=%s (silhouette=%s)' % (best_k, round(best_silhouette, 2))

    def _estimate_gaussians(self, X):
        gaussians, percentiles = [], []
        thresholds = []

        for cluster_label in range(self.n_clusters_):
            data = X[self.kmeans_.labels_ == cluster_label]
            values = np.vstack(data.T)
            gaussian = gaussian_kde(values)
            gaussians.append(gaussian)

            probs = gaussian.evaluate(data.T)
            perc = np.percentile(probs, self.border_threshold)
            thresholds.append(perc)

        self.gaussians = gaussians
        self.thresholds = thresholds

    def _cluster_borderline_gaussian(self, X):
        self.labels_ = [[] for i in range(len(X))]

        # compute the gaussian associated with each cluster
        self._estimate_gaussians(X)

        cluster_labels = self.kmeans_.labels_
        for cluster_label in range(self.n_clusters_):
            gaussian = self.gaussians[cluster_label]
            threshold = self.thresholds[cluster_label]
            for i, (row, label) in enumerate(zip(X, cluster_labels)):
                if cluster_label == label:
                    self.labels_[i].append(cluster_label)
                else:
                    proba = gaussian.evaluate([[row[0]], [row[1]]])
                    if proba >= threshold:
                        self.labels_[i].append(cluster_label)

    def _cluster_borderline_silhouette(self, X):
        self.labels_ = [[] for i in range(len(X))]

        threshold = self.border_threshold
        cluster_labels = self.kmeans_.labels_
        ss = silhouette_samples(X, self.kmeans_.labels_)
        for i, (row, silhouette, cluster_label) in enumerate(zip(X, ss, cluster_labels)):
            if silhouette >= threshold:
                self.labels_[i].append(cluster_label)
            else:
                # intra_silhouette = np.mean(map(lambda x: euclidean(row, x),
                #                   [x for j, x in enumerate(X) if cluster_labels[j] == cluster_label]))
                intra_silhouette = euclidean(row, self.kmeans_.cluster_centers_[cluster_label])
                for label in set(cluster_labels):
                    # inter_silhouette = np.mean(map(lambda x: euclidean(row, x),
                    #                           [x for j, x in enumerate(X) if cluster_labels[j] == label]))
                    inter_silhouette = euclidean(row, self.kmeans_.cluster_centers_[label])
                    silhouette = (inter_silhouette - intra_silhouette) / max(inter_silhouette, intra_silhouette)
                    if silhouette <= threshold:
                        self.labels_[i].append(label)

    def _cluster_borderline(self, X):
        """
        Assign clusters to borderline points, according to the method and the borderline_threshold
        specified in the constructor
        """
        if self.method == 'gaussian':
            self._cluster_borderline_gaussian(X)
        else:
            self._cluster_borderline_silhouette(X)

    def _generate_matrix(self):
        """
        Generate a matrix for optimizing the predict function
        """
        matrix = {}
        X = []

        for i in range(0, 101):
            for j in range(0, 101):
                X.append([i, j])

        if self.method == 'gaussian':
            multi_labels = self._predict_with_gaussian(X)
        else:
            multi_labels = self._predict_with_silhouette(X)
        for row, labels in zip(X, multi_labels):
            matrix[tuple(row)] = labels
        self._matrix = matrix

    def fit(self, X, y=None):
        """
        Compute performance clustering.

        Parameters
        ----------
            X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster.

            y: ignored
        """
        self._find_clusters(X)  # find the clusters with kmeans
        self._cluster_borderline(X)  # assign multiclusters to borderline performances
        self._generate_matrix()  # generate the matrix for optimizing the predict function
        return self

    def _predict_with_gaussian(self, X):
        cluster_labels = self.kmeans_.predict(X)
        multicluster_labels = [[] for cluster_label in cluster_labels]
        for cluster_label in range(self.n_clusters_):
            gaussian = self.gaussians[cluster_label]
            threshold = self.thresholds[cluster_label]

            for i, (row, label) in enumerate(zip(X, cluster_labels)):
                if cluster_label == label:
                    multicluster_labels[i].append(cluster_label)
                else:
                    proba = gaussian.evaluate([[row[0]], [row[1]]])
                    if proba >= threshold:
                        multicluster_labels[i].append(cluster_label)

        return np.array(multicluster_labels)

    def _predict_with_silhouette(self, X):
        cluster_labels, threshold = self.kmeans_.predict(X), self.border_threshold
        multicluster_labels = [[] for _ in cluster_labels]
        if len(set(cluster_labels)) == 1:
            return [[cluster_label] for cluster_label in cluster_labels]
        ss = silhouette_samples(X, cluster_labels)
        for i, (row, silhouette, cluster_label) in enumerate(zip(X, ss, cluster_labels)):
            if silhouette >= threshold:
                multicluster_labels[i].append(cluster_label)
            else:
                intra_silhouette = euclidean(row, self.cluster_centers_[cluster_label])
                for label in set(cluster_labels):
                    inter_silhouette = euclidean(row, self.cluster_centers_[label])
                    silhouette = (inter_silhouette - intra_silhouette) / max(inter_silhouette, intra_silhouette)
                    if silhouette <= threshold:
                        multicluster_labels[i].append(label)

        return np.array(multicluster_labels)

    def predict(self, X, y=None):
        """
        Predict the closest cluster each sample in X belongs to.
        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------
        multi_labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        multi_labels = []
        for row in X:
            x, y = tuple(row)
            labels = self._matrix[(int(x), int(y))]
            multi_labels.append(labels)
        return multi_labels

    def predict_proba(self, X):
        X_t = X.T
        probas = []
        for cluster_label in range(self.n_clusters_):
            gaussian = self.gaussians[cluster_label]
            probas.append(gaussian.evaluate(X_t))

        print
        np.array(probas).T

    def player_entropy(self, X):
        multi_clusters = pc.predict(X)
        clusters_probs = [0.0 for cl in range(pc.n_clusters_)]
        s = len(multi_clusters)
        for multi_cluster in multi_clusters:
            for cluster in multi_cluster:
                clusters_probs[cluster] += 1.0

        probs = np.array(clusters_probs) / s
        entropy = 0.0
        for proba in probs:
            if proba != 0.0:
                entropy += proba * np.log2(proba)
        entropy = -entropy / np.log2(pc.n_clusters_)
        return entropy
