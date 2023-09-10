import umap
import hdbscan
import numpy as np
from functools import partial
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score

from hyperopt import fmin, tpe, hp, STATUS_OK, space_eval, Trials

def generate_clusters(embeddings,n_neighbors,n_components,min_cluster_size,min_samples=None,random_state=None):

    umap_embeddings = (umap.UMAP(n_neighbors=n_neighbors,n_components=n_components,metric='cosine',
                            random_state=random_state).fit_transform(embeddings))

    clusters = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,metric='euclidean',
                               gen_min_span_tree=True, cluster_selection_method= 'eom').fit(umap_embeddings)

    probabilities = clusters.probabilities_

    return clusters, probabilities

def score_clusters(clusters, prob_threshold, embeddings):
    clusters_labels = clusters.labels_
    label_count = len(np.unique(clusters_labels))
    total_num = len(clusters.labels_)
    cost = silhouette_score(embeddings, clusters_labels)
    """
    cost = metrics.calinski_harabasz_score(embeddings, clusters_labels)
    cost = (np.count_nonzero(clusters.probabilities_ < prob_threshold) / total_num)
    cost = silhouette_score(embeddings, clusters_labels)
"""
    return label_count, cost

def objective(params, embeddings, label_lower, label_upper):

    clusters, probabilities = generate_clusters(embeddings,n_neighbors=params['n_neighbors'],
                                 n_components=params['n_components'], min_cluster_size=params['min_cluster_size'],
                                 random_state=params['random_state'])

    label_count, cost = score_clusters(clusters, prob_threshold=0.05, embeddings=embeddings)

    # 15% penalty on the cost function if outside the desired range of groups
    if (label_count < label_lower) | (label_count > label_upper):
        penalty = 0.25
    else:
        penalty = 0

    cost = -cost
    loss = cost + penalty + 1

    return {'loss': loss, 'label_count': label_count, 'status': STATUS_OK}

def bayesian_search(embeddings, space, label_lower, label_upper,max_evals):
    trials = Trials()
    fmin_objective=partial(objective,embeddings=embeddings,label_lower=label_lower,label_upper=label_upper)

    best = fmin(fmin_objective,space=space,algo=tpe.suggest,max_evals=max_evals,trials=trials)

    best_params = space_eval(space, best)
    print(f'best:{best_params}')
    print(f"label count: {trials.best_trial['result']['label_count']}")

    best_clusters, best_probabilities = generate_clusters(embeddings,n_neighbors=best_params['n_neighbors'],
                                    n_components=best_params['n_components'],
                                    min_cluster_size=best_params['min_cluster_size'],
                                 random_state=best_params['random_state'])

    return best_params, best_clusters, trials, best_probabilities
