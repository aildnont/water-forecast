import numpy as np
import yaml
import os
import pandas as pd
from datetime import datetime
from kmodes.kprototypes import KPrototypes
from kmodes.util.dissim import euclidean_dissim, matching_dissim
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import silhouette_score
from src.data.preprocess import load_raw_data, prepare_for_clustering
from src.visualization.visualize import visualize_silhouette_plot


def cluster_clients(k=None, save_centroids=True, save_clusters=True):
    '''
    Runs k-prototypes clustering algorithm on preprocessed dataset
    :param k: Desired number of clusters
    :param save_centroids: Boolean indicating whether to save cluster centroids
    :param save_clusters: Boolean indicating whether to save client cluster assignments
    :return: A KPrototypes object that describes the best clustering of all the runs
    '''
    cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

    # Load preprocessed client data
    try:
        client_df = pd.read_csv(cfg['PATHS']['CLIENT_DATA'])
    except FileNotFoundError:
        print("No file found at " + cfg['PATHS']['CLIENT_DATA'] + ". Running preprocessing of client data.")
        raw_df = load_raw_data(cfg)
        client_df = prepare_for_clustering(cfg, raw_df,  save_df=False)
    excluded_feats = cfg['K-PROTOTYPES']['FEATS_TO_EXCLUDE']
    client_df.drop(excluded_feats, axis=1, inplace=True)   # Features we don't want to see in clustering
    client_feats_df = client_df.copy()
    client_ids = client_df.pop('CONTRACT_ACCOUNT').tolist()
    cat_feats = [f for f in cfg['DATA']['CATEGORICAL_FEATS'] if f not in excluded_feats]
    bool_feats = [f for f in cfg['DATA']['BOOLEAN_FEATS'] if f not in excluded_feats]
    ordinal_encoder = OrdinalEncoder()
    client_df[cat_feats] = ordinal_encoder.fit_transform(client_df[cat_feats])
    X = np.array(client_df)

    # Get list of categorical feature indices. Boolean feats are considered categorical for clustering
    cat_feat_idxs = [client_df.columns.get_loc(c) for c in cat_feats + bool_feats if c in client_df]
    numcl_feat_idxs = [i for i in range(len(client_df.columns)) if i not in cat_feat_idxs]

    # Normalize noncategorical features
    X_noncat = X[:, numcl_feat_idxs]
    std_scaler = StandardScaler().fit(X_noncat)
    X_noncat = std_scaler.transform(X_noncat)
    X[:, numcl_feat_idxs] = X_noncat

    # Run k-prototypes algorithm on all clients and obtain cluster assignment (range [1, K]) for each client
    if k is None:
        k = cfg['K-PROTOTYPES']['K']
    k_prototypes = KPrototypes(n_clusters=k, verbose=1, n_init=cfg['K-PROTOTYPES']['N_RUNS'],
                               n_jobs=cfg['K-PROTOTYPES']['N_JOBS'], init='Cao', num_dissim=euclidean_dissim,
                               cat_dissim=matching_dissim)
    client_clusters = k_prototypes.fit_predict(X, categorical=cat_feat_idxs)
    k_prototypes.samples = X
    k_prototypes.labels = client_clusters
    k_prototypes.dist = lambda x0, x1: \
        k_prototypes.num_dissim(np.expand_dims(x0[numcl_feat_idxs], axis=0),
                                np.expand_dims(x1[numcl_feat_idxs], axis=0)) + \
        k_prototypes.gamma * k_prototypes.cat_dissim(np.expand_dims(x0[cat_feat_idxs], axis=0),
                                                     np.expand_dims(x1[cat_feat_idxs], axis=0))
    client_clusters += 1  # Enforce that cluster labels are integer range of [1, K]
    clusters_df = pd.DataFrame({'CONTRACT_ACCOUNT': client_ids, 'Cluster Membership': client_clusters})
    clusters_df = clusters_df.merge(client_feats_df, on='CONTRACT_ACCOUNT', how='left')
    clusters_df.set_index('CONTRACT_ACCOUNT')

    # Get centroids of clusters
    cluster_centroids = np.empty((k_prototypes.cluster_centroids_[0].shape[0],
                                  k_prototypes.cluster_centroids_[0].shape[1] +
                                  k_prototypes.cluster_centroids_[1].shape[1]))
    cluster_centroids[:, numcl_feat_idxs] = k_prototypes.cluster_centroids_[0]  # Numerical features
    cluster_centroids[:, cat_feat_idxs] = k_prototypes.cluster_centroids_[1]  # Categorical features

    # Scale noncategorical features of the centroids back to original range
    centroid_noncat_feats = cluster_centroids[:, numcl_feat_idxs]
    centroid_noncat_feats = std_scaler.inverse_transform(centroid_noncat_feats)
    cluster_centroids[:, numcl_feat_idxs] = centroid_noncat_feats

    # Create a DataFrame of cluster centroids
    centroids_df = pd.DataFrame(cluster_centroids, columns=list(client_df.columns))
    for i in range(len(cat_feats)):
        ordinal_dict = {j: ordinal_encoder.categories_[i][j] for j in range(len(ordinal_encoder.categories_[i]))}
        centroids_df[cat_feats[i]] = centroids_df[cat_feats[i]].map(ordinal_dict)
    centroids_df[bool_feats] = centroids_df[bool_feats].round()
    cluster_num_series = pd.Series(np.arange(1, cluster_centroids.shape[0] + 1))
    centroids_df.insert(0, 'Cluster', cluster_num_series)

    # Get fraction of clients in each cluster
    cluster_freqs = np.bincount(client_clusters) / float(client_clusters.shape[0])
    centroids_df.insert(1, '% of Clients', cluster_freqs[1:] * 100)

    # Save centroid features and cluster assignments to spreadsheet
    if save_centroids:
        centroids_df.to_csv(cfg['PATHS']['K-PROTOTYPES_CENTROIDS'] + datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv',
                            index_label=False, index=False)
    if save_clusters:
        clusters_df.to_csv(cfg['PATHS']['K-PROTOTYPES_CLUSTERS'] + datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv',
                           index_label=False, index=False)
    return k_prototypes


def silhouette_analysis(k_min=2, k_max=20):
    '''
    Perform Silhouette Analysis to determine the optimal value for k. For each value of k, run k-prototypes and
    calculate the average Silhouette Score. The optimal k is the one that maximizes the average Silhouette Score over
    the range of k provided.
    :param k_min: Smallest k to consider
    :param k_max: Largest k to consider
    :return: Optimal value of k
    '''
    cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))
    k_range = list(np.arange(k_min, k_max + 1, 1))    # Range is [k_min, k_max]
    k_to_remove = []
    silhouette_scores = []

    # Run k-prototypes for each value of k in the specified range and calculate average Silhouette Score
    for k in k_range:
        print('Running k-prototypes with k = ' + str(k))
        try:
            clustering = cluster_clients(k=k, save_centroids=False, save_clusters=False)
        except:
            print('A clustering failed. Incrementing k and trying again.')
            k_to_remove.append(k)
            continue
        if np.unique(clustering.labels).shape[0] == 1:
            print('Only 1 cluster. Setting average Silhouette Score to 0.')
            k_to_remove.append(k) # Cannot calculate Silhouette Score if all clients associated with 1 cluster
        else:
            print('Calculating average Silhouette Score for k = ' + str(k))
            silhouette_avg = silhouette_score(clustering.samples, clustering.labels, metric=clustering.dist, sample_size=1500)
            silhouette_scores.append(silhouette_avg)
    k_range = [k for k in k_range if k not in k_to_remove]
    optimal_k = k_range[np.argmax(silhouette_scores)]   # Optimal k is that which maximizes average Silhouette Score

    # Visualize the Silhouette Plot
    visualize_silhouette_plot(k_range, silhouette_scores, optimal_k=optimal_k, save_fig=True)
    return optimal_k


if __name__ == '__main__':
    cfg = yaml.full_load(open("./config.yml", 'r'))
    if cfg['K-PROTOTYPES']['EXPERIMENT'] == 'cluster_clients':
        _ = cluster_clients(k=None, save_centroids=True, save_clusters=True)
    else:
        optimal_k = silhouette_analysis(k_min=cfg['K-PROTOTYPES']['K_MIN'], k_max=cfg['K-PROTOTYPES']['K_MAX'])