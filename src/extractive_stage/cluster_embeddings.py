import numpy as np
from generate_embeddings import load_example
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min
from sklearn.preprocessing import normalize
import pandas as pd
import umap
import hdbscan

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def embed_cluster(docs, query, embeddings, model, dim_reduction_model, n_tokens=None, n_documents=None):
    print('\n\n')
    print(query)
    print(len(docs))
    clf = NearestCentroid(metric='cosine')
    input_embeddings = embeddings[0]
    output_embeddings = embeddings[1]
    title_embedding = np.array([output_embeddings[0].tolist()])
    umap_input_embeddings = normalize(dim_reduction_model.fit_transform(input_embeddings)) # reduce dim of input texts and normalize
    umap_title_embeddings = normalize(dim_reduction_model.fit_transform(title_embedding)) # reduce dim of title and normalize
    model.fit(umap_input_embeddings) # cluster with DBSCAN
    print(model.labels_)
    point_clusters, clusters_sizes = np.unique(model.labels_, return_counts=True)
    print(clusters_sizes)
    centroids = None
    if(len(point_clusters) > 1):
        clf.fit(umap_input_embeddings, model.labels_) # get clusters centroids
        centroids = clf.centroids_[1:] # disconsider outliers label (first index)
    else:
        centroids = np.array([np.mean(umap_input_embeddings, axis=0)])
    print(centroids.shape)
    #print(centroids)
    argmin_sent_centroid, distances_sent_centroid = pairwise_distances_argmin_min(centroids, umap_input_embeddings, metric='euclidean') # get index and distance of sentence closer to centroid
    print(argmin_sent_centroid)
    #to_out_ind = sorted(range(len(point_clusters[1:])), key=lambda i: point_clusters[i], reverse=True) # sort distances
    centroid_to_title_distances = pairwise_distances(umap_title_embeddings, centroids, metric='euclidean') # get distances of centroids that are closer to the title embedding
    print(centroid_to_title_distances)
    to_out_ind = sorted(range(len(centroid_to_title_distances[0])), key=lambda i: centroid_to_title_distances[0][i], reverse=False) # sort distances
    # return extractive summary
    to_out = []
    if(n_tokens != None):
        n = 0
        for ind in to_out_ind:
            n = n + len(docs[ind].split(' '))
            if(n > n_tokens):
                break
            to_out.append(docs[ind])
    elif(n_documents != None):
        for ind in to_out_ind[:n_documents]:
            to_out.append(docs[ind])
    print(to_out_ind)
    print(to_out)
    return to_out

def dataset_embed_cluster(input_file_path, target_file_path, output_file_path, model, dim_reduction_model, n_tokens=None, n_documents=None): 
    with open(embeddings_input_file_path, 'rb') as embeddings_input_file:
        with open(embeddings_target_file_path, 'rb') as embeddings_target_file:
            with open(output_file_path, 'wb') as output_file:
                for sample in load_example(input_file_path, target_file_path):
                    # load pre-generated embeddings from files 
                    input_embeddings = np.load(embeddings_input_file)
                    output_embeddings = np.load(embeddings_target_file)
                    extractive_summary = embed_cluster(sample['input_sentences'], sample['title'], [input_embeddings, output_embeddings], model, dim_reduction_model, n_tokens=n_tokens, n_documents=n_documents)
                    extractive_summary_str = ""
                    for sent in extractive_summary:
                        extractive_summary_str = extractive_summary_str + sent.replace('\n', '') + ' </s>'
                    output_file.write('{}\n'.format(extractive_summary_str).encode('utf-8'))

if __name__ == '__main__':
    input_file_path = "../../data/wikisum_ptbr/train_test_split/input_train.csv"
    target_file_path = "../../data/wikisum_ptbr/train_test_split/output_train.csv"
    embeddings_input_file_path = "../../data/extractive_stage/cluster_embeddings/inputs_bert-base-portuguese-cased_train.npy"
    embeddings_target_file_path = "../../data/extractive_stage/cluster_embeddings/outputs_bert-base-portuguese-cased_train.npy"
    output_file_path = "../../data/extractive_stage/cluster_embeddings/input_train.csv.embed_cluster"
    eps = 0.1
    min_samples = 3
    #cluster_model = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    cluster_model = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='leaf')
    dim_reduction_model = umap.UMAP(n_neighbors=5, n_components=5, metric='euclidean')
    dataset_embed_cluster(input_file_path, target_file_path, output_file_path, cluster_model, dim_reduction_model, n_tokens=500, n_documents=None)