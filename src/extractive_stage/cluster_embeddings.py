import numpy as np
from generate_embeddings import load_example
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min
from sklearn.preprocessing import normalize
import pandas as pd
import umap
import hdbscan
import time
from sparse_models import tfidf

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def embed_cluster(docs, query, embeddings, model, dim_reduction_model, n_tokens=None, n_documents=None):
    #print('\n\n')
    #print(query)
    #print(len(docs))
    #clf = NearestCentroid(metric='euclidean')
    input_embeddings = embeddings[0]
    output_embeddings = embeddings[1]
    title_embedding = np.array([output_embeddings[0].tolist()])
    umap_input_embeddings = dim_reduction_model.fit_transform(input_embeddings) # normalize(dim_reduction_model.fit_transform(input_embeddings)) # reduce dim of input texts and normalize
    umap_title_embeddings = dim_reduction_model.fit_transform(title_embedding) # normalize(dim_reduction_model.fit_transform(title_embedding)) # reduce dim of title and normalize
    model.fit(umap_input_embeddings) # cluster with HDBSCAN
    point_clusters, clusters_sizes = np.unique(model.labels_, return_counts=True)
    #print('labels', model.labels_)
    # sort clusters by its proximity to title embedding
    title_membership_vector = hdbscan.prediction.membership_vector(model, umap_title_embeddings)[0]
    #print(title_membership_vector)
    #print(len(title_membership_vector))
    #print(max(title_membership_vector))
    sorted_clusters_ind = sorted(range(len(title_membership_vector)), key=lambda i: title_membership_vector[i], reverse=True)
    # sort clusters by its persistence, according to de HDBSCAN method
    #sorted_clusters_ind = sorted(range(len(model.cluster_persistence_)), key=lambda i: model.cluster_persistence_[i], reverse=True)
    #print(sorted_clusters_ind)
    #print('point_clusters', point_clusters)
    #print('cluster sizes', clusters_sizes)
    #print('cluster persistence', model.cluster_persistence_)
    #print(argmin_sent_centroid)
    #to_out_ind = sorted(range(len(point_clusters[1:])), key=lambda i: point_clusters[i], reverse=True) # sort distances
    #print('sorted cluster indexes', sorted_clusters_ind)
    #print('len sorted cluster indexes', len(sorted_clusters_ind))
    #print('len labels', len(point_clusters))
    #print('n points', len(model.labels_))
    #print('len exemplars', len(model.exemplars_))
    #print(model.exemplars_)
    clusters_dict = {}
    for i, cluster_id in enumerate(model.labels_):
        if(cluster_id not in clusters_dict):
            clusters_dict[cluster_id] = []
        clusters_dict[cluster_id].append(docs[i])
    to_out = []
    n = 0
    for cluster in sorted_clusters_ind:
        #print(cluster_docs)
        highest_tfidf = tfidf(clusters_dict[cluster], query, n_documents=1)[0]
        #print(highest_tfidf)
        if(n_tokens != None):
            #print(to_out_sent_ind)
            n = n + len(highest_tfidf.split(' '))
            if(n > n_tokens):
                break
            to_out.append(highest_tfidf)
        elif(n_documents != None):
            n = n + 1
            if(n > n_documents):
                break
            to_out.append(highest_tfidf)
        '''
        #print('cluster_id', cluster)
        cluster_exemplars = model.exemplars_[cluster]
        #print('exemplars', cluster_exemplars)
        argmin_exemplar_title, distances_sent_centroid = pairwise_distances_argmin_min(umap_title_embeddings, cluster_exemplars, metric='euclidean')
        #print(argmin_exemplar_title)
        #pos = np.where(umap_input_embeddings == cluster_exemplars[argmin_exemplar_title[0]].reshape(1, -1))
        pos, dist = pairwise_distances_argmin_min(cluster_exemplars[argmin_exemplar_title[0]].reshape(1, -1), umap_input_embeddings, metric='euclidean')
        #print(pos)
        to_out_sent_ind = pos[0]
        to_out_sent_ind = None
        if (len(pos[0]) != 0):
            to_out_sent_ind = pos[0][0]
        else:
            break
        if(n_tokens != None):
            #print(to_out_sent_ind)
            n = n + len(docs[to_out_sent_ind].split(' '))
            if(n > n_tokens):
                break
            to_out.append(docs[to_out_sent_ind])
        elif(n_documents != None):
            n = n + 1
            if(n > n_documents):
                break
            to_out.append(docs[to_out_sent_ind])
        '''
    #print(to_out)
    return to_out

 
    '''
        for exemplar in cluster_exemplars:
            pos = np.where(umap_input_embeddings == exemplar)
            #print('pos result', pos)
            print(docs[pos[0][0]])
    #print(clusters_sizes)
    centroids = None
    if(len(point_clusters) > 1):
        clf.fit(umap_input_embeddings, model.labels_) # get clusters centroids
        centroids = clf.centroids_[1:] # disconsider outliers label (first index)
    else:
        centroids = np.array([np.mean(umap_input_embeddings, axis=0)])
    #print(centroids.shape)
    #print(centroids)
    argmin_sent_centroid, distances_sent_centroid = pairwise_distances_argmin_min(centroids, umap_input_embeddings, metric='euclidean') # get index and distance of sentence closer to centroid
    centroid_to_title_distances = pairwise_distances(umap_title_embeddings, centroids, metric='euclidean') # get distances of centroids that are closer to the title embedding
    #print(centroid_to_title_distances)
    dist_sorted_centroid_ind = sorted(range(len(centroid_to_title_distances[0])), key=lambda i: centroid_to_title_distances[0][i], reverse=False) # sort distances
    # return extractive summary
    to_out = []
    if(n_tokens != None):
        n = 0
        for centroid_index in dist_sorted_centroid_ind:
            to_out_sent_ind = argmin_sent_centroid[centroid_index]
            #print(to_out_sent_ind)
            n = n + len(docs[to_out_sent_ind].split(' '))
            if(n > n_tokens):
                break
            to_out.append(docs[to_out_sent_ind])
    elif(n_documents != None):
        for centroid_index in dist_sorted_centroid_ind[:n_documents]:
            to_out_sent_ind = argmin_sent_centroid[centroid_index]
            to_out.append(docs[to_out_sent_ind])
    #print(to_out)
    return to_out
    '''

def dataset_embed_cluster(input_file_path, target_file_path, output_file_path, n_tokens=None, n_documents=None): 
    i = 0
    start_time = time.time()
    with open(embeddings_input_file_path, 'rb') as embeddings_input_file:
        with open(embeddings_target_file_path, 'rb') as embeddings_target_file:
            for sample in load_example(input_file_path, target_file_path):
                # load pre-generated embeddings from files 
                input_embeddings = np.load(embeddings_input_file)
                output_embeddings = np.load(embeddings_target_file)
                # reload models
                cluster_model = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='leaf', prediction_data=True)
                dim_reduction_model = umap.UMAP(n_neighbors=5, n_components=5, metric='euclidean')
                extractive_summary = embed_cluster(sample['input_sentences'], sample['title'], [input_embeddings, output_embeddings], cluster_model, dim_reduction_model, n_tokens=n_tokens, n_documents=n_documents)
                extractive_summary_str = ""
                for sent in extractive_summary:
                    extractive_summary_str = extractive_summary_str + sent.replace('\n', '') + ' </s>'
                with open(output_file_path, 'ab+') as output_file:
                    output_file.write('{}\n'.format(extractive_summary_str).encode('utf-8'))
                if(i % 100 == 0):
                    new_time = time.time()
                    print("{}, {} - {:.1f} s".format(i, sample['title'], new_time - start_time))
                    start_time = new_time
                i = i + 1

if __name__ == '__main__':
    input_file_path = "../../data/wikisum_ptbr/train_test_split/input_test.csv"
    target_file_path = "../../data/wikisum_ptbr/train_test_split/output_test.csv"
    embeddings_input_file_path = "../../data/extractive_stage/cluster_embeddings/inputs_bert-base-portuguese-cased_test_COMPLETE.npy"
    embeddings_target_file_path = "../../data/extractive_stage/cluster_embeddings/outputs_bert-base-portuguese-cased_test_COMPLETE.npy"
    output_file_path = "../../data/extractive_stage/cluster_embeddings/input_test.csv.embed_cluster"
    eps = 0.1
    min_samples = 3
    #cluster_model = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    dataset_embed_cluster(input_file_path, target_file_path, output_file_path, n_tokens=600, n_documents=None)