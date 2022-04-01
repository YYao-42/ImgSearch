import numpy as np
import pickle
# from src.main_retrieve import load_path_features
from src.utils.nnsearch import *
from src.datasets.testdataset import configdataset
from src.utils.evaluate import compute_map_and_print
from src.utils.general import get_data_root

def load_path_features(dataset):
    file_path_feature = 'outputs/' + dataset + '_path_feature.pkl'
    with open(file_path_feature, 'rb') as pickle_file:
        path_feature = pickle.load(pickle_file)
    vecs = path_feature['feature']
    img_r_path = path_feature['path']
    return vecs, img_r_path

datasets = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k']

for dataset in datasets:
    file_vecs = 'outputs/' + dataset + '_vecs.npy'
    file_qvecs = 'outputs/' + dataset + '_qvecs.npy'
    vecs = np.load(file_vecs)
    qvecs = np.load(file_qvecs)

    flickr100k, _ = load_path_features('flickr100k')
    vecs = np.concatenate((vecs, flickr100k), axis=1)
    # search, rank, and print
    n_database = vecs.shape[1]
    # K = n_database
    K = 100

    # match_idx, time_per_query = matching_L2(K, vecs.T, qvecs.T)
    match_idx, time_per_query = matching_Nano_PQ(K, vecs.T, qvecs.T, 16, 8)
    # match_idx, time_per_query = matching_ANNOY(K, vecs.T, qvecs.T, 'euclidean')
    # match_idx, time_per_query = matching_HNSW(K, vecs.T, qvecs.T)
    # embedded_code, Codewords, _ = Nano_PQ(vecs.T, 16, 256)
    # match_idx, time_per_query = matching_HNSW_PQ(K, Codewords, qvecs.T, embedded_code)

    print('matching time per query: ', time_per_query)
    # ranks = match_idx.T
    # cfg = configdataset(dataset, os.path.join(get_data_root(), 'test'))
    # compute_map_and_print(dataset, ranks, cfg['gnd'])
