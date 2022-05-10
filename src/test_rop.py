import numpy as np
import parser
from src.utils.nnsearch import *
from src.datasets.testdataset import configdataset
from src.utils.evaluate import compute_map_and_print
from src.utils.general import get_data_root, load_path_features, visualization_umap

parser = argparse.ArgumentParser(description='test: google landmark')
parser.add_argument('--ifflickr100k', '-100k', action='store_true', help='includes 100k distractors')
parser.add_argument('--matching_method', '-mm', default='L2', help="select matching methods: L2, PQ, ANNOY, HNSW, PQ_HNSW")
args = parser.parse_args()

datasets = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k']
# datasets = ['oxford5k', 'paris6k']

for dataset in datasets:
    vecs, _ = load_path_features(dataset + '_database')
    qvecs, _ = load_path_features(dataset + '_query')
    if args.ifflickr100k:
        flickr100k, _ = load_path_features('flickr100k')
        vecs = np.concatenate((vecs, flickr100k), axis=1)
    # search, rank, and print
    visualization_umap(vecs, dataset, n_components=2)
    n_database = vecs.shape[1]

    # To report mAP, use K = n_database; To compare the matching time, use K = 100
    # K = n_database
    K = 100

    if args.matching_method == 'L2':
        match_idx, time_per_query = matching_L2(K, vecs.T, qvecs.T)
    elif args.matching_method == 'PQ':
        match_idx, time_per_query = matching_Nano_PQ(K, vecs.T, qvecs.T, dataset, ifgenerate=True)
    elif args.matching_method == 'ANNOY':
        match_idx, time_per_query = matching_ANNOY(K, vecs.T, qvecs.T, 'euclidean', dataset, ifgenerate=True)
    elif args.matching_method == 'HNSW':
        match_idx, time_per_query = matching_HNSW(K, vecs.T, qvecs.T, dataset, ifgenerate=True)
    elif args.matching_method == 'PQ_HNSW':
        match_idx, time_per_query = matching_HNSW_NanoPQ(K, vecs.T, qvecs.T, dataset, ifgenerate=True)
    else:
        print('Invalid method')

    print('matching time per query: ', time_per_query)
    ranks = match_idx.T
    cfg = configdataset(dataset, os.path.join(get_data_root(), 'test'))
    compute_map_and_print(dataset, ranks, cfg['gnd'])
