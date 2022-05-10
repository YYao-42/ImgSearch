import os
import os.path
import numpy as np
import pickle
import parser
from src.utils.nnsearch import *
from src.utils.evaluate import mAP_custom
from src.utils.general import load_path_features

parser = argparse.ArgumentParser(description='test: google landmark')
parser.add_argument('--ifflickr100k', '-100k', action='store_true', help='includes 100k distractors')
parser.add_argument('--matching_method', '-mm', default='L2', help="select matching methods: L2, PQ, ANNOY, HNSW, PQ_HNSW")
args = parser.parse_args()

Custom_q, relpaths_q = load_path_features('Custom/query')
Custom_d, relpaths_d = load_path_features('Custom/database')
if args.ifflickr100k:
    flickr100k, relpaths_flickr100k = load_path_features('flickr100k')
    Custom_d = np.concatenate((Custom_d, flickr100k), axis=1)
    relpaths_d = relpaths_d + relpaths_flickr100k
n_database = Custom_d.shape[1]

# K = n_database
K = 100

if args.matching_method == 'L2':
    match_idx, time_per_query = matching_L2(K, Custom_d.T, Custom_q.T)
elif args.matching_method == 'PQ':
    match_idx, time_per_query = matching_Nano_PQ(K, Custom_d.T, Custom_q.T, dataset='Custom', N_books=16, n_bits_perbook=12, ifgenerate=False)
elif args.matching_method == 'ANNOY':
    match_idx, time_per_query = matching_ANNOY(K, Custom_d.T, Custom_q.T, 'euclidean', dataset='Custom', n_trees=100, ifgenerate=True)
elif args.matching_method == 'HNSW':
    match_idx, time_per_query = matching_HNSW(K, Custom_d.T, Custom_q.T, dataset='Custom', m=8, ef=16, ifgenerate=True)
elif args.matching_method == 'PQ_HNSW':
    match_idx, time_per_query = matching_HNSW_NanoPQ(K, Custom_d.T, Custom_q.T, dataset='Custom', N_books=16, N_words=2**12, m=8, ef=16, ifgenerate=True)
    # match_idx, time_per_query = matching_HNSW_NanoPQ(K, Custom_d.T, Custom_q.T, dataset='Custom', m=16, ef=32, ifgenerate=True)
else:
    print('Invalid method')
mAP = mAP_custom(K, match_idx, relpaths_q, relpaths_d)
print('mean average precision: ', mAP)
print('matching time per query: ', time_per_query)