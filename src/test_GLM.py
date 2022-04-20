import os.path
import pickle
import parser
from src.utils.nnsearch import *
from src.utils.evaluate import mAP_GLM
from src.utils.general import load_path_features

parser = argparse.ArgumentParser(description='test: google landmark')
parser.add_argument('--matching_method', '-mm', default='L2', help="select matching methods: L2, PQ, ANNOY, HNSW, PQ_HNSW")
args = parser.parse_args()

GLM_q, relpaths_q = load_path_features('GLM/test')
GLM_d, relpaths_d = load_path_features('GLM/index')
n_database = GLM_d.shape[1]
K = 100
if args.matching_method == 'L2':
    match_idx, time_per_query = matching_L2(K, GLM_d.T, GLM_q.T)
elif args.matching_method == 'PQ':
    match_idx, time_per_query = matching_Nano_PQ(K, GLM_d.T, GLM_q.T, 'GLM', N_books=16, n_bits_perbook=13, ifgenerate=False)
elif args.matching_method == 'ANNOY':
    match_idx, time_per_query = matching_ANNOY(K, GLM_d.T, GLM_q.T, 'euclidean', 'GLM', n_trees=100, ifgenerate=False)
elif args.matching_method == 'HNSW':
    match_idx, time_per_query = matching_HNSW(K, GLM_d.T, GLM_q.T, 'GLM', m=16, ef=100, ifgenerate=False)
elif args.matching_method == 'PQ_HNSW':
    match_idx, time_per_query = matching_HNSW_NanoPQ(K, GLM_d.T, GLM_q.T, 'GLM', N_books=16, N_words=2**13, m=16, ef=100, ifgenerate=False)
else:
    print('Invalid method')

mAP = mAP_GLM(K, match_idx, relpaths_q, relpaths_d)
print('mean average precision: ', mAP)
print('matching time per query: ', time_per_query)
