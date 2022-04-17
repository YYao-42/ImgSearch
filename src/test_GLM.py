import os.path
import pickle
from src.utils.nnsearch import *
from src.utils.evaluate import mAP_GLM

def load_path_features(dataset):
    if '/' in dataset:
        dataset = dataset.replace('/', '_')
    file_path_feature = 'outputs/' + dataset + '_path_feature.pkl'
    with open(file_path_feature, 'rb') as pickle_file:
        path_feature = pickle.load(pickle_file)
    vecs = path_feature['feature']
    img_r_path = path_feature['path']
    return vecs, img_r_path

GLM_q, relpaths_q = load_path_features('GLM/test')
GLM_d, relpaths_d = load_path_features('GLM/index')
n_database = GLM_d.shape[1]
K = 100
# match_idx, time_per_query = matching_L2(K, GLM_d.T, GLM_q.T)
match_idx, time_per_query = matching_Nano_PQ(K, GLM_d.T, GLM_q.T, 16, 13, 'GLM', ifgenerate=True)
# match_idx, time_per_query = matching_ANNOY(K, GLM_d.T, GLM_q.T, 'euclidean', 'GLM', ifgenerate=True)
# match_idx, time_per_query = matching_HNSW(K, GLM_d.T, GLM_q.T, 'GLM', ifgenerate=True)
# match_idx, time_per_query = matching_HNSW_NanoPQ(K, GLM_d.T, GLM_q.T, 16, 256, 'GLM', ifgenerate=True)
mAP = mAP_GLM(K, match_idx, relpaths_q, relpaths_d)
print('mean average precision: ', mAP)
print('matching time per query: ', time_per_query)
