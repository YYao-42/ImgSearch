import os
import os.path
import numpy as np
import pickle
from src.utils.nnsearch import *
from src.utils.evaluate import mAP_custom

def load_path_features(dataset):
    if '/' in dataset:
        dataset = dataset.replace('/', '_')
    file_path_feature = 'outputs/' + dataset + '_path_feature.pkl'
    with open(file_path_feature, 'rb') as pickle_file:
        path_feature = pickle.load(pickle_file)
    vecs = path_feature['feature']
    img_r_path = path_feature['path']
    return vecs, img_r_path

Custom_q, relpaths_q = load_path_features('Custom/query')
Custom_d, relpaths_d = load_path_features('Custom/database')
# flickr100k, relpaths_flickr100k = load_path_features('flickr100k')
# Custom_d = np.concatenate((Custom_d, flickr100k), axis=1)
# relpaths_d = relpaths_d + relpaths_flickr100k
n_database = Custom_d.shape[1]
K = n_database
match_idx, time_per_query = matching_L2(K, Custom_d.T, Custom_q.T)
mAP = mAP_custom(K, match_idx, relpaths_q, relpaths_d)
print('mean average precision: ', mAP)
