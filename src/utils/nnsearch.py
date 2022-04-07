"""
Image Search Engine for Historical Research: A Prototype
This file contains classes/functions related to data compression and nearest neighbor search
"""

import time
import argparse
import pickle
import torch as T
import numpy as np
import numba
import numba.cuda
from numba import jit

import torchvision
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
from sklearn.cluster import KMeans
from numpy import linalg as LA
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
import faiss

import nanopq
import annoy
# import nmslib
# from extractor import *
import pprint
import sys
from heapq import heapify, heappop, heappush, heapreplace, nlargest, nsmallest
from math import log2
from operator import itemgetter
from random import random
from progressbar import *

def squared_distances(x, y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    return T.sum(diff * diff, -1)


# @jit(nopython=True, parallel=True)
def fractional_distance(x, y, p=0.5):
    '''
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the fractional distance between x[i,:] and y[j,:]
    '''
    x = np.expand_dims(x, axis=1)
    y = np.expand_dims(y, axis=0)
    diff = np.abs(x - y)
    diff_fraction = diff ** p
    return np.sum(diff_fraction, axis=-1) ** (1/p)


class HNSW(object):
    # self._graphs[level][i] contains a {j: dist} dictionary,
    # where j is a neighbor of i and dist is distance

    def l2_distance(self, a, b):
        dist = np.linalg.norm(a - b)
        return dist

    def cosine_distance(self, a, b):
        try:
            return np.dot(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))
        except ValueError:
            print(a)
            print(b)

    def l2_distance_PQ_symmetric(self, encoded_a, encoded_b):
        # if (encoded_a == encoded_b).all():
        #     dist = 0
        # else:
        #     idx_diff = list(np.where(encoded_a != encoded_b)[0])
        #     dist = 0
        #     # for i in range(N_books):
        #     for i in idx_diff:
        #         sub_a = self.c[i][encoded_a[i]].numpy()
        #         sub_b = self.c[i][encoded_b[i]].numpy()
        #         dist = dist + np.sum((sub_a-sub_b)**2)

        # This implementation is much faster
        filter = encoded_a != encoded_b
        idx_diff = list(np.where(filter)[0])
        encoded_a = encoded_a[filter].astype('int64')
        encoded_b = encoded_b[filter].astype('int64')
        a = self.reshaped_C[encoded_a, idx_diff, :]
        b = self.reshaped_C[encoded_b, idx_diff, :]
        dist = np.sum((a-b)**2)
        return dist

    def l2_distance_PQ_asymmetric(self, encoded_x, dist_table):
        _, N_books = dist_table.shape
        dist = np.sum(dist_table[encoded_x, range(N_books)])
        return dist

    def construct_dist_table(self, query, N_books):
        '''
        Inputs:
        query: query vector
        N_books: number of the sub-codebooks
        Outputs:
        dist_table: N_words * N_books
        '''
        reshaped_q = np.reshape(query, (1, N_books, -1))
        dist_table = np.sum((self.reshaped_C-reshaped_q)**2, axis=2)
        return dist_table


    def _distance(self, x, y):
        return self.distance_func(x, [y])[0]

    def vectorized_distance_(self, x, ys):
        # pprint.pprint([self.distance_func(x, y) for y in ys])
        return [self.distance_func(x, y) for y in ys]

    def vectorized_distance_PQ_(self, xs, dist_table):
        return [self.l2_distance_PQ_asymmetric(x, dist_table) for x in xs]

    def __init__(self, distance_type, m=5, ef=200, m0=None, Codewords=None, N_books=None, heuristic=True, vectorized=False):
        self.data = []
        Codewords = np.array(Codewords)
        self.Codewords = Codewords

        if (self.Codewords).any() == None:
            if distance_type == "l2":
                # l2 distance
                distance_func = self.l2_distance
            elif distance_type == "cosine":
                # cosine distance
                distance_func = self.cosine_distance
            else:
                raise TypeError('Please check your distance type!')
        else:
            distance_func = self.l2_distance_PQ_symmetric
            _, dim = Codewords.shape
            L_word = int(dim / N_books)
            self.reshaped_C = np.reshape(Codewords, (-1, N_books, L_word))
            Codewords = T.from_numpy(Codewords)
            self.c = T.split(Codewords, L_word, 1)

        self.distance_func = distance_func

        if vectorized:
            self.distance = self._distance
            self.vectorized_distance = distance_func
        else:
            self.distance = distance_func
            self.vectorized_distance = self.vectorized_distance_

        self._m = m     # number of established connections 5~48
        self._ef = ef   # size of the dynamic candidate list efConstruction
        self._m0 = 2 * m if m0 is None else m0  # maximum number of connections for each element
        self._level_mult = 1 / log2(m)  # normalization factor for level generation
        self._graphs = []
        self._enter_point = None

        self._select = (
            self._select_heuristic if heuristic else self._select_naive)

    def add(self, elem, ef=None):

        if ef is None:
            ef = self._ef

        distance = self.distance
        data = self.data
        graphs = self._graphs
        point = self._enter_point
        m = self._m

        # level at which the element will be inserted
        level = int(-log2(random()) * self._level_mult) + 1
        # print("level: %d" % level)

        # elem will be at data[idx]
        idx = len(data)
        data.append(elem)

        if point is not None:  # the HNSW is not empty, we have an entry point
            dist = distance(elem, data[point])
            # for all levels in which we dont have to insert elem,
            # we search for the closest neighbor
            for layer in reversed(graphs[level:]):
                point, dist = self._search_graph_ef1(elem, point, dist, layer)
            # at these levels we have to insert elem; ep is a heap of entry points.
            ep = [(-dist, point)]
            # pprint.pprint(ep)
            layer0 = graphs[0]
            for layer in reversed(graphs[:level]):
                level_m = m if layer is not layer0 else self._m0
                # navigate the graph and update ep with the closest
                # nodes we find
                ep = self._search_graph(elem, ep, layer, ef)
                # insert in g[idx] the best neighbors
                layer[idx] = layer_idx = {}
                self._select(layer_idx, ep, level_m, layer, heap=True)
                # assert len(layer_idx) <= level_m
                # insert backlinks to the new node
                for j, dist in layer_idx.items():
                    self._select(layer[j], (idx, dist), level_m, layer)
                    # assert len(g[j]) <= level_m
                # assert all(e in g for _, e in ep)
        for i in range(len(graphs), level):
            # for all new levels, we create an empty graph
            graphs.append({idx: {}})
            self._enter_point = idx

    def balanced_add(self, elem, ef=None):
        if ef is None:
            ef = self._ef

        distance = self.distance
        data = self.data
        graphs = self._graphs
        point = self._enter_point
        m = self._m
        m0 = self._m0

        idx = len(data)
        data.append(elem)

        if point is not None:
            dist = distance(elem, data[point])
            pd = [(point, dist)]
            # pprint.pprint(len(graphs))
            for layer in reversed(graphs[1:]):
                point, dist = self._search_graph_ef1(elem, point, dist, layer)
                pd.append((point, dist))
            for level, layer in enumerate(graphs):
                # print('\n')
                # pprint.pprint(layer)
                level_m = m0 if level == 0 else m
                candidates = self._search_graph(
                    elem, [(-dist, point)], layer, ef)
                layer[idx] = layer_idx = {}
                self._select(layer_idx, candidates, level_m, layer, heap=True)
                # add reverse edges
                for j, dist in layer_idx.items():
                    self._select(layer[j], [idx, dist], level_m, layer)
                    assert len(layer[j]) <= level_m
                if len(layer_idx) < level_m:
                    return
                if level < len(graphs) - 1:
                    if any(p in graphs[level + 1] for p in layer_idx):
                        return
                point, dist = pd.pop()
        graphs.append({idx: {}})
        self._enter_point = idx

    def search(self, q, k=None, ef=None):
        """Find the k points closest to q."""
        if (self.Codewords).any() == None:
            distance = self.distance
        else:
            distance = self.l2_distance_PQ_asymmetric
        graphs = self._graphs
        point = self._enter_point

        if ef is None:
            ef = self._ef

        if point is None:
            raise ValueError("Empty graph")
        if (self.Codewords).any() == None:
            dist = distance(q, self.data[point])
        else:
            encoded_x = self.data[point]
            dist_table = self.construct_dist_table(q, len(encoded_x))
            dist = distance(encoded_x, dist_table)
        # look for the closest neighbor from the top to the 2nd level
        if (self.Codewords).any() == None:
            for layer in reversed(graphs[1:]):
                point, dist = self._search_graph_ef1(q, point, dist, layer)
            # look for ef neighbors in the bottom level
            ep = self._search_graph(q, [(-dist, point)], graphs[0], ef)
        else:
            for layer in reversed(graphs[1:]):
                point, dist = self._search_graph_ef1_PQ(q, point, dist, layer, dist_table)
            # look for ef neighbors in the bottom level
            ep = self._search_graph_PQ(q, [(-dist, point)], graphs[0], ef, dist_table)

        if k is not None:
            ep = nlargest(k, ep)
        else:
            ep.sort(reverse=True)

        return [(idx, -md) for md, idx in ep]

    def _search_graph_ef1(self, q, entry, dist, layer):
        """Equivalent to _search_graph when ef=1."""

        vectorized_distance = self.vectorized_distance
        data = self.data

        best = entry
        best_dist = dist
        candidates = [(dist, entry)]
        visited = set([entry])

        while candidates:
            dist, c = heappop(candidates)
            if dist > best_dist:
                break
            edges = [e for e in layer[c] if e not in visited]
            visited.update(edges)
            dists = vectorized_distance(q, [data[e] for e in edges])
            for e, dist in zip(edges, dists):
                if dist < best_dist:
                    best = e
                    best_dist = dist
                    heappush(candidates, (dist, e))
                    # break

        return best, best_dist

    def _search_graph(self, q, ep, layer, ef):

        vectorized_distance = self.vectorized_distance
        data = self.data

        candidates = [(-mdist, p) for mdist, p in ep]
        heapify(candidates)
        visited = set(p for _, p in ep)

        while candidates:
            dist, c = heappop(candidates)
            mref = ep[0][0]
            if dist > -mref:
                break
            # pprint.pprint(layer[c])
            edges = [e for e in layer[c] if e not in visited]
            visited.update(edges)
            dists = vectorized_distance(q, [data[e] for e in edges])
            for e, dist in zip(edges, dists):
                mdist = -dist
                if len(ep) < ef:
                    heappush(candidates, (dist, e))
                    heappush(ep, (mdist, e))
                    mref = ep[0][0]
                elif mdist > mref:
                    heappush(candidates, (dist, e))
                    heapreplace(ep, (mdist, e))
                    mref = ep[0][0]

        return ep

    def _search_graph_ef1_PQ(self, q, entry, dist, layer, dist_table):
        """Equivalent to _search_graph when ef=1."""

        data = self.data

        best = entry
        best_dist = dist
        candidates = [(dist, entry)]
        visited = set([entry])

        while candidates:
            dist, c = heappop(candidates)
            if dist > best_dist:
                break
            edges = [e for e in layer[c] if e not in visited]
            visited.update(edges)
            dists = self.vectorized_distance_PQ_([data[e] for e in edges], dist_table)
            for e, dist in zip(edges, dists):
                if dist < best_dist:
                    best = e
                    best_dist = dist
                    heappush(candidates, (dist, e))
                    # break

        return best, best_dist

    def _search_graph_PQ(self, q, ep, layer, ef, dist_table):

        data = self.data

        candidates = [(-mdist, p) for mdist, p in ep]
        heapify(candidates)
        visited = set(p for _, p in ep)

        while candidates:
            dist, c = heappop(candidates)
            mref = ep[0][0]
            if dist > -mref:
                break
            # pprint.pprint(layer[c])
            edges = [e for e in layer[c] if e not in visited]
            visited.update(edges)
            dists = self.vectorized_distance_PQ_([data[e] for e in edges], dist_table)
            for e, dist in zip(edges, dists):
                mdist = -dist
                if len(ep) < ef:
                    heappush(candidates, (dist, e))
                    heappush(ep, (mdist, e))
                    mref = ep[0][0]
                elif mdist > mref:
                    heappush(candidates, (dist, e))
                    heapreplace(ep, (mdist, e))
                    mref = ep[0][0]

        return ep

    def _select_naive(self, d, to_insert, m, layer, heap=False):

        if not heap:
            idx, dist = to_insert
            assert idx not in d
            if len(d) < m:
                d[idx] = dist
            else:
                max_idx, max_dist = max(d.items(), key=itemgetter(1))
                if dist < max_dist:
                    del d[max_idx]
                    d[idx] = dist
            return

        assert not any(idx in d for _, idx in to_insert)
        to_insert = nlargest(m, to_insert)  # smallest m distances
        unchecked = m - len(d)
        assert 0 <= unchecked <= m
        to_insert, checked_ins = to_insert[:unchecked], to_insert[unchecked:]
        to_check = len(checked_ins)
        if to_check > 0:
            checked_del = nlargest(to_check, d.items(), key=itemgetter(1))
        else:
            checked_del = []
        for md, idx in to_insert:
            d[idx] = -md
        zipped = zip(checked_ins, checked_del)
        for (md_new, idx_new), (idx_old, d_old) in zipped:
            if d_old <= -md_new:
                break
            del d[idx_old]
            d[idx_new] = -md_new
            assert len(d) == m

    def _select_heuristic(self, d, to_insert, m, g, heap=False):

        nb_dicts = [g[idx] for idx in d]

        def prioritize(idx, dist):
            return any(nd.get(idx, float('inf')) < dist for nd in nb_dicts), dist, idx

        if not heap:
            idx, dist = to_insert
            to_insert = [prioritize(idx, dist)]
        else:
            to_insert = nsmallest(m, (prioritize(idx, -mdist)
                                      for mdist, idx in to_insert))

        assert len(to_insert) > 0
        assert not any(idx in d for _, _, idx in to_insert)

        unchecked = m - len(d)
        assert 0 <= unchecked <= m
        to_insert, checked_ins = to_insert[:unchecked], to_insert[unchecked:]
        to_check = len(checked_ins)
        if to_check > 0:
            checked_del = nlargest(to_check, (prioritize(idx, dist)
                                              for idx, dist in d.items()))
        else:
            checked_del = []
        for _, dist, idx in to_insert:
            d[idx] = dist
        zipped = zip(checked_ins, checked_del)
        for (p_new, d_new, idx_new), (p_old, d_old, idx_old) in zipped:
            if (p_old, d_old) <= (p_new, d_new):
                break
            del d[idx_old]
            d[idx_new] = d_new
            assert len(d) == m

    def __getitem__(self, idx):

        for g in self._graphs:
            try:
                yield from g[idx].items()
            except KeyError:
                return


def matching_HNSW(K, embedded_features_train, embedded_features_test):
    num_train, feature_len = embedded_features_train.shape
    num_test, _ = embedded_features_test.shape
    hnsw = HNSW('l2', m=4, ef=8)
    widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA()]
    pbar = ProgressBar(widgets=widgets, maxval=num_train).start()
    # Building HNSW graph
    print("==> Building HNSW graph ...")
    for i in range(len(embedded_features_train)):
        hnsw.add(embedded_features_train[i])
        pbar.update(i + 1)
    pbar.finish()
    file_path = 'outputs/' + 'HNSW.pkl'
    # afile = open(file_path, "wb")
    # pickle.dump(hnsw, afile)

    # Searching
    # with open(file_path, 'rb') as pickle_file:
    #     hnsw = pickle.load(pickle_file)
    idx = np.zeros((num_test, K), dtype=np.int64)
    t1 = time.time()
    for row in range(num_test):
        query = embedded_features_test[row, :]
        idx_res = np.array(hnsw.search(query, K, ef=K))[:, 0].astype('int')
        if len(idx_res) < K:
            idx_miss = np.where(np.in1d(range(num_train), idx_res) == False)[0]
            idx_res = np.concatenate((idx_res, idx_miss))
        idx[row, :] = idx_res
    t2 = time.time()
    time_per_query = (t2 - t1) / num_test
    return idx, time_per_query


def matching_HNSW_PQ(K, Codewords, embedded_features_test, CW_idx):
    CW_idx_unique, reverse_idx = np.unique(CW_idx, return_inverse=True, axis=0)
    num_train, N_books = CW_idx_unique.shape
    num_test, _ = embedded_features_test.shape
    # normalization
    eftest_norm = np.linalg.norm(embedded_features_test, axis=1)
    eftest_norm = np.expand_dims(eftest_norm, axis=1)
    embedded_features_test = embedded_features_test / eftest_norm

    hnsw = HNSW('l2', m=4, ef=8, Codewords=Codewords, N_books=N_books)
    widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA()]
    pbar = ProgressBar(widgets=widgets, maxval=num_train).start()
    # Building HNSW graph
    print("==> Building HNSW graph ...")
    for i in range(num_train):
        hnsw.add(CW_idx_unique[i])
        pbar.update(i + 1)
    pbar.finish()

    file_path = 'outputs/' + 'HNSW_PQ.pkl'
    afile = open(file_path, "wb")
    pickle.dump(hnsw, afile)
    
    # Searching
    # with open(file_path, 'rb') as pickle_file:
    #     hnsw = pickle.load(pickle_file)
    idx = np.zeros((num_test, K), dtype=np.int64)
    t1 = time.time()
    for row in range(num_test):
        query = embedded_features_test[row, :]
        # K_unique = num_train
        K_unique = min(K, num_train)
        idx_unique = np.array(hnsw.search(query, K_unique, ef=K_unique))[:, 0].astype('int')
        if len(idx_unique) < K_unique:
            idx_miss = np.where(np.in1d(range(K_unique), idx_unique) == False)[0]
            idx_unique = np.concatenate((idx_unique, idx_miss))
        idx_recover = np.concatenate([np.where(reverse_idx == t) for t in idx_unique], axis=1)
        idx_recover = np.squeeze(idx_recover, axis=0)
        idx[row, :] = idx_recover[:K]
    t2 = time.time()
    time_per_query = (t2 - t1) / num_test
    return idx, time_per_query


def matching_HNSW_NanoPQ(K, embedded_features, embedded_features_test, N_books, N_words):
    # normalization
    eftrain_norm = np.linalg.norm(embedded_features, axis=1)
    eftrain_norm = np.expand_dims(eftrain_norm, axis=1)
    eftest_norm = np.linalg.norm(embedded_features_test, axis=1)
    eftest_norm = np.expand_dims(eftest_norm, axis=1)
    embedded_features = embedded_features / eftrain_norm
    embedded_features_test = embedded_features_test / eftest_norm

    pq = nanopq.PQ(M=N_books, Ks=N_words, verbose=True)
    pq.fit(vecs=embedded_features, iter=20, seed=42)
    # Save PQ object
    PQfile_path = 'outputs/' + 'NanoPQ.pkl'
    # aPQfile = open(PQfile_path, "wb")
    # pickle.dump(pq, aPQfile)
    # Load PQ object
    # with open(PQfile_path, 'rb') as pickle_file:
    #     pq = pickle.load(pickle_file)
    
    CW_idx = pq.encode(vecs=embedded_features)
    # embedded_recon = pq.decode(codes=CW_idx)
    Codewords = pq.codewords
    Codewords = np.transpose(Codewords, (1, 0, 2))
    Codewords = np.reshape(Codewords, (N_words, -1))

    CW_idx_unique, reverse_idx = np.unique(CW_idx, return_inverse=True, axis=0)
    num_train, _ = CW_idx_unique.shape
    key_list = range(num_train)
    value_list = [np.where(reverse_idx == t)[0] for t in key_list]
    dict_recover = dict(zip(key_list, value_list))
    num_test, _ = embedded_features_test.shape
    hnsw = HNSW('l2', m=4, ef=8, Codewords=Codewords, N_books=N_books)
    widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA()]
    pbar = ProgressBar(widgets=widgets, maxval=num_train).start()
    # Building HNSW graph
    print("==> Building HNSW graph ...")
    for i in range(num_train):
        hnsw.add(CW_idx_unique[i])
        pbar.update(i + 1)
    pbar.finish()

    # Save HNSW object
    file_path = 'outputs/' + 'HNSW_NanoPQ.pkl'
    # afile = open(file_path, "wb")
    # pickle.dump(hnsw, afile)

    # Load HNSW object
    # with open(file_path, 'rb') as pickle_file:
    #     hnsw = pickle.load(pickle_file)
    
    idx = np.zeros((num_test, K), dtype=np.int64)
    t1 = time.time()
    for row in range(num_test):
        query = embedded_features_test[row, :]
        # K_unique = num_train
        K_unique = min(K, num_train)
        idx_unique = np.array(hnsw.search(query, K_unique, ef=K_unique))[:, 0].astype('int')
        if len(idx_unique) < K_unique:
            idx_miss = np.where(np.in1d(range(K_unique), idx_unique) == False)[0]
            idx_unique = np.concatenate((idx_unique, idx_miss))
        idx_recover = np.concatenate([dict_recover[i] for i in idx_unique])
        idx[row, :] = idx_recover[:K]
    t2 = time.time()
    time_per_query = (t2 - t1) / num_test
    return idx, time_per_query


# @jit(nopython=True, parallel=True)
def matching_L2(K, embedded_features_train, embedded_features_test):
    t1 = time.time()
    num_train, feature_len = embedded_features_train.shape
    num_test, _ = embedded_features_test.shape
    idx = np.zeros((num_test, K), dtype=np.int64)
    # normalization
    eftrain_norm = np.linalg.norm(embedded_features_train, axis=1)
    eftrain_norm = np.expand_dims(eftrain_norm, axis=1)
    eftest_norm = np.linalg.norm(embedded_features_test, axis=1)
    eftest_norm = np.expand_dims(eftest_norm, axis=1)
    embedded_features_train = embedded_features_train / eftrain_norm
    embedded_features_test = embedded_features_test / eftest_norm
    for row in range(num_test):
        query = embedded_features_test[row, :]
        dist = np.linalg.norm(query-embedded_features_train, axis=1)
        # idx[row, :] = np.argpartition(dist, K-1)[:K]
        idx[row, :] = np.argsort(dist)[:K]
    t2 = time.time()
    time_per_query = (t2-t1)/num_test
    return idx, time_per_query


def matching_fractional_dis(K, embedded_features_train, embedded_features_test):
    t1 = time.time()
    num_train, feature_len = embedded_features_train.shape
    num_test, _ = embedded_features_test.shape
    # idx = np.zeros((num_test, K), dtype=np.int64)
    # normalization
    eftrain_norm = np.linalg.norm(embedded_features_train, axis=1)
    eftrain_norm = np.expand_dims(eftrain_norm, axis=1)
    eftest_norm = np.linalg.norm(embedded_features_test, axis=1)
    eftest_norm = np.expand_dims(eftest_norm, axis=1)
    embedded_features_train = embedded_features_train / eftrain_norm
    embedded_features_test = embedded_features_test / eftest_norm
    dist = fractional_distance(embedded_features_test, embedded_features_train, 2)
    # idx = np.argpartition(dist, K-1, axis=1)
    idx = np.argsort(dist)[:K]
    idx = idx[:, :K]
    # for row in range(num_test):
    #     query = embedded_features_test[row, :]
    #     dist = np.linalg.norm(query-embedded_features_train, axis=1)
    #     idx[row, :] = np.argpartition(dist, K-1)[:K]
    t2 = time.time()
    time_per_query = (t2-t1)/num_test
    return idx, time_per_query


def matching_LSH_faiss(K, embedded_features_train, embedded_features_test, n_bits):
    embedded_features_train = np.ascontiguousarray(embedded_features_train)
    embedded_features_test = np.ascontiguousarray(embedded_features_test)
    num_train, feature_len = embedded_features_train.shape
    num_test, _ = embedded_features_test.shape
    lsh = faiss.IndexLSH(feature_len, n_bits)
    lsh.add(embedded_features_train)
    t1 = time.time()
    _, idx = lsh.search(embedded_features_test, K)
    t2 = time.time()
    time_per_query = (t2-t1)/num_test
    return idx, time_per_query


def matching_PQ_faiss(K, embedded_features_train, embedded_features_test, N_books, n_bits_perbook):
    # t1 = time.time()
    num_train, feature_len = embedded_features_train.shape
    num_test, _ = embedded_features_test.shape
    # normalization
    eftrain_norm = np.linalg.norm(embedded_features_train, axis=1)
    eftrain_norm = np.expand_dims(eftrain_norm, axis=1)
    eftest_norm = np.linalg.norm(embedded_features_test, axis=1)
    eftest_norm = np.expand_dims(eftest_norm, axis=1)
    embedded_features_train = embedded_features_train / eftrain_norm
    embedded_features_test = embedded_features_test / eftest_norm

    res = faiss.StandardGpuResources()
    # code_size = 32
    # ncentroids = 64 # np.floor(np.sqrt(feature_len)).astype('int64')  # sqrt(l)
    # coarse_quantizer = faiss.IndexFlatL2(feature_len)
    # index = faiss.IndexIVFPQ(coarse_quantizer, feature_len, ncentroids, code_size, 8)
    # index.nprobe = 5
    index = faiss.IndexPQ(feature_len, N_books, n_bits_perbook)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.train(embedded_features_train)
    gpu_index.add(embedded_features_train)
    t1 = time.time()
    _, idx = gpu_index.search(embedded_features_test, K)
    t2 = time.time()
    time_per_query = (t2-t1)/num_test
    return idx, time_per_query


def matching_HNSW_faiss(K, embedded_features_train, embedded_features_test, M=4):
    # t1 = time.time()
    num_train, feature_len = embedded_features_train.shape
    num_test, _ = embedded_features_test.shape
    # normalization
    eftrain_norm = np.linalg.norm(embedded_features_train, axis=1)
    eftrain_norm = np.expand_dims(eftrain_norm, axis=1)
    eftest_norm = np.linalg.norm(embedded_features_test, axis=1)
    eftest_norm = np.expand_dims(eftest_norm, axis=1)
    embedded_features_train = embedded_features_train / eftrain_norm
    embedded_features_test = embedded_features_test / eftest_norm

    res = faiss.StandardGpuResources()
    index = faiss.IndexHNSWFlat(feature_len, M)
    index.verbose = True
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.train(embedded_features_train)
    gpu_index.add(embedded_features_train)
    t1 = time.time()
    _, idx = gpu_index.search(embedded_features_test, K)
    t2 = time.time()
    time_per_query = (t2-t1)/num_test
    return idx, time_per_query


def matching_PQ_HNSW_faiss(K, embedded_features_train, embedded_features_test, pq_M=4, M=4):
    # t1 = time.time()
    num_train, feature_len = embedded_features_train.shape
    num_test, _ = embedded_features_test.shape
    # normalization
    eftrain_norm = np.linalg.norm(embedded_features_train, axis=1)
    eftrain_norm = np.expand_dims(eftrain_norm, axis=1)
    eftest_norm = np.linalg.norm(embedded_features_test, axis=1)
    eftest_norm = np.expand_dims(eftest_norm, axis=1)
    embedded_features_train = embedded_features_train / eftrain_norm
    embedded_features_test = embedded_features_test / eftest_norm
    embedded_features_train=np.ascontiguousarray(embedded_features_train)
    embedded_features_test=np.ascontiguousarray(embedded_features_test)

    res = faiss.StandardGpuResources()
    index = faiss.IndexHNSWPQ(feature_len, pq_M, M)
    index.verbose = True
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.train(embedded_features_train)
    gpu_index.add(embedded_features_train)
    t1 = time.time()
    _, idx = gpu_index.search(embedded_features_test, K)
    t2 = time.time()
    time_per_query = (t2-t1)/num_test
    return idx, time_per_query

def Nano_PQ(embedded_features, N_books, N_words):
    # https://nanopq.readthedocs.io/en/latest/source/tutorial.html#basic-of-pq

    # normalization
    eftrain_norm = np.linalg.norm(embedded_features, axis=1)
    eftrain_norm = np.expand_dims(eftrain_norm, axis=1)
    embedded_features = embedded_features / eftrain_norm

    pq = nanopq.PQ(M=N_books, Ks=N_words, verbose=True)
    pq.fit(vecs=embedded_features, iter=20, seed=42)
    embedded_code = pq.encode(vecs=embedded_features)
    embedded_recon = pq.decode(codes=embedded_code)
    Codewords = pq.codewords
    Codewords = np.transpose(Codewords, (1, 0, 2))
    Codewords = np.reshape(Codewords, (N_words, -1))

    return embedded_code, Codewords, embedded_recon

def matching_Nano_PQ(K, embedded_features_train, embedded_features_test, N_books, n_bits_perbook):
    # https://nanopq.readthedocs.io/en/latest/source/tutorial.html#basic-of-pq
    N_words = 2**n_bits_perbook
    num_train, feature_len = embedded_features_train.shape
    num_test, _ = embedded_features_test.shape
    # normalization
    eftrain_norm = np.linalg.norm(embedded_features_train, axis=1)
    eftrain_norm = np.expand_dims(eftrain_norm, axis=1)
    eftest_norm = np.linalg.norm(embedded_features_test, axis=1)
    eftest_norm = np.expand_dims(eftest_norm, axis=1)
    embedded_features_train = embedded_features_train / eftrain_norm
    embedded_features_test = embedded_features_test / eftest_norm

    pq = nanopq.PQ(M=N_books, Ks=N_words, verbose=True)
    pq.fit(vecs=embedded_features_train, iter=20, seed=42)
    embedded_train_code = pq.encode(vecs=embedded_features_train)
    t1 = time.time()
    idx = np.zeros((num_test, K), dtype=np.int64)
    for row in range(num_test):
        query = embedded_features_test[row, :]
        dist = pq.dtable(query=query).adist(codes=embedded_train_code)
        # idx[row, :] = np.argpartition(dist, K-1)[:K]
        idx[row, :] = np.argsort(dist)[:K]
    t2 = time.time()
    time_per_query = (t2 - t1) / num_test
    return idx, time_per_query


# @jit
def matching_PQ_Net(K, Codewords, Query, N_books, CW_idx):
    '''

    Args:
        K: nearest K neighbors
        Codewords: N_words * (N_books * L_word)
        Query: N_query * (N_books * L_word)
        N_books: number of codebooks N_words: number of codewords per codebook L_word: length of codewords
        CW_idx: N_train_images * N_books

    Returns:
        idx: matching index
        time_per_query: matching time per query

    '''
    t1 = time.time()
    N_words, dim = Codewords.shape
    N_query, _ = Query.shape
    L_word = int(dim/N_books)
    Query = T.from_numpy(Query)
    q = T.split(Query, L_word, 1)
    Codewords = T.from_numpy(Codewords)
    c = T.split(Codewords, L_word, 1)
    # Generate a distance table: N_query * N_words * N_books
    for i in range(N_books):
        if i == 0:
            dist_table = squared_distances(q[i], c[i])
            dist_table = T.unsqueeze(dist_table, 2)
        else:
            temp = squared_distances(q[i], c[i])
            temp = T.unsqueeze(temp, 2)
            dist_table = T.cat((dist_table, temp), dim=2)
    dist_table = dist_table.cpu().detach().numpy()
    idx = np.zeros((N_query, K), dtype=np.int64)
    for i in range(N_query):
        dtable_per_query = dist_table[i, :, :]
        d_query_to_train = np.sum(dtable_per_query[CW_idx, range(N_books)], axis=1)
        idx[i, :] = np.argsort(d_query_to_train)[:K]
        # idx[i, :] = np.argpartition(d_query_to_train, K-1)[:K]
    t2 = time.time()
    time_per_query = (t2 - t1) / N_query
    return idx, time_per_query


def matching_PQ_Net_bucket(K, Codewords, Query, N_books, CW_idx, Gallery_features):
    '''

    Args:
        K: nearest K neighbors
        Codewords: N_words * (N_books * L_word)
        Query: N_query * (N_books * L_word)
        N_books: number of codebooks N_words: number of codewords per codebook L_word: length of codewords
        CW_idx: N_train_images * N_books
        Gallery_features: features of the gallery

    Returns:
        idx: matching index
        time_per_query: matching time per query

    '''
    # TODO: the cardinality of the selected bucket < K
    # TODO: select multiple candidate buckets
    kmeans = KMeans(n_clusters=10, random_state=0).fit(Gallery_features)
    N_words, dim = Codewords.shape
    N_query, _ = Query.shape
    L_word = int(dim/N_books)
    t1 = time.time()
    bucket = kmeans.predict(Query)
    Query_t = T.from_numpy(Query)
    q = T.split(Query_t, L_word, 1)
    Codewords = T.from_numpy(Codewords)
    c = T.split(Codewords, L_word, 1)
    # Generate a distance table: N_query * N_words * N_books
    for i in range(N_books):
        if i == 0:
            dist_table = squared_distances(q[i], c[i])
            dist_table = T.unsqueeze(dist_table, 2)
        else:
            temp = squared_distances(q[i], c[i])
            temp = T.unsqueeze(temp, 2)
            dist_table = T.cat((dist_table, temp), dim=2)
    dist_table = dist_table.cpu().detach().numpy()
    idx = np.zeros((N_query, K), dtype=np.int64)
    for i in range(N_query):
        bucketind = np.where(bucket[i] == kmeans.labels_)
        CW_idx_refined = CW_idx[bucketind[0], :]
        dtable_per_query = dist_table[i, :, :]
        d_query_to_train = np.sum(dtable_per_query[CW_idx_refined, range(N_books)], axis=1)
        # idxtemp = np.argpartition(d_query_to_train, K-1)[:K]
        idxtemp = np.argsort(d_query_to_train)[:K]
        idx[i, :] = bucketind[0][idxtemp]
    t2 = time.time()
    time_per_query = (t2 - t1) / N_query
    return idx, time_per_query


def matching_Greedyhash(K, hash_codes_train, hash_codes_test):
    t1 = time.time()
    num_train, code_len = hash_codes_train.shape
    num_test, _ = hash_codes_test.shape
    idx = np.zeros((num_test, K), dtype=np.int64)
    for row in range(num_test):
        query = hash_codes_test[row, :]
        dist = (query ^ hash_codes_train).sum(axis=1)
        # idx[row, :] = np.argpartition(dist, K-1)[:K]
        idx[row, :] = np.argsort(dist)[:K]
    t2 = time.time()
    time_per_query = (t2-t1)/num_test
    return idx, time_per_query


def matching_ANNOY(K, embedded_features_train, embedded_features_test, metric):
    num_train, feature_len = embedded_features_train.shape
    num_test, _ = embedded_features_test.shape
    t = annoy.AnnoyIndex(feature_len, metric)
    n_trees = 5
    for n, x in enumerate(embedded_features_train):
        t.add_item(n, x)
    t.build(n_trees)
    idx = np.zeros((num_test, K), dtype=np.int64)
    t1 = time.time()
    for i in range(num_test):
        idx[i, :] = t.get_nns_by_vector(embedded_features_test[i, :], K)
    t2 = time.time()
    time_per_query = (t2 - t1) / num_test
    return idx, time_per_query


# def matching_HNSW_NMSLIB(K, embedded_features_train, embedded_features_test):
#     # https://nmslib.github.io/nmslib/quickstart.html
#     num_train, feature_len = embedded_features_train.shape
#     num_test, _ = embedded_features_test.shape
#     # normalization
#     eftrain_norm = np.linalg.norm(embedded_features_train, axis=1)
#     eftrain_norm = np.expand_dims(eftrain_norm, axis=1)
#     eftest_norm = np.linalg.norm(embedded_features_test, axis=1)
#     eftest_norm = np.expand_dims(eftest_norm, axis=1)
#     embedded_features_train = embedded_features_train / eftrain_norm
#     embedded_features_test = embedded_features_test / eftest_norm
#     space_params = {
#         'ef': 32,
#         'M': 16,
#     }
#     index = nmslib.init(method='hnsw', space='cosinesimil', space_params=space_params)
#     # index = nmslib.init(method='hnsw', space='cosinesimil')
#     index.addDataPointBatch(embedded_features_train)
#     index.createIndex()
#     idx = np.zeros((num_test, K), dtype=np.int64)
#     t1 = time.time()
#     for i in range(num_test):
#         idx[i, :] = index.knnQuery(embedded_features_test[i, :], K)[0]
#     t2 = time.time()
#     time_per_query = (t2 - t1) / num_test
#     return idx, time_per_query


def cal_mAP(idx, labels_train, labels_test):
    num_queries, K = idx.shape
    matched = np.zeros_like(idx, dtype=np.int8)
    for i in range(num_queries):
        count = 0
        for j in range(K):
            if labels_test[i] == labels_train[idx[i, j]]:
                count += 1
                matched[i, j] = count
    # N_truth = np.max(matched, axis=1, keepdims=True)+1e-16
    AP = np.sum(matched/(np.array(range(K))+1)/K, axis=1)
    mAP = AP.mean()
    return mAP

