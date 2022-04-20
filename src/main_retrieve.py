"""
Image Search Engine for Historical Research: A Prototype
Run this file to generate feature vectors of the images in the database,
and retrieve relevant images of given queries
"""

import argparse
import os
from this import d
import time
import pickle
import pdb
import pandas as pd

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from torch.utils.model_zoo import load_url
from torchvision import transforms

import torch.nn as nn
import torch.utils.data

from src.imageretrievalnet import extract_vectors_PQ, init_network, extract_vectors
from src.datasets.datahelpers import cid2filename
from src.datasets.testdataset import configdataset
from src.utils.download import download_train, download_test
from src.layers.whiten import whitenlearn, whitenapply
from src.utils.evaluate import compute_map_and_print
from src.utils.general import get_data_root, htime, path_all_jpg, save_path_feature, load_path_features
# from src.extractor import *
# from src.utils.utils import *
from src.utils.nnsearch import *

PRETRAINED = {
    'retrievalSfM120k-resnet101-gem'    : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth',
    # new networks with whitening learned end-to-end
    'rSfM120k-tl-resnet101-gem-w'       : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet101-gem-w-a155e54.pth',
    }

datasets_names = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k']
whitening_names = ['retrieval-SfM-30k', 'retrieval-SfM-120k']


# Instantiate the parser
parser = argparse.ArgumentParser(description='Historical Image Retrieval Testing')

# network setting
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--network-path', '-npath', metavar='NETWORK',
                    help="pretrained network or network path (destination where network is saved)")
group.add_argument('--network-offtheshelf', '-noff', metavar='NETWORK',
                    help="off-the-shelf network, in the format 'ARCHITECTURE-POOLING' or 'ARCHITECTURE-POOLING-{reg-lwhiten-whiten}'," +
                        " examples: 'resnet101-gem' | 'resnet101-gem-reg' | 'resnet101-gem-whiten' | 'resnet101-gem-lwhiten' | 'resnet101-gem-reg-whiten'")

# test options
parser.add_argument('--datasets', '-d', metavar='DATASETS', default='oxford5k,paris6k',
                    help="comma separated list of test datasets: " +
                        " | ".join(datasets_names) +
                        " (default: 'oxford5k,paris6k')")
parser.add_argument('--image-size', '-imsize', default=1024, type=int, metavar='N',
                    help="maximum size of longer image side used for testing (default: 1024)")
parser.add_argument('--multiscale', '-ms', metavar='MULTISCALE', default='[1]',
                    help="use multiscale vectors for testing, " +
                    " examples: '[1]' | '[1, 1/2**(1/2), 1/2]' | '[1, 2**(1/2), 1/2**(1/2)]' (default: '[1]')")
parser.add_argument('--whitening', '-w', metavar='WHITENING', default=None, choices=whitening_names,
                    help="dataset used to learn whitening for testing: " +
                        " | ".join(whitening_names) +
                        " (default: None)")

parser.add_argument('--deep-quantization', '-dq', dest='deep_quantization', action='store_true',
                    help='model with deep quantization (supervised PQ)')

# GPU ID
parser.add_argument('--gpu-id', '-g', default='0', metavar='N',
                    help="gpu id used for testing (default: '0')")

# parse the arguments
args = parser.parse_args()

def extr_selfmade_dataset(net, selfmadedataset, transform, ms, msp, Lw):
    # folder_path = os.path.join(get_data_root(), 'test', selfmadedataset)
    # local disk
    # folder_path = os.path.join('/home/yuanyuanyao/data/test', selfmadedataset) 
    # img_r_path = os.listdir(folder_path)
    # images = [os.path.join(folder_path, rel_path) for rel_path in img_r_path]
    if selfmadedataset == 'GLM/test':
        path_head = '/home/yuanyuanyao/data/test/GLM/'
        df = pd.read_csv(path_head + 'retrieval_solution_v2.1.csv', usecols= ['id','images'])
        df_filtered = df.loc[df['images'] != 'None']
        query_id = df_filtered['id'].tolist()
        images = [path_head+'test/'+id[0]+'/'+id[1]+'/'+id[2]+'/'+id+'.jpg' for id in query_id]
        img_r_path = [os.path.relpath(path, "/home/yuanyuanyao/data/") for path in images]
    else:
        folder_path = os.path.join('/home/yuanyuanyao/data/test', selfmadedataset)
        images, img_r_path = path_all_jpg(folder_path, start="/home/yuanyuanyao/data/")
    # extract database vectors
    print('>> {}: database images...'.format(selfmadedataset))
    vecs = extract_vectors(net, images, args.image_size, transform, ms=ms, msp=msp)
    # convert to numpy
    vecs = vecs.numpy()

    if Lw is not None:
        # whiten the vectors
        vecs_lw  = whitenapply(vecs, Lw['m'], Lw['P'])
        vecs = vecs_lw

    save_path_feature(selfmadedataset, vecs, img_r_path)

def main():
    # check if there are unknown datasets
    for dataset in args.datasets.split(','):
        if dataset not in datasets_names:
            raise ValueError('Unsupported or unknown dataset: {}!'.format(dataset))

    # check if test dataset are downloaded
    # and download if they are not
    download_train(get_data_root())
    download_test(get_data_root())

    # setting up the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # loading network from path
    if args.network_path is not None:

        print(">> Loading network:\n>>>> '{}'".format(args.network_path))
        if args.network_path in PRETRAINED:
            # pretrained networks (downloaded automatically)
            state = load_url(PRETRAINED[args.network_path], model_dir=os.path.join(get_data_root(), 'networks'))
        else:
            # fine-tuned network from path
            state = torch.load(args.network_path)

        # parsing net params from meta
        # architecture, pooling, mean, std required
        # the rest has default values, in case that is doesnt exist
        net_params = {}
        net_params['architecture'] = state['meta']['architecture']
        net_params['pooling'] = state['meta']['pooling']
        net_params['local_whitening'] = state['meta'].get('local_whitening', False)
        net_params['regional'] = state['meta'].get('regional', False)
        net_params['whitening'] = state['meta'].get('whitening', False)
        net_params['deep_quantization'] = state['meta'].get('deep_quantization', False)
        net_params['mean'] = state['meta']['mean']
        net_params['std'] = state['meta']['std']
        net_params['pretrained'] = False

        # load network
        cuda = torch.cuda.is_available()
        net = init_network(net_params)
        net.load_state_dict(state['state_dict'])

        # if whitening is precomputed
        if 'Lw' in state['meta']:
            net.meta['Lw'] = state['meta']['Lw']

        print(">>>> loaded network: ")
        print(net.meta_repr())

    # loading offtheshelf network
    elif args.network_offtheshelf is not None:

        # parse off-the-shelf parameters
        offtheshelf = args.network_offtheshelf.split('-')
        net_params = {}
        net_params['architecture'] = offtheshelf[0]
        net_params['pooling'] = offtheshelf[1]
        net_params['local_whitening'] = 'lwhiten' in offtheshelf[2:]
        net_params['regional'] = 'reg' in offtheshelf[2:]
        net_params['whitening'] = 'whiten' in offtheshelf[2:]
        net_params['deep_quantization'] = False
        net_params['pretrained'] = True

        # load off-the-shelf network
        print(">> Loading off-the-shelf network:\n>>>> '{}'".format(args.network_offtheshelf))
        net = init_network(net_params)
        print(">>>> loaded network: ")
        print(net.meta_repr())

    # setting up the multi-scale parameters
    ms = list(eval(args.multiscale))
    if len(ms)>1 and net.meta['pooling'] == 'gem' and not net.meta['regional'] and not net.meta['whitening'] and not net.meta['deep_quantization']:
        msp = net.pool.p.item()
        print(">> Set-up multiscale:")
        print(">>>> ms: {}".format(ms))
        print(">>>> msp: {}".format(msp))
    else:
        msp = 1

     # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    # set up the transform
    normalize = transforms.Normalize(
        mean=net.meta['mean'],
        std=net.meta['std']
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # compute whitening
    if args.whitening is not None:
        start = time.time()

        if 'Lw' in net.meta and args.whitening in net.meta['Lw']:

            print('>> {}: Whitening is precomputed, loading it...'.format(args.whitening))

            if len(ms)>1:
                Lw = net.meta['Lw'][args.whitening]['ms']
            else:
                Lw = net.meta['Lw'][args.whitening]['ss']

        else:

            # if we evaluate networks from path we should save/load whitening
            # not to compute it every time
            if args.network_path is not None:
                whiten_fn = args.network_path + '_{}_whiten'.format(args.whitening)
                if len(ms) > 1:
                    whiten_fn += '_ms'
                whiten_fn += '.pth'
            else:
                whiten_fn = None

            if whiten_fn is not None and os.path.isfile(whiten_fn):
                print('>> {}: Whitening is precomputed, loading it...'.format(args.whitening))
                Lw = torch.load(whiten_fn)

            else:
                print('>> {}: Learning whitening...'.format(args.whitening))

                # loading db
                db_root = os.path.join(get_data_root(), 'train', args.whitening)
                ims_root = os.path.join(db_root, 'ims')
                db_fn = os.path.join(db_root, '{}-whiten.pkl'.format(args.whitening))
                with open(db_fn, 'rb') as f:
                    db = pickle.load(f)
                images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]

                # extract whitening vectors
                print('>> {}: Extracting...'.format(args.whitening))
                wvecs = extract_vectors(net, images, args.image_size, transform, ms=ms, msp=msp)

                # learning whitening
                print('>> {}: Learning...'.format(args.whitening))
                wvecs = wvecs.numpy()
                m, P = whitenlearn(wvecs, db['qidxs'], db['pidxs'])
                Lw = {'m': m, 'P': P}

                # saving whitening if whiten_fn exists
                if whiten_fn is not None:
                    print('>> {}: Saving to {}...'.format(args.whitening, whiten_fn))
                    torch.save(Lw, whiten_fn)

        print('>> {}: elapsed time: {}'.format(args.whitening, htime(time.time()-start)))

    else:
        Lw = None


    #############
    # RETRIEVAL
    #############
    datasets = args.datasets.split(',')
    for dataset in datasets:
        start = time.time()
        print('>> {}: Extracting...'.format(dataset))

        # prepare config structure for the test dataset
        data_root = get_data_root()
        cfg = configdataset(dataset, os.path.join(data_root, 'test'))
        images = [cfg['im_fname'](cfg, i) for i in range(cfg['n'])]
        images_r_path = [os.path.relpath(path, data_root) for path in images]
        # images_r_path = [cfg['im_fname'](cfg, i).split('/')[-1] for i in range(cfg['n'])]
        qimages = [cfg['qim_fname'](cfg, i) for i in range(cfg['nq'])]
        qimages_r_path = [os.path.relpath(path, data_root) for path in qimages]
        # qimages_r_path = [cfg['qim_fname'](cfg, i).split('/')[-1] for i in range(cfg['nq'])]
        try:
            bbxs = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]
        except:
            bbxs = None  # for holidaysmanrot and copydays

        # extract database and query vectors
        print('>> {}: database images...'.format(dataset))
        if args.deep_quantization:
            quantized_vecs, CW_idx, Codewords, vecs = extract_vectors_PQ(net, images, args.image_size, transform)
            CW_idx = CW_idx.numpy()
            CW_idx = CW_idx.astype(int)
            Codewords = Codewords.numpy()
        else:
            vecs = extract_vectors(net, images, args.image_size, transform, ms=ms, msp=msp)
        print('>> {}: query images...'.format(dataset))
        qextract_start = time.time()
        if args.deep_quantization:
            _, _, _, qvecs = extract_vectors_PQ(net, qimages, args.image_size, transform)
        else:
            qvecs = extract_vectors(net, qimages, args.image_size, transform, bbxs=bbxs, ms=ms, msp=msp)
        qextract_end = time.time()
        qextract_per_query = (qextract_end - qextract_start)/55
        print('>> {}: Evaluating...'.format(dataset))
        # convert to numpy
        vecs = vecs.numpy()
        qvecs = qvecs.numpy()

        if Lw is not None:
            # whiten the vectors
            qextract_start_w = time.time()
            vecs_lw  = whitenapply(vecs, Lw['m'], Lw['P'])
            qvecs_lw = whitenapply(qvecs, Lw['m'], Lw['P'])
            qextract_end_w = time.time()
            qextract_per_query = (qextract_end_w - qextract_start_w)/55+qextract_per_query
            # for search & rank
            vecs = vecs_lw
            qvecs = qvecs_lw

        # Save paths and features    
        save_path_feature(dataset + '_database', vecs, images_r_path)
        save_path_feature(dataset + '_query', qvecs, qimages_r_path)
        
        if args.deep_quantization:
            # Codewords = net.C
            # Codewords = Codewords.cpu().detach().numpy()
            file_CW_idx = 'outputs/' + dataset + '_CW_idx.npy'
            file_Codewords = 'outputs/' + dataset + '_Codewords.npy'
            np.save(file_CW_idx, CW_idx)
            np.save(file_Codewords, Codewords)

        # search, rank, and print
        n_database = vecs.shape[1]
        K = n_database
        # K = 100
        if args.deep_quantization:
            CW_idx = np.load(file_CW_idx)
            Codewords = np.load(file_Codewords)
            N_books = net.N_books
            match_idx, time_per_query = matching_PQ_Net(K, Codewords, qvecs.T, N_books, CW_idx.T)
        else:
            match_idx, time_per_query = matching_L2(K, vecs.T, qvecs.T)
            # match_idx, time_per_query = matching_Nano_PQ(K, vecs.T, qvecs.T, dataset, 16, 12, ifgenerate=False)
            # match_idx, time_per_query = matching_ANNOY(K, vecs.T, qvecs.T, 'euclidean', dataset, ifgenerate=False)
            # match_idx, time_per_query = matching_HNSW(K, vecs.T, qvecs.T, dataset, ifgenerate=False)
            # match_idx, time_per_query = matching_HNSW_NanoPQ(K, vecs.T, qvecs.T, dataset, ifgenerate=False)
        print('matching time per query: ', time_per_query)
        ranks = match_idx.T
        compute_map_and_print(dataset, ranks, cfg['gnd'])

        # Output ranked images
        # rank_res = {}
        # for i in range(len(qimages_r_path)):
        #     rank_res[qimages_r_path[i]] = [images_r_path[j] for j in match_idx[i,:]]
        # file_rankres = 'outputs/' + dataset + '_ranking_result.pkl'
        # a_file = open(file_rankres, "wb")
        # pickle.dump(rank_res, a_file)
        # a_file.close()

        # Visualization
        # Visualize the selected query image and its matching images
        # gnd = cfg['gnd']
        # K_show = 20
        # idx_select = 1
        # query_image = qimages[idx_select]
        # matching_images = [images[j] for j in match_idx[idx_select, :K_show]]
        # plt.close('all')
        # plt.figure(figsize=(10, 4), dpi=80)
        # ax = plt.subplot2grid((2, K_show), (0, 0))
        # ax.axis('off')
        # ax.set_title('Query')
        # img = mpimg.imread(query_image)
        # plt.imshow(img)
        # for i in range(K_show):
        #     if dataset == 'oxford5k' or dataset == 'paris6k':
        #         if np.in1d(match_idx[idx_select, i], gnd[idx_select]['ok'])[0]:
        #             plt.rcParams["axes.edgecolor"] = "green"
        #         else:
        #             plt.rcParams["axes.edgecolor"] = "red"
        #     if dataset == 'roxford5k' or dataset == 'rparis6k':
        #         if np.in1d(match_idx[idx_select, i], gnd[idx_select]['easy'])[0]:
        #             plt.rcParams["axes.edgecolor"] = "green"
        #         elif np.in1d(match_idx[idx_select, i], gnd[idx_select]['hard'])[0]:
        #             plt.rcParams["axes.edgecolor"] = "blue"
        #         elif np.in1d(match_idx[idx_select, i], gnd[idx_select]['junk'])[0]:
        #             plt.rcParams["axes.edgecolor"] = "red"
        #     plt.rcParams["axes.linewidth"] = 2.50
        #     ax = plt.subplot2grid((2, K_show), (1, i))
        #     ax.xaxis.set_ticks([])
        #     ax.yaxis.set_ticks([])
        #     ax.set_title('Match #' + str(i + 1))
        #     img = mpimg.imread(matching_images[i])
        #     plt.imshow(img)
        # plt.tight_layout(pad=2)
        # file_vis_path = 'outputs/' + dataset + '_' + str(idx_select) + '_vis.png'
        # plt.savefig(file_vis_path)

        print("extracting time per query : ", qextract_per_query)
        # print("retrieve time per query: ", retrieve_per_query)
        print('>> {}: whole elapsed time: {}'.format(dataset, htime(time.time()-start)))

        # Extract features of selfmade datasets
        # extr_selfmade_dataset(net, 'Andrea', transform, ms, msp, Lw)
        # extr_selfmade_dataset(net, 'flickr100k', transform, ms, msp, Lw)
        # extr_selfmade_dataset(net, 'Custom/database', transform, ms, msp, Lw)
        # extr_selfmade_dataset(net, 'Custom/query', transform, ms, msp, Lw)
        # extr_selfmade_dataset(net, 'GLM/test', transform, ms, msp, Lw)
        # extr_selfmade_dataset(net, 'GLM/index', transform, ms, msp, Lw)

if __name__ == '__main__':
    main()
