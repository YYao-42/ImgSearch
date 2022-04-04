import argparse
import os
import torch
import time
import pickle
import numpy as np
from PIL import Image
from torchvision import transforms
from src.imageretrievalnet import init_network, extract_vectors, extract_vectors_single
from datetime import datetime as dt
from flask import Flask, request, render_template
from pathlib import Path
from torch.utils.model_zoo import load_url
from src.utils.general import get_data_root, htime
from src.layers.whiten import whitenlearn, whitenapply
from src.datasets.datahelpers import cid2filename
from src.datasets.testdataset import configdataset
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
parser.add_argument('--K-nearest-neighbour', '-K', default=30, type=int, metavar='K',
                    help="retreive top-K results (default: 30)")
parser.add_argument('--image-size', '-imsize', default=1024, type=int, metavar='N',
                    help="maximum size of longer image side used for testing (default: 1024)")
parser.add_argument('--multiscale', '-ms', metavar='MULTISCALE', default='[1]',
                    help="use multiscale vectors for testing, " +
                    " examples: '[1]' | '[1, 1/2**(1/2), 1/2]' | '[1, 2**(1/2), 1/2**(1/2)]' (default: '[1]')")
parser.add_argument('--whitening', '-w', metavar='WHITENING', default=None, choices=whitening_names,
                    help="dataset used to learn whitening for testing: " +
                        " | ".join(whitening_names) +
                        " (default: None)")

# GPU ID
parser.add_argument('--gpu-id', '-g', default='0', metavar='N',
                    help="gpu id used for testing (default: '0')")

# parse the arguments
args = parser.parse_args()

app = Flask(__name__)
# app = Flask(__name__, static_folder='C:\\Users\\31601\\Downloads\\test\\')

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
    net_params['pretrained'] = True

    # load off-the-shelf network
    print(">> Loading off-the-shelf network:\n>>>> '{}'".format(args.network_offtheshelf))
    net = init_network(net_params)
    print(">>>> loaded network: ")
    print(net.meta_repr())

# setting up the multi-scale parameters
ms = list(eval(args.multiscale))
if len(ms)>1 and net.meta['pooling'] == 'gem' and not net.meta['regional'] and not net.meta['whitening']:
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
# Read image features
datasets = args.datasets.split(',')
# TODO: give a parameter to 2048
vecs = np.empty((2048, 0))
img_paths = []
for dataset in datasets:
    cfg = configdataset(dataset, os.path.join(get_data_root(), 'test'))
    file_vecs = 'outputs/' + dataset + '_vecs.npy'
    vecs = np.concatenate([vecs, np.load(file_vecs)], axis=1)
    # images = [cfg['im_fname'](cfg, i) for i in range(cfg['n'])]
    images_r_path = [cfg['im_fname'](cfg, i).split('\\')[-1] for i in range(cfg['n'])]
    images = ['/static/test/' + dataset + '/jpg/' + i for i in images_r_path]
    img_paths = img_paths + images
# TODO: give a parameter to self-made dataset
file_path_feature = 'outputs/' + 'Andrea' + '_path_feature.pkl'
with open(file_path_feature, 'rb') as pickle_file:
    path_feature = pickle.load(pickle_file)
vecs = np.concatenate([vecs, path_feature['feature']], axis=1)
images = ['/static/test/' + 'Andrea/' + i for i in path_feature['path']]
img_paths = img_paths + images
# img_r_paths = images_r_path +[i for i in path_feature['path']]

K = args.K_nearest_neighbour

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        # if not os.path.exists("static/uploaded/"):
        #     os.makedirs("static/uploaded/")
        uploaded_img_path = "src/static/uploaded/" + dt.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)
        query_path = '/' + '/'.join(uploaded_img_path.split('/')[1:])
        # qvec = np.random.rand(2048, 1)
        # qvec = extract_vectors(net, uploaded_img_path, args.image_size, transform, ms=ms, msp=msp)
        qvec = extract_vectors_single(net, uploaded_img_path, args.image_size, transform, ms=ms, msp=msp)
        qvec = np.expand_dims(qvec.numpy(), axis=1)
        if Lw is not None:
            # whiten the vectors
            qvec = whitenapply(qvec, Lw['m'], Lw['P'])

        # Run search
        match_idx, _ = matching_L2(K, vecs.T, qvec.T)
        # match_idx, time_per_query = matching_Nano_PQ(K, vecs.T, qvecs.T, 16, 8)
        # match_idx, time_per_query = matching_ANNOY(K, vecs.T, qvecs.T, 'euclidean')
        # match_idx, time_per_query = matching_HNSW(K, vecs.T, qvecs.T)
        # embedded_code, Codewords, _ = Nano_PQ(vecs.T, 16, 256)
        # match_idx, time_per_query = matching_PQ_Net(K, Codewords, qvecs.T, 16, embedded_code)
        # match_idx, time_per_query = matching_HNSW_PQ(K, Codewords, qvecs.T, embedded_code)
        
        # TODO: id -> dist[id]
        scores = [(id, img_paths[id]) for id in np.squeeze(match_idx)]
        # scores = [(img_r_paths[id], img_paths[id]) for id in np.squeeze(match_idx)]

        return render_template('index.html',
                               query_path=query_path,
                               scores=scores)
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run("0.0.0.0")
    # app.run(host="localhost", port=8000, debug=True)