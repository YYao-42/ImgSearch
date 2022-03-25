"""
Image Search Engine for Historical Research: A Prototype
This file contains general functions that are not related to any specific module
but related to the overall process
"""

import os
import sys
import shutil
import numpy as np
import torch.backends.cudnn as cudnn
import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms

from numpy import linalg as LA
from torch.autograd import Variable


def train(net, sup_PQ, criterion, optimizer, scheduler, trainloader,
          valloader, start_epoch, epochs, is_gpu, valid_loss_min_input):
    """
    Training process.
    Args:
        net: Triplet Net
        sup_PQ: whether using supervised product quantization
        criterion: TripletMarginLoss
        optimizer: SGD with momentum optimizer
        scheduler: scheduler
        trainloader: training set loader
        valloader: validation set loader
        start_epoch: checkpoint saved epoch
        epochs: training epochs
        is_gpu: whether use GPU
        valid_loss_min_input: initial value of the minimum validation loss
    """
    print("==> Start training ...")

    if is_gpu:
      net.cuda()

    valid_loss_min = valid_loss_min_input
    for epoch in range(start_epoch, epochs + start_epoch):

        ##################
        # Train the model
        ##################
        net.train()
        running_loss = 0.0
        for batch_idx, (data1, data2, data3) in enumerate(trainloader):

            if is_gpu:
                data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()

            # wrap in torch.autograd.Variable
            data1, data2, data3 = Variable(
                data1), Variable(data2), Variable(data3)

            # compute output and loss
            if sup_PQ:
                embedded_a, quantized_p, quantized_n, _, _ = net(data1, data2, data3)
                loss = criterion(embedded_a, quantized_p, quantized_n)
            else:
                embedded_a, embedded_p, embedded_n = net(data1, data2, data3)
                loss = criterion(embedded_a, embedded_p, embedded_n)

            # loss = torch.sigmoid(loss)

            # compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data.item()

            if batch_idx % 30 == 0:
                print("mini Batch Training Loss: {}".format(loss.data.item()))
        ##################
        # Validate the model
        ##################
        net.eval()
        valid_loss = 0.0
        for batch_idx, (data1, data2, data3) in enumerate(valloader):

            if is_gpu:
                data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()

            # wrap in torch.autograd.Variable
            data1, data2, data3 = Variable(
                data1), Variable(data2), Variable(data3)

            # compute output and loss
            if sup_PQ:
                embedded_a, quantized_p, quantized_n, _, _ = net(data1, data2, data3)
                loss = criterion(embedded_a, quantized_p, quantized_n)
            else:
                embedded_a, embedded_p, embedded_n = net(data1, data2, data3)
                loss = criterion(embedded_a, embedded_p, embedded_n)

            # print statistics
            valid_loss += loss.data.item()

            if batch_idx % 30 == 0:
                print("mini Batch Validation Loss: {}".format(loss.data.item()))


        # Normalizing the loss by the total number of train batches
        running_loss /= len(trainloader)
        valid_loss /= len(valloader)

        print("Training Epoch: {0} | Training Loss: {1} | Val Loss: {2} ".format(epoch+1, running_loss, valid_loss))

        if valid_loss <= valid_loss_min:
            is_best = True
            valid_loss_min = valid_loss
        else:
            is_best = False

        # remember best acc and save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, is_best)

    print('==> Finished Training ...')


def calculate_distance(i1, i2):
    """
    Calculate euclidean distance of the ranked results from the query image.

    Args:
        i1: query image
        i2: ranked result
    """
    return np.sum((i1 - i2) ** 2)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Save checkpoint."""
    directory = "../checkpoint/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, directory + 'model_best.pth.tar')


def get_feature(imageloader, net, is_gpu):
    """Generate features of the images in the database (without PQ layer)"""
    if is_gpu:
        net = torch.nn.DataParallel(net).cuda()
        cudnn.benchmark = True

    print('==> Retrieve model parameters ...')
    # checkpoint = torch.load("../checkpoint/checkpoint.pth.tar")
    checkpoint = torch.load("../checkpoint/model_best.pth.tar")
    net.load_state_dict(checkpoint['state_dict'])
    # net.load_state_dict(checkpoint['state_dict'], strict=False)
    net.eval()

    embedded_features = []
    labels = []
    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(imageloader):

            if is_gpu:
                image = image.cuda()

            # wrap in torch.autograd.Variable
            image = Variable(image)

            # compute output
            embedded_a, _, _ = net(image, image, image)
            embedded_a_numpy = embedded_a.data.cpu().numpy()
            embedded_a_numpy.astype('float32')

            embedded_features.append(embedded_a_numpy)
            labels.append(label)

    embedded_features_tot = np.concatenate(embedded_features, axis=0)
    labels_tot = np.concatenate(labels, axis=0)
    return embedded_features_tot, labels_tot


def get_feature_PQ(imageloader, net, is_gpu):
    """Generate features of the images in the database (with PQ layer)"""
    if is_gpu:
        net = torch.nn.DataParallel(net).cuda()
        cudnn.benchmark = True

    print('==> Retrieve model parameters ...')
    # checkpoint = torch.load("../checkpoint/checkpoint.pth.tar")
    checkpoint = torch.load("../checkpoint/model_best.pth.tar")
    net.load_state_dict(checkpoint['state_dict'])
    # net.load_state_dict(checkpoint['state_dict'], strict=False)
    net.eval()

    embedded_features = []
    quantized_features =[]
    idx = []
    labels = []
    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(imageloader):

            if is_gpu:
                image = image.cuda()

            # wrap in torch.autograd.Variable
            image = Variable(image)

            # compute output
            embedded_a, embedded_p, _, idx_p, _ = net(image, image, image)
            embedded_a_numpy = embedded_a.cpu().detach().numpy()
            embedded_a_numpy.astype('float32')
            embedded_p_numpy = embedded_p.cpu().detach().numpy()
            embedded_p_numpy.astype('float32')
            idx_numpy = idx_p.cpu().detach().numpy()

            embedded_features.append(embedded_a_numpy)
            quantized_features.append(embedded_p_numpy)
            idx.append(idx_numpy)
            labels.append(label)

    embedded_features_tot = np.concatenate(embedded_features, axis=0)
    quantized_features_tot = np.concatenate(quantized_features, axis=0)
    idx_tot = np.concatenate(idx, axis=0)
    labels_tot = np.concatenate(labels, axis=0)
    return embedded_features_tot, quantized_features_tot, idx_tot, labels_tot


def save_feature(embedded_features, labels):
    np.save('../embedded_features.npy', embedded_features)
    np.save('../labels.npy', labels)


def save_feature_PQ(embedded_features, labels, idx):
    np.save('../embedded_features.npy', embedded_features)
    np.save('../labels.npy', labels)
    np.save('../CWindex.npy', idx)


def hard_quantization(Codewords, CW_idx, N_books):
    """Get hard-quantized features"""
    Codewords = torch.from_numpy(Codewords)
    N_words, dim = Codewords.shape
    L_word = int(dim / N_books)
    c = torch.split(Codewords, L_word, 1)
    for i in range(N_books):
        ci = c[i]
        idx_all_query = CW_idx[:, i]
        if i == 0:
            hard_quantized_feature = ci[idx_all_query]
        else:
            hard_quantized_feature = torch.cat((hard_quantized_feature, ci[idx_all_query]), dim=1)
    hard_quantized_feature = hard_quantized_feature.cpu().detach().numpy()
    return hard_quantized_feature
