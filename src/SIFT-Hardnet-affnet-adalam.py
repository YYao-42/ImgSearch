'''
1 import
'''
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import torch
import kornia as K
import kornia.feature as KF
from kornia_moons.feature import *
import argparse
import time

from utils.adalam import AdalamFilter

'''
2 read images
'''
img1_pth = '/users/cristianmeo/Image_retrieval/pipeline/src/all_souls_000006.jpg'
img2_pth = '/users/cristianmeo/Image_retrieval/pipeline/src/all_souls_000002.jpg'
img1_raw = cv2.imread(img1_pth, cv2.COLOR_BGR2RGB)
img2_raw = cv2.imread(img2_pth, cv2.COLOR_BGR2RGB)
img1 = cv2.resize(img1_raw, (640, 480))
img2 = cv2.resize(img2_raw, (640, 480))

h1,w1 = img1.shape[:2]
h2,w2 = img2.shape[:2]

device = torch.device('cpu')
try:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print ("GPU mode")
except:
    print ('CPU mode')

'''
3 提取特征点
'''
# SIFT (DoG) Detector
sift_det = cv2.xfeatures2d.SIFT_create(8000, contrastThreshold=-10000, edgeThreshold=-10000)
# HardNet8 descriptor
hardnet8 = KF.HardNet8(True).eval().to(device)
# Affine shape estimator
affnet = KF.LAFAffNetShapeEstimator(True).eval().to(device)

Tk1 = time.time()
keypoints1 = sift_det.detect(img1, None)[:8000]
Tk2= time.time()
print(Tk2 - Tk1)

Td1= time.time()
with torch.no_grad():
    timg1 = K.image_to_tensor(img1, False).float() / 255.
    timg1 = timg1.to(device)
    timg_gray1 = K.rgb_to_grayscale(timg1)

    lafs1 = laf_from_opencv_SIFT_kpts(keypoints1, device=device)
    lafs_new1 = affnet(lafs1, timg_gray1)

    patches = KF.extract_patches_from_pyramid(timg_gray1, lafs_new1, 32)
    B1, N1, CH1, H1, W1 = patches.size()
    # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
    # So we need to reshape a bit :)
    descriptors1 = hardnet8(patches.view(B1 * N1, CH1, H1, W1)).view(B1 * N1, -1).detach().cpu().numpy()
    # np.save("/descriptors1.npy", descriptors1)
Td2= time.time()
print("Descriptor time for a image: {}".format(Td2 - Td1))

keypoints2 = sift_det.detect(img2, None)[:8000]

with torch.no_grad():
    timg2 = K.image_to_tensor(img2, False).float() / 255.
    timg2 = timg2.to(device)
    timg_gray2 = K.rgb_to_grayscale(timg2)

    lafs2 = laf_from_opencv_SIFT_kpts(keypoints2, device=device)
    lafs_new2 = affnet(lafs2, timg_gray2)

    patches = KF.extract_patches_from_pyramid(timg_gray2, lafs_new2, 32)
    B2, N2, CH2, H2, W2 = patches.size()
    # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
    # So we need to reshape a bit :)
    descriptors2 = hardnet8(patches.view(B2 * N2, CH2, H2, W2)).view(B2 * N2, -1).detach().cpu().numpy()

'''
(4) 特征点转换
'''
def convert_kpts(cv2_kpts):
    keypoints = np.array([(x.pt[0], x.pt[1]) for x in cv2_kpts ]).reshape(-1, 2)
    scales = np.array([12.0* x.size for x in cv2_kpts ]).reshape(-1, 1)
    angles = np.array([x.angle for x in cv2_kpts ]).reshape(-1, 1)
    responses = np.array([x.response for x in cv2_kpts]).reshape(-1, 1)
    return keypoints, scales, angles, responses

'''
5 match
'''
Tm1= time.time()
matcher = AdalamFilter()
kp1, s1, a1, r1 = convert_kpts(keypoints1)
# np.save("/kp1.npy", kp1)
# np.save("/s1.npy", s1)
# np.save("/a1.npy", a1)
# np.save("/r1.npy", r1)
kp2, s2, a2, r2 = convert_kpts(keypoints2)

idxs = matcher.match_and_filter(kp1, kp2,
                            descriptors1, descriptors2,
                            im1shape=(h1,w1),
                            im2shape=(h2,w2),
                            o1=a1.reshape(-1),
                            o2=a2.reshape(-1),
                            s1=s1.reshape(-1),
                            s2=s2.reshape(-1)).detach().cpu().numpy()

Tm2= time.time()
print(Tm2 - Tm1)

# print(idxs)

# src_pts = kp1[idxs[:,0]]
# dst_pts = kp2[idxs[:,1]]
#
# F, inliers_mask = cv2.findFundamentalMat(
#         src_pts,
#         dst_pts,
#         method=cv2.USAC_MAGSAC,
#         ransacReprojThreshold=0.25,
#         confidence=0.99999,
#         maxIters=100000)
#
# inliers_mask = np.array(inliers_mask).astype(bool).reshape(-1)
#
# matches_results = {}
# matches_results[0] = np.concatenate([src_pts[inliers_mask], dst_pts[inliers_mask]], axis=1)

def show_matches(img1, img2, points1, points2, target_dim=800.):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    def resize_horizontal(h1, w1, h2, w2, target_height):
        scale_to_align = float(h1) / h2
        current_width = w1 + w2 * scale_to_align
        scale_to_fit = target_height / h1
        target_w1 = int(w1 * scale_to_fit)
        target_w2 = int(w2 * scale_to_align * scale_to_fit)
        target_h = int(target_height)
        return (target_w1, target_h), (target_w2, target_h), scale_to_fit, scale_to_fit * scale_to_align, [target_w1, 0]

    target_1, target_2, scale1, scale2, offset = resize_horizontal(h1, w1, h2, w2, target_dim)

    im1 = cv2.resize(img1, target_1, interpolation=cv2.INTER_AREA)
    im2 = cv2.resize(img2, target_2, interpolation=cv2.INTER_AREA)

    h1, w1 = target_1[::-1]
    h2, w2 = target_2[::-1]

    vis = np.ones((max(h1, h2), w1 + w2, 3), np.uint8) * 255
    vis[:h1, :w1] = im1
    vis[:h2, w1:w1 + w2] = im2

    p1 = [np.int32(k * scale1) for k in points1]
    p2 = [np.int32(k * scale2 + offset) for k in points2]

    for (x1, y1), (x2, y2) in zip(p1, p2):
        cv2.line(vis, (x1, y1), (x2, y2), [0, 255, 0], 1)

    cv2.imwrite('/users/cristianmeo/Image_retrieval/pipeline/src/SIFT-Hardnet-Affnet-AdaLAM.jpg', vis)

show_matches(img1, img2, points1=kp1[idxs[:, 0]], points2=kp2[idxs[:, 1]])

print(len(idxs))