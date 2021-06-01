#!/usr/bin/env python3
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# Load grayscale images
img1 = cv.imread("./datasets/light/img1.ppm", cv.IMREAD_GRAYSCALE)
img2 = cv.imread("./datasets/light/img2.ppm", cv.IMREAD_GRAYSCALE)

# Load homography (geometric transformation between image)

homography=np.loadtxt("./datasets/light/H1to2p")
print(f"Homography from img1 to img2:\n{homography}")

detector = cv.ORB_create(100)
kpts1 = detector.detect(img1, None)
kpts2 = detector.detect(img2, None)

# Comment or uncomment to use ORB or BEBLID
descriptor = cv.xfeatures2d.BEBLID_create(0.75)
# descriptor = cv.ORB_create()

kpts1, desc1 = descriptor.compute(img1, kpts1)
kpts2, desc2 = descriptor.compute(img2, kpts2)

matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)
nn_matches = matcher.knnMatch(desc1, desc2, 2)

matched1 = []
matched2 = []
nn_match_ratio = 0.8  # Nearest neighbor matching ratio

for m, n in nn_matches:
    if m.distance < nn_match_ratio * n.distance:
        matched1.append(kpts1[m.queryIdx])
        matched2.append(kpts2[m.trainIdx])


inliers1 = []
inliers2 = []
good_matches = []
inlier_threshold = 2.5  # Distance threshold to identify inliers with homography check
for i, m in enumerate(matched1):
    # Create the homogeneous point
    col = np.ones((3, 1), dtype=np.float64)
    col[0:2, 0] = m.pt
    # Project from image 1 to image 2
    col = np.dot(homography, col)
    col /= col[2, 0]
    # Calculate euclidean distance
    dist = np.sqrt(pow(col[0, 0] - matched2[i].pt[0], 2) + pow(col[1, 0] - matched2[i].pt[1], 2))
    if dist < inlier_threshold:
        good_matches.append(cv.DMatch(len(inliers1), len(inliers2), 0))
        inliers1.append(matched1[i])
        inliers2.append(matched2[i])

res = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
# cv.drawMatches(img1, inliers1, img2, inliers2, good_matches, res)

all_matches = []
all_matches.append(cv.DMatch(len(matched1), len(matched2), 0))
cv.drawMatches(img1, matched1, img2, matched2, good_matches, res)

# plt.figure(figsize=(15, 5))
cv.imshow('matches',res)
cv.waitKey(0)