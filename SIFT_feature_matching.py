'''
SIFT: Scale Invariant Feature Transform
FLANN: Fast Library for Approximate Nearest Neighbors
1. Feature point generation using SIFT -> keypoint with descriptor
2. FLANN based feature matching
3. Estimate Homography using RANSAC based outlier filtering
4. Perspective transformation of the small part image into whole image to locate area.
'''

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img1_str = 'part1.JPG'
img2_str = 'whole_image.JPG'
img1 = cv.imread(img1_str,cv.IMREAD_GRAYSCALE)  # queryImage
img2 = cv.imread(img2_str,cv.IMREAD_GRAYSCALE)  # trainImage
img1_bgr = cv.imread(img1_str)                  # queryImage
img2_bgr = cv.imread(img2_str)                  # trainImage 
# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]
# ratio test as per Lowe's paper
good = []
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
        good.append(m)

MIN_MATCH_COUNT = 10
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    img2 = cv.polylines(img2_bgr,[np.int32(dst)],True,255,3, cv.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
                  
img3 = cv.drawMatches(img1_bgr,kp1,img2,kp2,good,None,**draw_params)
plt.imshow(img3),plt.show()
cv.imwrite('match_result.jpg', cv.cvtColor(img3, cv.COLOR_BGR2RGB))