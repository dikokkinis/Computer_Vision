#!/usr/bin/env python
# coding: utf-8

# # 2η Εργαστηριακή Άσκηση στην Όραση Υπολογιστών
# 
# Συμμετέχοντες:
# 
# Δημήτριος Κοκκίνης 03118896
# 
# Χριστίνα Ρεντίφη 03118217
# 
# **Θέμα: Εκτίμηση Οπτικής Ροής (Optical Flow), Εξαγωγή Χαρακτηριστικών σε Βίντεο για Αναγνώριση Δράσεων, Συνένωση Εικόνων (Image Stitching)**

# ## ***Μέρος 3: Συνένωση Εικόνων (Image Stitching) για Δημιουργία Πανοράματος***

# In[2]:


pip install pyflann-py3


# In[1]:


#imports
from PIL import Image

import sys
import os
import os, os.path

import numpy as np
import cv2

from matplotlib import pyplot as plt

from pyflann import *
from numpy import *
from numpy.random import *


# In[2]:


sys.path.insert(0,'D:\\Επιφάνεια εργασίας\\OY-Lab2\\cv23_lab2_material\\part3 - ImageStitching')


# In[3]:


os.chdir('D:\\Επιφάνεια εργασίας\\OY-Lab2\\cv23_lab2_material\\part3 - ImageStitching')


# *Βήμα 0: Διάβασμα εικόνων*

# In[4]:


IMAGES = []
path = os.getcwd()
valid_images = [".jpg",".gif",".png",".tga"]
for f in os.listdir(path):
    name = os.path.splitext(f)[0]
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    
    BGRImage = cv2.imread(f)
    img = cv2.cvtColor(BGRImage, cv2.COLOR_BGRA2RGB)
    
    IMAGES.append(img)
      


# *Βήμα 1: Εντοπισμός χαρακτηριστικών ενδιαφέροντος και εξαγωγή περιγραφητών*

# In[10]:


def keyp_desc(images):
    
    keypoints = []
    descriptors = []
    for i in range(len(images)):
        # Converting image to grayscale
        gray = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY)

        # Applying SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = (sift.detectAndCompute(gray, None))
    
        keypoints.append(kp)
        descriptors.append(des)
        
    return keypoints, descriptors


# In[11]:


keypoints, descriptors = keyp_desc(IMAGES)
for i in range(len(IMAGES)):
    # Marking the keypoint on the image using circles
    img_key = cv2.drawKeypoints(IMAGES[i], keypoints[i], None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
    plt.imshow(img_key)
    plt.axis('off')
    plt.show()


# *Βήμα 2: Διαδικασία ταιριάσματος χαρακτηριστικών μεταξύ δύο εικόνων και Βήμα 3: Διατήρηση μόνο των πιο έγκυρων αντιστοιχίσεων μέσω εφαρμογής του
# κριτηρίου Lowe*

# In[5]:


#Function returns the feature matched image
def Flanned_Matcher(main_image,sub_image, keyp, descr, threshold):
   
    #Keypoints and descriptors with SIFT.
    key_point1 = keyp[0]
    key_point2 = keyp[1]
    descr1 = descr[0]
    descr2 = descr[1]
 

    ######   STEP 2   ######
    # FLANN parameters.
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = None
 
    # FLANN based matcher with implementation of k nearest neighbour.
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(descr1,descr2,k=2)
 
    # selecting only good matches.
    matchesMask = [[0,0] for i in range(len(matches))]
    good = []
    
    
    ######   STEP 3   ######
    # Lowe's ratio test
    for i, (m,n) in enumerate(matches):
        if m.distance < threshold*n.distance:
            matchesMask[i]=[1,0]
            good.append(m)
    good = np.asarray(good)   
    
    draw_params1 = dict(matchColor = (62,242,134),
                       singlePointColor = (0,255,255),flags = 0)
    
    draw_params2 = dict(matchColor = (62,242,134),
                       singlePointColor = (0,255,255), matchesMask = matchesMask, flags = 0)
    
    print("Features before Lowe Threshold: ", np.shape(matches)[0])
    print("Features after Lowe Threshold: ", np.shape(good)[0])
    
    # drawing nearest neighbours before Lowe's criterion
    img = cv2.drawMatchesKnn(main_image,
                            key_point1,
                            sub_image,
                            key_point2,
                            matches,
                            None,
                            **draw_params1)
    
    # drawing nearest neighbours after Lowe's criterion
    img2 = cv2.drawMatchesKnn(main_image,
                            key_point1,
                            sub_image,
                            key_point2,
                            matches,
                            None,
                            **draw_params2)
    
    
    return img, img2,  good


# In[13]:


#Passing two input images
keyp1 = [keypoints[0], keypoints[1]]
descr1 = [descriptors[0], descriptors[1]]

output1, output11, _ = Flanned_Matcher(IMAGES[0],IMAGES[1], keyp1, descr1, 0.8)

fig = plt.figure(dpi=200)

fig.add_subplot(1,2,1)
plt.imshow(output1)
plt.axis('off')

fig.add_subplot(1,2,2)
plt.imshow(output11)
plt.axis('off')


# *Βήμα 4: Υπολογισμός ομογραφίας με RANSAC*

# In[6]:


def Homography(img1, img2, matches):


    src_pts = np.float32([ keyp1[0][m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ keyp1[1][m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

    homoMatrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w, d = IMAGES[0].shape
    
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    
    dst = cv2.perspectiveTransform(pts,homoMatrix)

    dst = dst.reshape((4,2))
    image_line = img2.copy()

    homoline = cv2.polylines(image_line,[np.int32(dst)],True,255,2, cv2.LINE_AA)
    
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    img3 = cv2.drawMatches(img1, keyp1[0], homoline, keyp1[1], matches, None,**draw_params)
    
    return homoMatrix, homoline, img3


# In[15]:


_,_, good1 = Flanned_Matcher(IMAGES[0],IMAGES[1], keyp1, descr1,0.8)

Matrix, homoline, img3 = Homography(IMAGES[0], IMAGES[1], good1)

print(Matrix)

plt.imshow(homoline)
plt.axis('off')


# *Βήμα 5: Υπολογισμός μετασχηματισμένης εικόνας μέσω εφαρμογής Inverse warping*

# In[47]:


def stitchImages(H, img1, img2, panorama_style):
    
    h, w, d = img1.shape
    
    #######    Four corners of the image    #########
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    
    #######   Corners of warped image after transformation based on homography  ######
    dst = cv2.perspectiveTransform(pts, H)
    dst = dst.reshape((4,2))


    gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

    pts_src = np.array(pts)

    
    #######   Changing current dimensions to panorama dimensions  #######
    dim_y = img2.shape[0]
    dim_x = img2.shape[1]
    

    widths = dst[:,0]
    heights = dst[:,1]
    
    
    negative_widths = []
    larger_widths = []
    negative_heights = []
    larger_heights = []
    
    for width in widths:
        if width < 0:
            negative_widths.append(abs(width))
        elif width > dim_x:
            larger_widths.append(width)
    
    for height in heights:
        if height < 0:
            negative_heights.append(abs(height))
        elif height > dim_y:
            larger_heights.append((height - img2.shape[0]))
      
    
    if len(negative_widths) > 0:
        dim_x = dim_x + max(negative_widths)
    elif len(larger_widths) > 0:
        dim_x = dim_x + (max(larger_widths) - img2.shape[1])
  
    if len(negative_heights) > 0:
        dim_y = dim_y + negative_heights[0]
    if len(larger_heights) > 0 :
        dim_y = dim_y + larger_heights[0]
    
    #########     Panorama dimensions    #########
    dim_x = int(dim_x)
    dim_y = int(dim_y)    
    
    ########   Points of warped image to new dimensions  #######
    if panorama_style == 'left_to_right':
        pts_dst = np.zeros((4,2))
        pts_dst[:, 0] = [i + -dst[0][0] for i in dst[:,0]]
        pts_dst[:, 1] = [i + -dst[0][1] for i in dst[:,1]]
    elif panorama_style == 'right_to_left':
        pts_dst = np.zeros((4,2))
        pts_dst[:, 0] = dst[:,0]
        if len(negative_heights) > 0 :
            pts_dst[:, 1] = [i + -dst[3][1] for i in dst[:,1]]
        else:
            pts_dst[:, 1] = dst[:,1]

    h, _= cv2.findHomography(pts_src, pts_dst)

    ######   Warp source image to destination based on homography   #######
    warp_image = cv2.warpPerspective(img1, h, (dim_x, dim_y), flags=cv2.INTER_LINEAR)
    
    ######   Add two images together   #######
    if panorama_style == 'left_to_right':
        if len(negative_widths) > 0:
            x_offset= int(np.round((max(negative_widths))))
        else:
            x_offset = 0
        if len(negative_heights) > 0:
            y_offset= int(np.round(negative_heights[0]))
        else:
            y_offset = 0
        
        panorama = warp_image.copy()
        panorama = cv2.resize(panorama,(img2.shape[1] + x_offset, warp_image.shape[0]))
        panorama[y_offset:y_offset+img2.shape[0], x_offset:x_offset+img2.shape[1]] = img2
        
    elif panorama_style == 'right_to_left':
        #x_offset= int(np.round(max(larger_widths)-img2.shape[1]))
        if len(negative_heights) > 0:
            y_offset= int(np.round(negative_heights[0]))
        else:
            y_offset= 0
        
        panorama = warp_image.copy()
        #panorama = cv2.resize(panorama, (warp_image.shape[1], warp_image.shape[0]))
        panorama[y_offset:y_offset + img2.shape[0], 0 : img2.shape[1]] = img2
    
    return warp_image, dst[0], panorama


# In[23]:


output_img, top_left_coords, panorama = stitchImages(Matrix, IMAGES[0], IMAGES[1], 'left_to_right')

plt.imshow(panorama)
plt.axis('off')
print(top_left_coords)


# ***Συνένωση όλων των εικόνων***

# Θα συνοψίσουμε όλη τη διαδικασία σε μια συνάρτηση

# In[24]:


#Redesigning keyp_desc function

def keyp_desc2(img1, img2):
    
    keypoints = []
    descriptors = []

    # Converting image to grayscale
    gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

    # Applying SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = (sift.detectAndCompute(gray, None))
    
    keypoints.append(kp)
    descriptors.append(des)
    
    # Converting image to grayscale
    gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Applying SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = (sift.detectAndCompute(gray, None))
    
    keypoints.append(kp)
    descriptors.append(des)
        
    return keypoints, descriptors


# In[25]:


def Make_Panorama(img1, img2, panorama_style, threshold):
    
    ######  Finding keypoints and Descriptors  ######
    if panorama_style == 'left_to_right':
        keypoints, descriptors = keyp_desc2(img1 ,img2)
    elif panorama_style == 'right_to_left':
        keypoints, descriptors = keyp_desc2(img1 ,img2)
        
    ######  Finding matches between two images  ######
    _, matches, good = Flanned_Matcher(img1,img2, keypoints, descriptors, threshold)
            
    ######  Warping img1 based on homography  ######
    src_pts = np.float32([ keypoints[0][m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoints[1][m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    homoMatrix, mask1 = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    warped_img, top_left_coords, panorama = stitchImages(homoMatrix, img1, img2, panorama_style)
                
               
    return matches, warped_img, panorama, top_left_coords


# In[57]:


matches, output_img, panorama1, top_left_coords = Make_Panorama(IMAGES[0],IMAGES[1],'left_to_right', 0.8)

plt.imshow(panorama1)
plt.axis('off')


# In[59]:


panA = panorama1[int(np.round(abs(top_left_coords[1]))) :int(np.round(abs(top_left_coords[1]))) + IMAGES[1].shape[0], 0 : panorama1.shape[1]]

plt.imshow(panA)


# In[76]:


matches2, output_img2, panorama2, top_left_coords = Make_Panorama(IMAGES[1],IMAGES[2],'left_to_right', 0.8)

plt.imshow(panorama2)
plt.axis('off')


# In[77]:


panB = panorama2[int(np.round(abs(top_left_coords[1]))) :int(np.round(abs(top_left_coords[1]))) + IMAGES[2].shape[0], 28 : panorama2.shape[1]]

plt.imshow(panB)


# In[79]:


matches3, output_img3, panorama3, top_left_coords = Make_Panorama(panA,panB,'left_to_right', 0.8)

plt.imshow(panorama3)
plt.axis('off')


# In[86]:


panC = panorama3[int(np.round(abs(top_left_coords[1]))) :int(np.round(abs(top_left_coords[1]))) + panB.shape[0], 160 : panorama3.shape[1]]

plt.imshow(panC)


# In[111]:


plt.imshow(panC)
plt.axis('off')


# In[88]:


matches4, output_img4, panorama4, top_left_coords = Make_Panorama(IMAGES[5],IMAGES[4],'right_to_left', 0.8)

plt.imshow(panorama4)
plt.axis('off')


# In[99]:


panD = panorama4[95 : 95 + IMAGES[4].shape[0], 0 : panorama4.shape[1]]

plt.imshow(panD)
plt.axis('off')


# In[100]:


matches5, output_img5, panorama5, top_left_coords = Make_Panorama(IMAGES[4],IMAGES[3],'right_to_left', 0.8)

plt.imshow(panorama5)
plt.axis('off')


# In[104]:


panE = panorama5[95 : 95 + IMAGES[3].shape[0], 0 : panorama5.shape[1]]

plt.imshow(panE)
plt.axis('off')


# In[106]:


matches6, output_img6, panorama6, top_left_coords = Make_Panorama(panE,panD,'left_to_right', 0.8)

plt.imshow(panorama6)
plt.axis('off')


# In[107]:


panF = panorama6[int(np.round(abs(top_left_coords[1]))) :int(np.round(abs(top_left_coords[1]))) + panD.shape[0], 0 : panorama6.shape[1]]


# In[108]:


plt.imshow(panF)
plt.axis('off')


# In[113]:


matches7, output_img7, panorama7, top_left_coords = Make_Panorama(panorama6,panorama3,'right_to_left', 0.6)

plt.imshow(panorama7)
plt.axis('off')


# In[ ]:




