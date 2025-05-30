#!/usr/bin/env python
# coding: utf-8

# ## **Μέρος 2: Εντοπισμός Χωρο-χρονικών Σημείων Ενδιαφέροντος και Εξαγωγή Χαρακτηριστικών σε Βίντεο Ανθρωπίνων Δράσεων**
# 

# ### **2.1 Χωρο-χρονικά Σημεία Ενδιαφέροντος**

# In[1]:


import sys  

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import scipy
from scipy import ndimage
import math
import sklearn
from sklearn import preprocessing
import os
from pathlib import Path


# In[2]:


sys.path.insert(0,'C:\\Users\\chris\\Desktop\\cv23_lab2_material\\part2 - SpatioTemporal')


# In[3]:


os.chdir('C:\\Users\\chris\\Desktop\\cv23_lab2_material\\part2 - SpatioTemporal')


# In[4]:


import cv23_lab2_2_utils
from cv23_lab2_2_utils import read_video
from cv23_lab2_2_utils import show_detection
from cv23_lab2_2_utils import orientation_histogram
from cv23_lab2_2_utils import bag_of_words
from cv23_lab2_2_utils import svm_train_test


# In[5]:


video_r = read_video('C:\\Users\\chris\\Desktop\\cv23_lab2_material\\part2 - SpatioTemporal\\running\\person01_running_d1_uncomp.avi',200)
video_w = read_video('C:\\Users\\chris\\Desktop\\cv23_lab2_material\\part2 - SpatioTemporal\\walking\\person04_walking_d1_uncomp.avi',200)
video_h = read_video('C:\\Users\\chris\\Desktop\\cv23_lab2_material\\part2 - SpatioTemporal\\handwaving\\person15_handwaving_d4_uncomp.avi',200)
print(video_w.shape)


# In[6]:


# Plot a video frame
frame_index = 25

video_frame_r = video_r[:, :, frame_index]
video_frame_w = video_w[:, :, frame_index]
video_frame_h = video_h[:, :, frame_index]

plt.imshow(video_frame_r, cmap='gray')  
plt.axis('off')  
plt.show()

plt.imshow(video_frame_w, cmap='gray')  
plt.axis('off')  
plt.show()

plt.imshow(video_frame_h, cmap='gray')  
plt.axis('off')  
plt.show()


# **2.1.1 Harris Detector**

# In[7]:


#Recommended vaules:
sigma = 4 # κλίμακα διαφόρησης- for Gaussian Gσ 
taf = 1.5 # χρονική κλίμακα - for Gaussian Gτ
s=2 # s*sigma , s*taf
k = 0.005 # for trace 


# In[8]:


#1D-Gaussian kernel - Used in Time Domain t
from math import pi, sqrt, exp

def gaussian1D(ksize_t,taf):
    r = range(int(-2*taf),int(2*taf))
    return [1 / (taf * sqrt(2*pi)) * exp(-float(t)**2/(2*taf**2)) for t in r]


# In[9]:


def Harris(video, sigma, taf, s):
    
    # convert to floats
    video = video.astype(np.float)

    ########   1D Gaussian Kernel - Time Domain  #######
    ksize_t = int(np.ceil(3*s*taf)*2 + 1)
    g_kernel_t = gaussian1D(ksize_t, s*taf)
    
    ########   2D Gaussian Kernel - Spatial Domain  #########
    ksize_s = int(np.ceil(3*s*sigma)*2 + 1)
    g_kernel_s = cv2.getGaussianKernel(ksize_s, s*sigma)
    g_space = g_kernel_s @ g_kernel_s.transpose()
    
    #######     Smooth the video    #########
    
    #spatial 2D smoothing
    smooth_video_s = cv2.filter2D(video, -1, g_space)
    #temporal 1D smoothing
    smooth_video_st = scipy.ndimage.convolve1d(smooth_video_s, g_kernel_t, axis=0, mode='reflect')
    
    #Derivative Filter
    derivative_filter = np.array([-1, 0, 1])

    #axis=0 is t    #axis = 1 is y     # axis = 2 is x
    
    #######     First Derivatives   #######
    gradient_x = scipy.ndimage.convolve1d(smooth_video_st, derivative_filter.T, axis=2, mode='reflect')
    gradient_y = scipy.ndimage.convolve1d(smooth_video_st, derivative_filter.T, axis=1, mode='reflect')
    gradient_t = scipy.ndimage.convolve1d(smooth_video_st, derivative_filter.T, axis=0, mode='reflect')

    #######   Second Derivatives   #########
    Lx = scipy.ndimage.convolve1d(gradient_x, derivative_filter.T, axis=2, mode='reflect')
    Ly = scipy.ndimage.convolve1d(gradient_y, derivative_filter.T, axis=1, mode='reflect')
    Lt = scipy.ndimage.convolve1d(gradient_t, derivative_filter.T, axis=0, mode='reflect')
    
    det_harris = (Lx**2)*((Ly**2)*(Lt**2) - (Ly*Lt)*(Ly*Lt)) - (Lx*Ly)*((Lx*Ly)*(Lt**2)-(Lx*Lt)*(Ly*Lt)) + (Lx*Lt)*((Lx*Ly)*(Ly*Lt) - (Lx*Lt)*(Ly**2))
    trace_harris = Lx**2 + Ly**2 + Lt**2
  
    return det_harris, trace_harris


# Κριτήριο 3D γωνιότητας

# In[10]:


def criterion_Harris(video, sigma, taf, s):
    det , trace = Harris(video, sigma, taf, s)
    cr_image = det - k*np.power(trace,3)
    
    for i in range(cr_image.shape[2]):
        cr_image[:, :, i] = cv2.normalize(cr_image[:, :, i], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
    max_value = np.amax(cr_image)
    cr_image = max_value - cr_image
    
    return cr_image.astype(float)


# In[11]:


cr_Harris_h = criterion_Harris(video_h, sigma, taf, s)
plt.imshow(cr_Harris_h[:,:,25])
plt.show()


# In[12]:


cr_Harris_w = criterion_Harris(video_w, sigma, taf, s)
plt.imshow(cr_Harris_w[:,:,25])
plt.show()


# In[13]:


cr_Harris_r = criterion_Harris(video_r, sigma, taf, s)
plt.imshow(cr_Harris_r[:,:,113])
plt.show()


# In[14]:


# Create an OpenCV window to display the video
cv2.namedWindow("Harris Criterion for handwaving video", cv2.WINDOW_NORMAL)

for frame in range(cr_Harris_h.shape[2]):
    current_frame = cr_Harris_h[:, :, frame]
    current_frame = np.uint8(current_frame)
    
    cv2.imshow("Harris Criterion for handwaving video",current_frame)

    cv2.waitKey(30)

# Close the OpenCV window after displaying all frames
cv2.destroyAllWindows()


# In[15]:


# Create an OpenCV window to display the video
cv2.namedWindow("Harris Criterion for walking video", cv2.WINDOW_NORMAL)

for frame in range(cr_Harris_w.shape[2]):
    current_frame = cr_Harris_w[:, :, frame]
    current_frame = np.uint8(current_frame)
    
    cv2.imshow("Harris Criterion for walking video",current_frame)

    cv2.waitKey(30)

# Close the OpenCV window after displaying all frames
cv2.destroyAllWindows()


# In[16]:


# Create an OpenCV window to display the video
cv2.namedWindow("Harris Criterion for running video", cv2.WINDOW_NORMAL)

for frame in range(cr_Harris_r.shape[2]):
    current_frame = cr_Harris_r[:, :, frame]
    current_frame = np.uint8(current_frame)
    
    cv2.imshow("Harris Criterion for running video",current_frame)

    cv2.waitKey(30)

# Close the OpenCV window after displaying all frames
cv2.destroyAllWindows()


# **2.1.2 Gabor Detector**

# In[17]:


def GaborDetector(taf):
    
    r = np.arange(int(-2*taf),int(2*taf))
    w=4/taf
    
    hev = np.array([np.cos(2*np.pi*t*w)*np.exp(-(t**2)/2*(taf**2)) for t in r])
    hod = np.array([np.sin(2*np.pi*t*w)*np.exp(-(t**2)/2*(taf**2)) for t in r])
    
    #L1 Normalization
    hev_norm = hev/np.linalg.norm(hev,1)
    hod_norm = hod/np.linalg.norm(hod,1)
    
    return hev_norm, hod_norm


# In[18]:


def criterion_Gabor(video, sigma, taf, s):
    hev , hod = GaborDetector(taf)

    ksize_s = int(np.ceil(3*sigma)*2 + 1)
    g_kernel_s = cv2.getGaussianKernel(ksize_s, sigma)
    g_kernel_s2d = g_kernel_s @ g_kernel_s.T

    #Smooth image
    smoothed_images=cv2.filter2D(video[:,:,:],-1,g_kernel_s2d)

    #axis=0 is t    #axis = 1 is y     # axis = 2 is x
    cr_image = np.power(scipy.ndimage.convolve1d(smoothed_images, hev, axis=0, mode='reflect'),2) + np.power(scipy.ndimage.convolve1d(smoothed_images, hod, axis=0, mode='reflect'),2)
    
    for i in range(cr_image.shape[2]):
        cr_image[:, :, i] = cv2.normalize(cr_image[:, :, i], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return cr_image.astype(float)


# In[19]:


cr_Gabor_h = criterion_Gabor(video_h, sigma, taf, s)
plt.imshow(cr_Gabor_h[:,:,25])
plt.show()


# In[20]:


cr_Gabor_w = criterion_Gabor(video_w, sigma, taf, s)
plt.imshow(cr_Gabor_w[:,:,25])
plt.show()


# In[21]:


cr_Gabor_r = criterion_Gabor(video_r, sigma, taf, s)
plt.imshow(cr_Gabor_r[:,:,113])
plt.show()


# In[25]:


# Create an OpenCV window to display the criterion video
cv2.namedWindow("Gabor Criterion for handwaving video", cv2.WINDOW_NORMAL)

for frame in range(cr_Gabor_h.shape[2]):
    current_frame = cr_Gabor_h[:, :, frame]
    current_frame = np.uint8(current_frame)
    cv2.imshow("Gabor Criterion for handwaving video",current_frame)
    cv2.waitKey(30)

# Close the OpenCV window after displaying all frames
cv2.destroyAllWindows()


# In[26]:


# Create an OpenCV window to display the criterion video
cv2.namedWindow("Gabor Criterion for walking video", cv2.WINDOW_NORMAL)

for frame in range(cr_Gabor_w.shape[2]):
    current_frame = cr_Gabor_w[:, :, frame]
    current_frame = np.uint8(current_frame)
    cv2.imshow("Gabor Criterion for walking video",current_frame)
    cv2.waitKey(30)

# Close the OpenCV window after displaying all frames
cv2.destroyAllWindows()


# In[27]:


# Create an OpenCV window to display the criterion video
cv2.namedWindow("Gabor Criterion for running video", cv2.WINDOW_NORMAL)

for frame in range(cr_Gabor_r.shape[2]):
    current_frame = cr_Gabor_r[:, :, frame]
    current_frame = np.uint8(current_frame)
    cv2.imshow("Gabor Criterion for running video",current_frame)
    cv2.waitKey(30)

# Close the OpenCV window after displaying all frames
cv2.destroyAllWindows()


# 2.1.3 Εφαρμογή κατωφλίου για την απόρριψη σχετικά ομαλών περιοχών

# In[28]:


def find_coords(criterion, sigma, theta_corn):
    
    threshold = theta_corn*np.amax(criterion)
    arr=[]
    for x in range(criterion.shape[0]):
        for y in range(criterion.shape[1]):
            for t in range(criterion.shape[2]):
                if(criterion[x][y][t] > threshold):
                    arr.append((x,y,t,sigma))
    arr = np.array(arr)

    return arr


# In[29]:


interest_points_Harris_h = find_coords(cr_Harris_h, sigma, 0.25)
print(interest_points_Harris_h.shape)

interest_points_Harris_r = find_coords(cr_Harris_r, sigma, 0.25)
print(interest_points_Harris_r.shape)

interest_points_Harris_w = find_coords(cr_Harris_w, sigma, 0.25)
print(interest_points_Harris_w.shape)


# In[31]:


interest_points_Gabor_h = find_coords(cr_Gabor_h, sigma, 0.705)
print(interest_points_Gabor_h.shape)

interest_points_Gabor_r = find_coords(cr_Gabor_r, sigma, 0.705)
print(interest_points_Gabor_r.shape)

interest_points_Gabor_w = find_coords(cr_Gabor_w, sigma, 0.705)
print(interest_points_Gabor_w.shape)


# 2.1.4 Σημεία Ενδιαφέροντος ως Τοπικά Μέγιστα του Κριτηρίου Σημαντικότητας

# In[30]:


def final_points(criterion, interest_points, size):
   
    criterion_values = np.empty([interest_points.shape[0],5], dtype = int)
    i = 0
    for x,y,t,s in interest_points:
        criterion_values[i][0] = criterion[int(x),int(y),int(t)]
        criterion_values[i][1] = int(y)
        criterion_values[i][2] = int(x)
        criterion_values[i][3] = int(t)
        criterion_values[i][4] = int(s)
        i += 1 
        
    criterion_flat = criterion_values[:,0].flatten()
    criterion_sorted = criterion_flat.argsort()
    final_indices = criterion_sorted[:size]

    final_coords = np.empty([size,4], dtype = int)
    k=0
    for i in final_indices:
        final_coords[k,[0,1,2,3]] = criterion_values[i,[1,2,3,4]]
        k += 1
    
    return final_coords


# In[32]:


video_h_c = video_h.copy()
video_r_c = video_r.copy()
video_w_c = video_w.copy()


# In[34]:


points_Harris_h = final_points(cr_Harris_h, interest_points_Harris_h, 500)
print(points_Harris_h.shape)

points_Harris_r = final_points(cr_Harris_r, interest_points_Harris_r, 500)
print(points_Harris_r.shape)

points_Harris_w = final_points(cr_Harris_w, interest_points_Harris_w, 500)
print(points_Harris_w.shape)


# In[33]:


points_Gabor_h = final_points(cr_Gabor_h, interest_points_Gabor_h, 500)
print(points_Gabor_h.shape)

points_Gabor_r = final_points(cr_Gabor_r, interest_points_Gabor_r, 500)
print(points_Gabor_r.shape)

points_Gabor_w = final_points(cr_Gabor_w, interest_points_Gabor_w, 500)
print(points_Gabor_w.shape)


# In[35]:


show_detection(video_h_c, points_Harris_h)
show_detection(video_r_c, points_Harris_r)
show_detection(video_w_c, points_Harris_w)


# In[36]:


show_detection(video_h_c, points_Gabor_h)
show_detection(video_r_c, points_Gabor_r)
show_detection(video_w_c, points_Gabor_w)


# Πειραματιζόμαστε με διαφορετικές χωρικές και χρονικές κλίμακες

# In[39]:


cr_Harris_videos = []
for space_scale in range(2,7,1):
    cr_Harris_videos.append(criterion_Harris(video_h_c, space_scale, taf, s))


# In[40]:


space_scale = 2
for cr_Har_scales in cr_Harris_videos:
    interest_points_Har = find_coords(cr_Har_scales, space_scale, 0.25)
    print("Found ", interest_points_Har.shape[0], "interest points with spatial scale sigma", space_scale )
    points_Har = final_points(cr_Har_scales, interest_points_Har, 500)
    show_detection(video_h_c, points_Har)
    space_scale += 1


# In[41]:


cr_Harris_videos = []
for time_scale in np.arange(1.0, 3, 0.5):
    cr_Harris_videos.append(criterion_Harris(video_h_c, sigma, time_scale, s))


# In[43]:


time_scale=1.0
for cr_Har_scales in cr_Harris_videos:
    interest_points_Har = find_coords(cr_Har_scales, sigma, 0.25)
    print("Found ", interest_points_Har.shape[0], "interest points with time scale taf", time_scale, "and spatial scale sigma", sigma)
    points_Har = final_points(cr_Har_scales, interest_points_Har, 500)
    show_detection(video_h_c, points_Har)
    time_scale += 0.5


# In[44]:


cr_Gabor_videos = []
for time_scale in np.arange(1.0, 3, 0.5):
    for space_scale in np.arange(2,7,1):
        cr_Gabor_videos.append(criterion_Gabor(video_h_c, space_scale, time_scale, s))


# In[45]:


time_scale = 1.0
space_scale = 2
for cr_Gabor_scales in cr_Gabor_videos:
    while space_scale<7 and time_scale<3 :
        interest_points_Gab = find_coords(cr_Gabor_scales, space_scale, 0.705)
        print("Found ", interest_points_Gab.shape[0], "interest points with time scale taf", time_scale, "and spatial scale sigma", space_scale)
        points_Gab = final_points(cr_Gabor_scales, interest_points_Gab, 500)
        show_detection(video_h_c, points_Gab)
        space_scale += 1
        if space_scale == 7 :
            time_scale += 0.5
            space_scale = 2
            break


# ### **2.2 Χωρο-χρονικοί Ιστογραφικοί Περιγραφητές**

# **2.2.1 Gradient Vector και TV-L1 Optical Flow**

# In[46]:


def get_gradient(video):
    dy, dx, _ = np.gradient(video)
    return (dy,dx)

def get_optical_flow(video):
    
    video = video.astype(np.uint8)
    
    flow_x = np.zeros((video.shape[0], video.shape[1], video.shape[2]))
    flow_y = np.zeros((video.shape[0], video.shape[1], video.shape[2]))
    
    for f in range(video.shape[2]):
        t = f
        if f == video.shape[2]-1:
            t = f-1
                
        temp = cv2.DualTVL1OpticalFlow_create(nscales=1).calc(video[:,:,t], video[:,:,t+1], None)
        flow_x[:,:,t] = temp[:,:,1]
        flow_y[:,:,t] = temp[:,:,0]
        
    return (flow_y,flow_x)


# In[47]:


Gx, Gy = get_gradient(video_h)


# In[48]:


#Plot Gx
frame_index = 25
video_frame_x = Gx[:, :, frame_index]
 
plt.imshow(video_frame_x, cmap='gray')  
plt.axis('off')  
plt.show()


# In[49]:


# Create an OpenCV window to display the video
cv2.namedWindow("Gx Video", cv2.WINDOW_NORMAL)

for frame in range(Gx.shape[2]):
    current_frame = Gx[:, :, frame]
    current_frame = np.uint8(current_frame)
    
    cv2.imshow("Gx Video",current_frame)

    cv2.waitKey(30)

# Close the OpenCV window after displaying all frames
cv2.destroyAllWindows()


# In[50]:


#Plot Gy
video_frame_y = Gy[:, :, frame_index]
plt.imshow(video_frame_y, cmap='gray')  
plt.axis('off')  
plt.show()


# In[51]:


# Create an OpenCV window to display the video
cv2.namedWindow("Gy Video", cv2.WINDOW_NORMAL)

for frame in range(Gy.shape[2]):
    current_frame = Gy[:, :, frame]
    current_frame = np.uint8(current_frame)
    
    cv2.imshow("Gy Video",current_frame)

    cv2.waitKey(30)

# Close the OpenCV window after displaying all frames
cv2.destroyAllWindows()


# In[52]:


flw_y, flw_x = get_optical_flow(video_h)


# In[54]:


#Plot flw_x
video_frame_x = flw_x[:, :, frame_index]
plt.imshow(video_frame_x, cmap='gray')  
plt.axis('off')  
plt.show()


# In[53]:


# Create an OpenCV window to display the video
cv2.namedWindow("Optical Flow x Video", cv2.WINDOW_NORMAL)

for frame in range(flw_x.shape[2]):
    current_frame = flw_x[:, :, frame]
    current_frame = np.uint8(current_frame)
    
    cv2.imshow("Optical Flow x Video",current_frame)

    cv2.waitKey(30)

# Close the OpenCV window after displaying all frames
cv2.destroyAllWindows()


# In[55]:


#Plot flw_y
video_frame_y = flw_y[:, :, frame_index]
plt.imshow(video_frame_y, cmap='gray')  
plt.axis('off')  
plt.show()


# In[56]:


# Create an OpenCV window to display the video
cv2.namedWindow("Optical Flow y Video", cv2.WINDOW_NORMAL)

for frame in range(flw_y.shape[2]):
    current_frame = flw_y[:, :, frame]
    current_frame = np.uint8(current_frame)
    
    cv2.imshow("Optical Flow y Video",current_frame)

    cv2.waitKey(30)

# Close the OpenCV window after displaying all frames
cv2.destroyAllWindows()


# In[59]:


def HOG_desc(video, interest_points, nbins, grid, neighb):
    
    Gx, Gy = get_gradient(video)
    
    Gx_padded = np.zeros([120+2*neighb[0],160+2*neighb[1],200])
    Gy_padded = np.zeros([120+2*neighb[0],160+2*neighb[1],200])

    for i in range (Gx.shape[2]):
        Gx_padded[:,:,i] = np.pad(Gx[:,:,i], ((neighb[0],neighb[0]),(neighb[1], neighb[1])), 'constant')
        Gy_padded[:,:,i] = np.pad(Gy[:,:,i], ((neighb[0],neighb[0]),(neighb[1], neighb[1])), 'constant')
        
    
    hogi = np.zeros([interest_points.shape[0], grid[0]*grid[1]*nbins])
    
    for i in range(int(interest_points.shape[0])):
        
        x=interest_points[i][1]+neighb[0]
        y=interest_points[i][0]+neighb[1]
        t=interest_points[i][2]
        
        Gx_patch = Gx_padded[(x-neighb[0]):(x+neighb[0]+1), (y-neighb[1]):(y+neighb[1]+1), t]
        Gy_patch = Gy_padded[(x-neighb[0]):(x+neighb[0]+1), (y-neighb[1]):(y+neighb[1]+1), t]
        
        HOG = orientation_histogram(Gx_patch, Gy_patch, nbins, grid)
        for j in range(HOG.shape[0]):
            hogi[i][j] = HOG[j]
         
    return hogi


# In[60]:


def HOF_desc(video, interest_points, nbins, grid, neighb):
    
    flw_y, flw_x = get_optical_flow(video)
    
    flw_x_padded = np.zeros([120+2*neighb[0],160+2*neighb[1],200])
    flw_y_padded = np.zeros([120+2*neighb[0],160+2*neighb[1],200])
    
    for i in range (flw_x.shape[2]):
        flw_x_padded[:,:,i] = np.pad(flw_x[:,:,i], ((neighb[0],neighb[0]),(neighb[1], neighb[1])), 'constant')
        flw_y_padded[:,:,i] = np.pad(flw_y[:,:,i], ((neighb[0],neighb[0]),(neighb[1], neighb[1])), 'constant')
    
    hofi = np.zeros([interest_points.shape[0],grid[0]*grid[1]*nbins])
    
    for i in range(int(interest_points.shape[0])):
        
        x=interest_points[i][1]+neighb[0]
        y=interest_points[i][0]+neighb[1]
        t=interest_points[i][2]
        
        Fx_patch = flw_x_padded[(x-neighb[0]):(x+neighb[0]+1), (y-neighb[1]):(y+neighb[1]+1), t]
        Fy_patch = flw_y_padded[(x-neighb[0]):(x+neighb[0]+1), (y-neighb[1]):(y+neighb[1]+1), t]
        
        HOF = orientation_histogram(Fx_patch, Fy_patch, nbins, grid)
        for j in range(HOF.shape[0]):
            hofi[i][j] = HOF[j]
         
    return hofi


# In[61]:


HOG = HOG_desc(video_h, points_Harris_h, 3, np.array([8,8]), np.array([4,sigma]))
HOF = HOF_desc(video_h, points_Harris_h, 3, np.array([8,8]), np.array([4,sigma]))


# In[63]:


print(HOG.shape)
print(HOF.shape)


# In[62]:


def HOG_HOF_desc(video, interest_points, nbins, grid, neighb): 
    hogi_hofi =[]
    
    HOG = HOG_desc(video, interest_points, nbins, grid, neighb)
    HOF = HOF_desc(video, interest_points, nbins, grid, neighb)
    
    for i in range(interest_points.shape[0]):
        HOG_HOF = np.concatenate((HOG[i],HOF[i]),axis=0)
        hogi_hofi.append(HOG_HOF)
    
    return np.array(hogi_hofi)


# In[64]:


HOG_HOF = HOG_HOF_desc(video_h, points_Harris_h, 3, np.array([8,8]), np.array([4,sigma]))


# In[65]:


print(HOG_HOF.shape)


# **2.3: Κατασκευή Bag of Visual Words και χρήση Support Vector Machines για
# την ταξινόμηση δράσεων**

# 2.3.1 Χωρίζουμε το σύνολο των βίντεο σε train_set και test_set

# In[66]:


with open("C:\\Users\\chris\\Desktop\\cv23_lab2_material\\part2 - SpatioTemporal\\traininng_videos.txt", 'r') as file:
    train_set=[]
    for line in file:
        train_set.append(line.rstrip("\n"))


# In[68]:


path1 = 'C:\\Users\\chris\\Desktop\\cv23_lab2_material\\part2 - SpatioTemporal\\running\\'
path2 = 'C:\\Users\\chris\\Desktop\\cv23_lab2_material\\part2 - SpatioTemporal\\handwaving\\'
path3 = 'C:\\Users\\chris\\Desktop\\cv23_lab2_material\\part2 - SpatioTemporal\\walking\\'

paths_list = [path1, path2, path3] 


# In[69]:


desc_train = np.zeros(36, dtype = object)
train_labels = []
index = 0

for video_sample in train_set:
    if index < 12 :
        vid = read_video(str(Path(os.path.join(path1, video_sample))), 200, 0)
        print(video_sample)
        train_labels.append("running")
        desc_train[index] = vid
        index += 1
    elif index >= 12 and index < 24 :
        vid = read_video(str(Path(os.path.join(path2, video_sample))), 200, 0)
        print(video_sample)
        train_labels.append("handwaving")
        desc_train[index] = vid
        index += 1
    else:
        vid =  read_video(str(Path(os.path.join(path3, video_sample))), 200, 0)
        print(video_sample)
        train_labels.append("walking")
        desc_train[index] = vid
        index += 1


# In[71]:


desc_test = np.zeros(12, dtype = object)
index = 0

for path in paths_list :
    length = 0
    if length < 4 :
        videos_list = os.listdir(path + "\\")
        for video_sample in videos_list:
            if video_sample not in train_set:
                vid = read_video(str(Path(os.path.join(path, video_sample))), 200, 0)
                desc_test[index] = vid
                length = len(desc_test)
                index += 1


# In[70]:


test_labels = []
for i in range(12) :
    if i<= 3 :
        test_labels.append("running")
    if i>3 and i<=7 :
        test_labels.append("handwaving")
    if i>7 and i <= 11 :
        test_labels.append("walking")


# 2.3.2 Υπολογισμός της τελικής αναπαράστασης κάθε βίντεο με την BoW

# In[72]:


train_list_Harris_HOG = []

#fill the empty test_list_HOG

for i in range(desc_train.shape[0]):
    video_train = desc_train[i]
    
    cr_HarrisS_1 = criterion_Harris(video_train, sigma, taf, s)
    all_points_HarrisS_1 = find_coords(cr_HarrisS_1, sigma, 0.1)
    points_HarrisS_1 = final_points(cr_HarrisS_1, all_points_HarrisS_1, 500)
    
    HOG_Harris_train = HOG_desc(video_train, points_HarrisS_1, 3, np.array([8,8]), np.array([4, sigma]))
    
    train_list_Harris_HOG.append(HOG_Harris_train)
    
    print(i) 


# In[75]:


train_list_Harris_HOF = []
#fill the empty test_list_HOF
for i in range(desc_train.shape[0]):
    video_train = desc_train[i]
    
    cr_HarrisS_2 = criterion_Harris(video_train, sigma, taf, s)
    all_points_HarrisS_2 = find_coords(cr_HarrisS_2, sigma, 0.1)
    points_HarrisS_2 = final_points(cr_HarrisS_2, all_points_HarrisS_2, 500)
    
    HOF_Harris_train = HOF_desc(video_train, points_HarrisS_2, 3, np.array([8,8]), np.array([4, sigma]))
    
    train_list_Harris_HOF.append(HOF_Harris_train)
    
    print(i) 


# In[79]:


train_list_Harris_HOG_HOF = []

#fill the empty test_list_HOG_HOF

for i in range(desc_train.shape[0]):
    video_train = desc_train[i]
    
    cr_HarrisS_3 = criterion_Harris(video_train, sigma, taf, s)
    all_points_HarrisS_3 = find_coords(cr_HarrisS_3, sigma, 0.1)
    points_HarrisS_3 = final_points(cr_HarrisS_3, all_points_HarrisS_3, 500)
    
    HOG_HOF_Harris_train = HOG_HOF_desc(video_train, points_HarrisS_3, 3, np.array([8,8]), np.array([4, sigma]))
    
    train_list_Harris_HOG_HOF.append(HOG_HOF_Harris_train)
    
    print(i) 


# In[80]:


test_list_Harris_HOG = []

#fill the empty test_list_HOG

for i in range(desc_test.shape[0]):
    video_test = desc_test[i]
    
    cr_HarrisS_4 = criterion_Harris(video_test, sigma, taf, s)
    all_points_HarrisS_4 = find_coords(cr_HarrisS_4, sigma, 0.1)
    points_HarrisS_4 = final_points(cr_HarrisS_4, all_points_HarrisS_4, 500)
    
    HOG_Harris_test = HOG_desc(video_test, points_HarrisS_4, 3, np.array([8,8]), np.array([4, sigma]))
    
    test_list_Harris_HOG.append(HOG_Harris_test)
    
    print(i) 


# In[82]:


test_list_Harris_HOF = []
#fill the empty test_list_HOF
for i in range(desc_test.shape[0]):
    video_test = desc_test[i]
    
    cr_HarrisS_5 = criterion_Harris(video_test, sigma, taf, s)
    all_points_HarrisS_5 = find_coords(cr_HarrisS_5, sigma, 0.1)
    points_HarrisS_5 = final_points(cr_HarrisS_5, all_points_HarrisS_5, 500)
    
    HOF_Harris_test = HOF_desc(video_test, points_HarrisS_5, 3, np.array([8,8]), np.array([4, sigma]))
    
    test_list_Harris_HOF.append(HOF_Harris_test)
    
    print(i) 


# In[83]:


test_list_Harris_HOG_HOF = []

#fill the empty test_list_HOG_HOF

for i in range(desc_test.shape[0]):
    video_test = desc_test[i]
    
    cr_HarrisS_6 = criterion_Harris(video_test, sigma, taf, s)
    all_points_HarrisS_6 = find_coords(cr_HarrisS_6, sigma, 0.1)
    points_HarrisS = final_points(cr_HarrisS_6, all_points_HarrisS_6, 500)
    
    HOG_HOF_Harris_test = HOG_HOF_desc(video_test, points_HarrisS, 3, np.array([8,8]), np.array([4, sigma]))
    
    test_list_Harris_HOG_HOF.append(HOG_HOF_Harris_test)
    
    print(i) 


# In[88]:


#HOG Descriptor

#train
train_Harris_HOG_labeled = np.empty([len(train_list_Harris_HOG),2],dtype=object) 
train_Harris_HOG_videos = []
for i in range(len(train_list_Harris_HOG)):
        train_Harris_HOG_labeled[i][0] = train_labels[i]
        train_Harris_HOG_labeled[i][1] = train_list_Harris_HOG[i]
        
        train_Harris_HOG_videos.append(train_Harris_HOG_labeled[i][1])
        
#test        
test_Harris_HOG_labeled = np.empty([len(test_list_Harris_HOG),2],dtype=object)  
test_Harris_HOG_videos = []
for i in range(len(test_list_Harris_HOG)):
        test_Harris_HOG_labeled[i][0] = test_labels[i]
        test_Harris_HOG_labeled[i][1] = test_list_Harris_HOG[i]
        
        test_Harris_HOG_videos.append(test_Harris_HOG_labeled[i][1])
        
#HOF Descriptor

#train
train_Harris_HOF_labeled = np.empty([len(train_list_Harris_HOF),2],dtype=object)
train_Harris_HOF_videos = []
for i in range(len(train_list_Harris_HOF)):
        train_Harris_HOF_labeled[i][0] = train_labels[i]
        train_Harris_HOF_labeled[i][1] = train_list_Harris_HOF[i]
        
        train_Harris_HOF_videos.append(train_Harris_HOF_labeled[i][1])
        
#test
test_Harris_HOF_labeled = np.empty([len(test_list_Harris_HOF),2],dtype=object) 
test_Harris_HOF_videos = []
for i in range(len(test_list_Harris_HOF)):
        test_Harris_HOF_labeled[i][0] = test_labels[i]
        test_Harris_HOF_labeled[i][1] = test_list_Harris_HOF[i]
        
        test_Harris_HOF_videos.append(test_Harris_HOF_labeled[i][1])
        
#HOG_HOF Descriptor

#train
train_Harris_HOG_HOF_labeled = np.empty([len(train_list_Harris_HOG_HOF),2],dtype=object)  
train_Harris_HOG_HOF_videos = []
for i in range(len(train_list_Harris_HOG_HOF)):
        train_Harris_HOG_HOF_labeled[i][0] = train_labels[i]
        train_Harris_HOG_HOF_labeled[i][1] = train_list_Harris_HOG_HOF[i]
        
        train_Harris_HOG_HOF_videos.append(train_Harris_HOG_HOF_labeled[i][1])
        
#test              
test_Harris_HOG_HOF_labeled = np.empty([len(test_list_Harris_HOG_HOF),2],dtype=object) 
test_Harris_HOG_HOF_videos = []
for i in range(len(test_list_Harris_HOG_HOF)):
        test_Harris_HOG_HOF_labeled[i][0] = test_labels[i]
        test_Harris_HOG_HOF_labeled[i][1] = test_list_Harris_HOG_HOF[i]
        
        test_Harris_HOG_HOF_videos.append(test_Harris_HOG_HOF_labeled[i][1])


# In[89]:


bow_train_Harris_HOG, bow_test_Harris_HOG = bag_of_words(train_Harris_HOG_videos, test_Harris_HOG_videos, num_centers=3)
bow_train_Harris_HOF, bow_test_Harris_HOF = bag_of_words(train_Harris_HOF_videos, test_Harris_HOF_videos, num_centers=3)
bow_train_Harris_HOG_HOF, bow_test_Harris_HOG_HOF = bag_of_words(train_Harris_HOG_HOF_videos, test_Harris_HOG_HOF_videos, num_centers=3)


# 2.3.3 Κατηγοριοποίηση με SVM 

# In[90]:


accuracy_Harris_HOG, pred_Harris_HOG = svm_train_test(bow_train_Harris_HOG, train_labels, bow_test_Harris_HOG, test_labels)
accuracy_Harris_HOF, pred_Harris_HOF = svm_train_test(bow_train_Harris_HOF, train_labels, bow_test_Harris_HOF, test_labels)
accuracy_Harris_HOG_HOF, pred_Harris_HOG_HOF = svm_train_test(bow_train_Harris_HOG_HOF, train_labels, bow_test_Harris_HOG_HOF, test_labels)


# In[91]:


print("Accuracy of classification using BoW of Harris Detector with HOG descriptors:",accuracy_Harris_HOG, '\n', "")
print("Predictions of classification using BoW of Harris Detector with HOG descriptors:", '\n' , pred_Harris_HOG.tolist(), '\n', "")
print("Real labels of test samples:", '\n', test_labels)


# In[92]:


print("Accuracy of classification using BoW og Harris Detector with HOF descriptors:",accuracy_Harris_HOF, '\n', "")
print("Predictions of classification using BoW of Harris Detector with HOF descriptors:", '\n' , pred_Harris_HOF.tolist(), '\n', "")
print("Real labels of test samples:", '\n', test_labels)


# In[93]:


print("Accuracy of classification using BoW of Harris Detector with HOG_HOF descriptors:",accuracy_Harris_HOG_HOF, '\n', "")
print("Predictions of classification using BoW of Harris Detector with HOG descriptors:", '\n' , pred_Harris_HOG_HOF.tolist(), '\n', "")
print("Real labels of test samples:", '\n', test_labels)


# 2.3.4

# Εκτελούμε τα ίδια βήματα και για τον Gabor ανιχνευτή μας

# In[94]:


train_list_Gabor_HOG = []
#fill the empty test_list_HOG
for i in range(desc_train.shape[0]):
    video_train = desc_train[i]
    
    cr_GaborR_1 = criterion_Gabor(video_train, sigma, taf, s)
    all_points_GaborR_1 = find_coords(cr_GaborR_1, sigma, 0.1)
    points_GaborR_1 = final_points(cr_GaborR_1, all_points_GaborR_1, 500)
    
    
    HOG_Gabor_train = HOG_desc(video_train, points_GaborR_1, 3, np.array([8,8]), np.array([4, sigma]))
    
    train_list_Gabor_HOG.append(HOG_Gabor_train)
    
    print(i) 


# In[95]:


test_list_Gabor_HOG = []
#fill the empty test_list_HOG
for i in range(desc_test.shape[0]):
    video_test = desc_test[i]
    
    cr_GaborR_2 = criterion_Gabor(video_train, sigma, taf, s)
    all_points_GaborR_2 = find_coords(cr_GaborR_2, sigma, 0.1)
    points_GaborR_2 = final_points(cr_GaborR_2, all_points_GaborR_2, 500)
    
    HOG_Gabor_test = HOG_desc(video_train, points_GaborR_2, 3, np.array([8,8]), np.array([4, sigma]))
    
    test_list_Gabor_HOG.append(HOG_Gabor_test)
    
    print(i) 


# In[97]:


#HOG Descriptor

#train
train_Gabor_HOG_labeled = np.empty([len(train_list_Gabor_HOG),2],dtype=object) 
train_Gabor_HOG_videos = []
for i in range(len(train_list_Gabor_HOG)):
        train_Gabor_HOG_labeled[i][0] = train_labels[i]
        train_Gabor_HOG_labeled[i][1] = train_list_Gabor_HOG[i]
        
        train_Gabor_HOG_videos.append(train_Gabor_HOG_labeled[i][1])
        
#test        
test_Gabor_HOG_labeled = np.empty([len(test_list_Gabor_HOG),2],dtype=object)  
test_Gabor_HOG_videos = []
for i in range(len(test_list_Harris_HOG)):
        test_Gabor_HOG_labeled[i][0] = test_labels[i]
        test_Gabor_HOG_labeled[i][1] = test_list_Gabor_HOG[i]
        
        test_Gabor_HOG_videos.append(test_Gabor_HOG_labeled[i][1])


# In[ ]:


#Gabor HOG
bow_train_Gabor_HOG, bow_test_Gabor_HOG = bag_of_words(train_Gabor_HOG_videos, test_Gabor_HOG_videos, num_centers=3)
accuracy_Gabor_HOG, pred_Gabor_HOG = svm_train_test(bow_train_Gabor_HOG, train_labels, bow_test_Gabor_HOG, test_labels)

print("Accuracy of classification using BoW of Gabor Detector with HOG descriptors:",accuracy_Gabor_HOG, '\n', "")
print("Predictions of classification using BoW of Gabor Detector with HOG descriptors:", '\n' , pred_Gabor_HOG.tolist(), '\n', "")
print("Real labels of test samples:", '\n', test_labels)


# In[ ]:


train_list_Gabor_HOF = []

#fill the empty test_list_HOF

for i in range(desc_train.shape[0]):
    video_train = desc_train[i]
    
    cr_GaborR_3 = criterion_Gabor(video_train, sigma, taf, s)
    all_points_GaborR_3 = find_coords(cr_GaborR_3, sigma, 0.1)
    points_GaborR_3 = final_points(cr_GaborR_3, all_points_GaborR_3, 500)
    
    HOF_Gabor_train = HOF_desc(video_train, points_GaborR_3, 3, np.array([8,8]), np.array([4, sigma]))
    
    train_list_Gabor_HOF.append(HOF_Gabor_train)
    
    print(i) 


# In[64]:


test_list_Gabor_HOF = []
#fill the empty test_list_HOF
for i in range(desc_test.shape[0]):
    video_test = desc_test[i]
    
    cr_GaborR_4 = criterion_Gabor(video_test, sigma, taf, s)
    all_points_GaborR_4 = find_coords(cr_GaborR_4, sigma, 0.1)
    points_GaborR = final_points(cr_GaborR_4, all_points_GaborR_4, 500)
    
    HOF_Gabor_test = HOF_desc(video_test, points_GaborR_4, 3, np.array([8,8]), np.array([4, sigma]))
    
    test_list_Gabor_HOF.append(HOF_Gabor_test)
    
    print(i) 


# In[ ]:


#HOF Descriptor

#train
train_Gabor_HOF_labeled = np.empty([len(train_list_Gabor_HOF),2],dtype=object)
train_Gabor_HOF_videos = []
for i in range(len(train_list_Gabor_HOF)):
        train_Gabor_labeled[i][0] = train_labels[i]
        train_Gabor_labeled[i][1] = train_list_Gabor_HOF[i]
        
        train_Harris_HOF_videos.append(train_Gabor_HOF_labeled[i][1])
        
#test
test_Gabor_HOF_labeled = np.empty([len(test_list_Gabor_HOF),2],dtype=object) 
test_Gabor_HOF_videos = []
for i in range(len(test_list_Gabor_HOF)):
        test_Gabor_HOF_labeled[i][0] = test_labels[i]
        test_Gabor_HOF_labeled[i][1] = test_list_Gabor_HOF[i]
        
        test_Gabor_HOF_videos.append(test_Gabor_HOF_labeled[i][1])


# In[ ]:


bow_train_Gabor_HOF, bow_test_Gabor_HOF = bag_of_words(train_Gabor_HOF_videos, test_Gabor_HOF_videos, num_centers=3)
accuracy_Gabor_HOF, pred_Gabor_HOF = svm_train_test(bow_train_Gabor_HOF, train_labels, bow_test_Gabor_HOF, test_labels)

print("Accuracy of classification using BoW og Gabor Detector with HOF descriptors:",accuracy_Gabor_HOF, '\n', "")
print("Predictions of classification using BoW og Gabor Detector with HOF descriptors:", '\n' , pred_Gabor_HOF.tolist(), '\n', "")
print("Real labels of test samples:", '\n', test_labels)


# In[70]:


train_list_Gabor_HOG_HOF = []
#fill the empty test_list_HOG_HOF
for i in range(desc_train.shape[0]):
    video = desc_train[i]
    
    cr_GaborR_5 = criterion_Gabor(video, sigma, taf, s)
    all_points_GaborR_5 = find_coords(cr_GaborR_5, sigma, 0.1)
    points_GaborR_5 = final_points(cr_GaborR_5, all_points_GaborR_5, 500)
    
    HOG_HOF_Gabor_train = HOG_HOF_desc(video, points_GaborR_5, 3, n.array([8,8]), np.array([4, sigma]))
    
    train_list_Gabor_HOG_HOF.append(HOG_HOF_Gabor_train)
    
    print(i) 


# In[72]:


test_list_Gabor_HOG_HOF = []
#fill the empty test_list_HOG_HOF
for i in range(desc_test.shape[0]):
    video_test = desc_test[i]
    
    cr_GaborR_6 = criterion_Gabor(video_test, sigma, taf, s)
    all_points_GaborR_6 = find_coords(cr_GaborR_6, sigma, 0.1)
    points_GaborR_6 = final_points(cr_GaborR_6, all_points_GaborR_6, 500)
    
    HOG_HOF_Gabor_test = HOG_HOF_desc(video_test, points_GaborR_6, 3, np.array([8,8]), np.array([4, sigma]))
    
    test_list_Gabor_HOG_HOF.append(HOG_HOF_Gabor_test)
    
    print(i) 


# In[ ]:


#HOG_HOF Descriptor

#train
train_Gabor_HOG_HOF_labeled = np.empty([len(train_list_Gabor_HOG_HOF),2],dtype=object)  
train_Gabor_HOG_HOF_videos = []
for i in range(len(train_list_Harris_HOG_HOF)):
        train_Gabor_HOG_HOF_labeled[i][0] = train_labels[i]
        train_Gabor_HOG_HOF_labeled[i][1] = train_list_Harris_HOG_HOF[i]
        
        train_Harris_HOG_HOF_videos.append(train_Gabor_HOG_HOF_labeled[i][1])
        
#test              
test_Gabor_HOG_HOF_labeled = np.empty([len(test_list_Gabor_HOG_HOF),2],dtype=object) 
test_Gabor_HOG_HOF_videos = []
for i in range(len(test_list_Gabor_HOG_HOF)):
        test_Gabor_HOG_HOF_labeled[i][0] = test_labels[i]
        test_Gabor_HOG_HOF_labeled[i][1] = test_list_Gabor_HOG_HOF[i]
        
        test_Gabor_HOG_HOF_videos.append(test_Gabor_HOG_HOF_labeled[i][1])


# In[ ]:


bow_train_Gabor_HOG_HOF, bow_test_Gabor_HOG_HOF = bag_of_words(train_Gabor_HOG_HOF_videos, test_Gabor_HOG_HOF_videos, num_centers=3)
accuracy_Gabor_HOG_HOF, pred_Gabor_HOG_HOF = svm_train_test(bow_train_Gabor_HOG_HOF, train_labels, bow_test_Gabor_HOG_HOF, test_labels)

print("Accuracy of classification using BoW with HOG_HOF descriptors:",accuracy_Gabor_HOG_HOF, '\n', "")
print("Predictions of classification using BoW with HOG descriptors:", '\n' , pred_Gabor_HOG_HOF.tolist(), '\n', "")
print("Real labels of test samples:", '\n', test_labels)

