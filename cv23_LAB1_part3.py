#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2

import matplotlib.pyplot as plt

import numpy as np

import math

import sys

from matplotlib.patches import Circle

from scipy import ndimage

from sklearn.preprocessing import MinMaxScaler


# In[2]:


np.version.version


# In[34]:


import sys  
sys.path.insert(0,'D:\\Επιφάνεια εργασίας\\cv23_lab1_part3_material')

from cv23_lab1_part3_utils import matching_evaluation
from cv23_lab1_part3_utils import featuresSURF
from cv23_lab1_part3_utils import featuresHOG
from cv23_lab1_part3_utils import FeatureExtraction
from cv23_lab1_part3_utils import createTrainTest
from cv23_lab1_part3_utils import svm
from cv23_lab1_part3_utils import BagOfWords


# **Μέρος 3: Εφαρμογές σε Ταίριασμα και Κατηγοριοποίηση Εικόνων με Χρήση Τοπικών Περιγραφητών στα Σημεία Ενδιαφέροντος**

# **3.1. Ταίριασμα Εικόνων υπό Περιστροφή και Αλλαγή Κλίμακας**
# 
# 3.1.1

# In[4]:


#Functions for creating LoG kernel
def LoG(sigma):
  
  n = int(2*np.ceil(3*sigma)+1)
  ax = np.linspace(-n/2, n/2, n)
    
  xx, yy = np.meshgrid(ax, ax)
    
  kernel = ((-1)/(np.pi*sigma**4))*((1-(xx**2+yy**2)/(2*sigma**2)))* np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
  return kernel


def kernels(sigma):
  """This function takes as input the variance value and returns
  a gaussian kernel and a LoG kernel created with this sigma value
  """

  #Dimensions of the kernel should be odd number 
  ksize = int(2*np.ceil(3*sigma)+1) 

  # Creates a 2-D Gaussian kernel
  gauss_kernel = cv2.getGaussianKernel(ksize, sigma)
  gauss_2d = gauss_kernel @ gauss_kernel.T

  log_kernel = LoG(sigma)
  
  return gauss_2d, log_kernel


# In[5]:


def laplaceImage(analysis,sigma, g_img):
  #Creating kernels based on sigma
  gaussian, log = kernels(sigma)

  """If analysis is linear the Laplacian is applied via the LoG kernel 
  with a linear approach, where if analysis is non_linear, the Laplacian is obtained using
  morphological filters"""
  
  if analysis == "linear":
    # Applying the filter2D() function
    L1 = cv2.filter2D(src=g_img, ddepth=-1, kernel=log)
    L = L1

  elif analysis == "non_linear":
    smooth_img = cv2.filter2D(g_img,-1,gaussian)
    # Create morphological kernel
    kernel = np.array([
        [0,1,0],
        [1,1,1],
        [0,1,0]
], dtype=np.uint8)

    #Functions
    img_erosion = cv2.erode(smooth_img, kernel)
    img_dilation = cv2.dilate(smooth_img, kernel)

    L2 = img_erosion + img_dilation - (smooth_img*2)
    L = L2

  return L


# In[6]:


#Creating tensor J for an image I

def tensor_J(image, s, r):
  """This function returns elements J1,J2,J3 of tensor J of an image"""
  
  #Defining gaussian kernels
  ksize_s = int(np.ceil(3*s)*2 + 1)
  ksize_r = int(np.ceil(3*r)*2 + 1)

  g_kernel_s = cv2.getGaussianKernel(ksize_s, s)
  g_kernel_r = cv2.getGaussianKernel(ksize_r, r)

  #Smoothing images
  image_smoothed = cv2.filter2D(image, -1, g_kernel_s)

  #Calculating gradients
  Ix, Iy = np.gradient(image_smoothed)

  first_mult= np.multiply(Ix, Ix)
  second_mult= np.multiply(Ix,Iy)
  third_mult= np.multiply(Iy,Iy)
  
  J1= cv2.filter2D(first_mult, -1, g_kernel_r)
  J2= cv2.filter2D(second_mult, -1, g_kernel_r)
  J3= cv2.filter2D(third_mult, -1, g_kernel_r)

  return J1,J2,J3


# In[7]:


def lamda_values(J1,J2,J3):
  #Returns the eigenvalues
  lamda_plus= 0.5*np.add(np.add(J1,J3),np.sqrt(np.add(np.power((np.subtract(J1,J3)),2),4*np.power(J2,2))))

  lamda_minus= 0.5*np.subtract(np.add(J1,J3),np.sqrt(np.add(np.power((np.subtract(J1,J3)),2),4*np.power(J2,2))))

  return lamda_plus, lamda_minus


# In[8]:


def produce_scales(sigma_zero, r_zero, s, N):
  #Produces σ and r for multiple scales
  sigma_scales = []
  r_scales = []
  for i in range(N):
    sigma_scales.append(math.pow(s, i)*sigma_zero)
    r_scales.append(math.pow(s, i)*r_zero)
  
  return sigma_scales, r_scales


# In[9]:


def disk_strel(n):
    '''
        Return a structural element, which is a disk of radius n.
    '''
    r = int(np.round(n))
    d = 2*r+1
    x = np.arange(d) - r
    y = np.arange(d) - r
    x, y = np.meshgrid(x,y)
    strel = x**2 + y**2 <= r**2
    return strel.astype(np.uint8)


# In[10]:


def find_coords(S, R, criterion):
  """This function takes as arguments the initial corner criterion of an image and 
  outputs the coordinates of the points that satisfy both Σ1 and Σ2"""

  arr=[]
  for x in range(S.shape[0]):
    for y in range(S.shape[1]):
      if (R[x][y] == S[x][y]) and (R[x][y] > criterion):
        arr.append((x,y))
  #arr = np.array(arr,dtype=object)

  return np.array(arr)


def fill_array(coordinates,sigma):
  """This function returns an array filled with the coordinates and the scale needed
  for the visulization function"""

  array=np.zeros((coordinates.shape[0],3))
  for x in range(coordinates.shape[0]):
    array[x] = (coordinates[x][0], coordinates[x][1], sigma) 

  return np.array(array,dtype=object)


def corner_coordinates(image, s, r, k, theta_corn):
  """This function, returns an array which contains the coordinates of the corner points
  of the input image on scale s,r along with the scale on which they were detected"""


  #Creating eigenvalues for the input image
  J1, J2, J3 = tensor_J(image, s, r)
  lamda_plus, lamda_minus = lamda_values(J1,J2,J3)

  #Creating corner criterion
  R = lamda_minus*lamda_plus - k*np.square(lamda_minus+lamda_plus)

  #R= (np.subtract((np.multiply(lamda[1],lamda[0])),(k*np.power(np.add(lamda[1],lamda[0]),2))))

  ns = int(np.ceil(3*s)*2+1)
  B_sq = disk_strel(ns)

  #Condition 1
  cond = cv2.dilate(R, B_sq)

  criterion = theta_corn*np.amax(R)

  #Coordinates of point that satisft both criterion
  coords = find_coords(cond,R,criterion)

  #Array for visualization
  arr = fill_array(coords, s) 

  return np.array(arr,dtype='object')


# In[11]:


def reject_corners(img, sigma, r, k, theta_corn):
    
  s = 1.5
  N = 4
    
  sigma_scale, r_scale = produce_scales(sigma, r, s, N)

  log = []
  for s in (sigma_scale):
      log.append(laplaceImage('linear', s, img)*np.power(s,2)) 
        
        
  scale_coords = []      
  for i in range(len(sigma_scale)):
      scale_coords.append(corner_coordinates(img, sigma_scale[i], r_scale[i], k, theta_corn))
    
  scale_coords = np.array(scale_coords,dtype=object)    
    

  true_corners=[]
  for i in range(len(scale_coords)):
    for j in range(scale_coords[i].shape[0]):
      x = int(scale_coords[i][j][0])
      y = int(scale_coords[i][j][1])
      if i == 0:
        if log[i][x][y] > log[i+1][x][y]:
          true_corners.append((x, y, scale_coords[i][j][2]))

      elif i > 0 and i < (len(scale_coords)-1):
        if log[i][x][y] > log[i-1][x][y] and log[i][x][y] > log[i+1][x][y]:
          true_corners.append((x, y, scale_coords[i][j][2]))

      elif i == (len(scale_coords)-1):
        if log[i][x][y] > log[i-1][x][y]:
          true_corners.append((x, y, scale_coords[i][j][2]))

  return np.array(true_corners,dtype=object)


# In[12]:


def Hessian(image, sigma):
  """This function returns the det of Hessian matrix of input image"""


  #First we smooth the image
  ksize_s = int(np.ceil(3*sigma)*2 + 1)
  g_kernel_s = cv2.getGaussianKernel(ksize_s, sigma)

  smooth_img = cv2.filter2D(image,-1,g_kernel_s)

  gradient1 = np.gradient(smooth_img)
  gradient2 = np.gradient(gradient1[0])
  gradient3 = np.gradient(gradient1[1])

  partial_xx = gradient2[0]
  partial_xy = gradient2[1]
  partial_yy = gradient3[1]

  hessian = np.subtract(np.multiply(partial_xx,partial_yy), np.power(partial_xy,2))
  
  return hessian


# In[13]:


def find_blobs(image, s, r, k, theta_corn):
  """Returns the points of the blobs"""
  
  #Finds blobs

  R = Hessian(image, s)

  ns = int(np.ceil(3*s)*2+1)
  B_sq = disk_strel(ns)

  #Condition 1
  cond = cv2.dilate(R, B_sq)

  criterion= theta_corn*R.max()

  #Coordinates of point that satisft both criterion
  coords = find_coords(cond,R,criterion)

  #Array for visualization
  arr= fill_array(coords, s) 

  return arr


# In[14]:


def reject_blobs(img, sigma, r, k, theta_corn):
    
  s = 1.5
  N = 4
    
  sigma_scale, r_scale = produce_scales(sigma, r, s, N)

  log = []
  for s in (sigma_scale):
      log.append(laplaceImage('linear', s, img)*np.power(s,2)) 
        
        
  blob_coords = []      
  for i in range(len(sigma_scale)):
      blob_coords.append(find_blobs(img, sigma_scale[i], r_scale[i], k, theta_corn))
    
  blob_coords = np.array(blob_coords,dtype=object)
    
  true_blobs=[]
  for i in range(len(blob_coords)):
    for j in range(blob_coords[i].shape[0]):
      x = int(blob_coords[i][j][0])
      y = int(blob_coords[i][j][1])
      if i == 0:
        if log[i][x][y] > log[i+1][x][y]:
          true_blobs.append((x, y, blob_coords[i][j][2]))

      if i > 0 and i < (len(blob_coords)-1):
        if log[i][x][y] > log[i-1][x][y] and log[i][x][y] > log[i+1][x][y]:
          true_blobs.append((x, y, blob_coords[i][j][2]))

      if i == (len(blob_coords)-1):
        if log[i][x][y] > log[i-1][x][y]:
          true_blobs.append((x, y, blob_coords[i][j][2]))

  return np.array(true_blobs,dtype=object)


# In[15]:


def integral(img):
    
    row = img.shape[0]
    col = img.shape[1]
    #integral image
    integral = np.zeros((img.shape[0],img.shape[1]))

    for i in range(0,row):
        for j in range(0,col):
            if i == 0 and j==0:
                integral[i][j] = img[i][j]
            elif i == 0:
                integral[i][j] = integral[i][j-1] + img[i][j]
            elif j==0:
                integral[i][j] = integral[i-1][j] + img[i][j]
            else:
                integral[i][j] = integral[i-1][j] + integral[i][j-1] - integral[i-1][j-1] + img[i][j]


    return integral


# In[16]:


def sum_points(i, j, integral, filter_type,s, padding_up, padding_left):
  """Returns the sum of a subarea of an image using the integral of the
  image through the points of coordinates:
  """
  n = int(2*np.ceil(3*s)+1)
  i = i + padding_up 
  j = j + padding_left

  if filter_type == 'Dxx':
    height = 4*np.floor(n/6)+1
    width = 2*np.floor(n/6)+1

    '''
           up
    left  (i,j)  right
          down
    '''
    down = int(i + np.floor(height/2))
    up = int(i - np.ceil(height/2))
    left = int(j - np.ceil(width/2))
    right = int(j + np.floor(width/2))

    x1, y1 = (down, right)
    x2, y2 = (up, right)
    x3, y3 = (up, left)
    x4, y4 = (down, left)
    x5, y5 = (up, int(left - width))
    x6, y6 = (down, int(left - width))
    x7, y7 = (down, int(right + width))
    x8, y8 = (up, int(right + width))

    first_window_sum = (integral[x4][y4] - integral[x3][y3] - integral[x6][y6] + integral[x5][y5])
    second_window_sum = (integral[x1][y1] - integral[x2][y2] - integral[x4][y4] + integral[x3][y3])
    third_window_sum = (integral[x7][y7] - integral[x8][y8] - integral[x1][y1] + integral[x2][y2])
    
    sum = first_window_sum + (-2)*second_window_sum + third_window_sum

  elif filter_type == 'Dyy':
    height = 2*np.floor(n/6)+1
    width = 4*np.floor(n/6)+1

    down = int(i + np.floor(height/2))
    up = int(i - np.ceil(height/2))
    left = int(j - np.ceil(width/2))
    right = int(j + np.floor(width/2))   

    x1, y1 = (down, right)
    x2, y2 = (up, right)
    x3, y3 = (up, left)
    x4, y4 = (down, left)
    x5, y5 = (int(up - height), right)
    x6, y6 = (int(up - height), left)
    x7, y7 = (int(down + height), right)
    x8, y8 = (int(down + height), left)

    first_window_sum = (integral[x2][y2] - integral[x5][y5] - integral[x3][y3] + integral[x6][y6])
    second_window_sum = (integral[x1][y1] - integral[x2][y2] - integral[x4][y4] + integral[x3][y3])
    third_window_sum = (integral[x7][y7] - integral[x8][y8] - integral[x1][y1] + integral[x4][y4])
    
    sum = first_window_sum + (-2)*second_window_sum + third_window_sum

  elif filter_type == 'Dxy':
    height = 2*np.floor(n/6)+1
    width = 2*np.floor(n/6)+1 

    x1, y1 = (int(i - 1), int(j - 1))
    x2, y2 = (int(i - (height + 1)), int(j-1))
    x3, y3 = (int(i - (height + 1) ), int(j - (width + 1)))
    x4, y4 = (int(i - 1), int(j - (width + 1)))
    x5, y5 = (int(i - 1), int(j + width))
    x6, y6 = (int(i - (height + 1)), int(j + width))
    x7, y7 = (int(i - (height + 1)), int(j))
    x8, y8 = (int(i - 1), int(j))
    x9, y9 = (int(i + height), int(j + width))
    x10, y10 = (int(i), int(j + width))
    x11, y11 = (int(i), int(j))
    x12, y12 = (int(i + height), int(j))
    x13, y13 = (int(i + height), int(j - 1))
    x14, y14 = (int(i), int(j - 1))
    x15, y15 = (int(i), int(j - (width + 1)))
    x16, y16 = (int(i + height), int(j - (width + 1)))

    first_window_sum = (integral[x1][y1] - integral[x2][y2] - integral[x4][y4] + integral[x3][y3])
    second_window_sum = (integral[x5][y5] - integral[x6][y6] - integral[x8][y8] + integral[x7][y7])
    third_window_sum = (integral[x13][y13] - integral[x14][y14] - integral[x16][y16] + integral[x15][y15])
    fourth_window_sum = (integral[x9][y9] - integral[x10][y10] - integral[x12][y12] + integral[x11][y11])

    sum = first_window_sum + (-1)*second_window_sum + (-1)*third_window_sum + fourth_window_sum


  return sum

  


def convolve_with_box(image, image_integral ,filter_type, s):
  """Depending on the filter type={Dxx,Dyy,Dxy} it outputs the 
  Lxx, Lyy and Lxy of the input image, using a box filter and the values of 
  its integral image"""

  n = int(2*np.ceil(3*s) + 1)
  X, Y = image.shape

  L = np.zeros((X , Y))
  

  if filter_type =='Dxx':

    #padding image with zeros for the output to have same size
    padding_up = int(2*np.floor(n/6)+1)
    padding_left = int(4*np.floor(n/6)+1)
    padding_down = int(2*np.floor(n/6)+1)
    padding_right = int(4*np.floor(n/6)+1)

    padded_img = np.pad(image_integral, ((padding_up, padding_down), (padding_left, padding_right)), 'reflect')
  
    for i in range(X):
      for j in range(Y):

          L[i][j] = sum_points(i, j, padded_img,'Dxx',s,padding_up, padding_left)

  elif filter_type =='Dyy':

    #padding image with zeros for the output to have same size
    padding_up = int(4*np.floor(n/6)+1)
    padding_left = int(2*np.floor(n/6)+1)
    padding_down = int(4*np.floor(n/6)+1)
    padding_right = int(2*np.floor(n/6)+1)

    padded_img = np.pad(image_integral, ((padding_up, padding_down), (padding_left, padding_right)), 'reflect')

    for i in range(X):
      for j in range(Y):

        L[i][j] = sum_points(i, j, padded_img,'Dyy',s, padding_up, padding_left)

  elif filter_type =='Dxy':

    #padding image with zeros for the output to have same size
    padding_up = int(2*np.floor(n/6)+2)
    padding_left = int(2*np.floor(n/6)+2)
    padding_down = int(2*np.floor(n/6)+2)
    padding_right = int(2*np.floor(n/6)+2)

    padded_img = np.pad(image_integral, ((padding_up, padding_down), (padding_left, padding_right)), 'reflect')

    for i in range(X):
      for j in range(Y):
      
        L[i][j] = sum_points(i, j, padded_img,'Dxy',s, padding_up, padding_left)

  return L


# In[17]:


def find_blobs_box(image, s, r, k, theta_corn):
  """Returns the points of the blobs"""
  
  # Integral image
  Integral = integral(image)

  #Finds blobs
  Lxx = convolve_with_box(image, Integral, 'Dxx', s)
  Lyy = convolve_with_box(image, Integral, 'Dyy', s)
  Lxy = convolve_with_box(image, Integral, 'Dxy', s)

  Rbox = np.subtract(np.multiply(Lxx,Lyy),0.9*np.power(Lxy,2))

  ns = int(np.ceil(3*s)*2+1)
  B_sq = disk_strel(ns)

  #Condition 1
  cond = cv2.dilate(Rbox, B_sq)

  criterion= theta_corn*Rbox.max()

  #Coordinates of point that satisft both criterion
  coords = find_coords(cond,Rbox,criterion)

  #Array for visualization
  arr= fill_array(coords, s) 

  return arr


# In[18]:


def multiscale_blobs_box(img, sigma, r, k, theta_corn):
    
  s = 1.5
  N = 4
    
  sigma_scale, r_scale = produce_scales(sigma, r, s, N)

  Integral = integral(img)

  blob_coords = []

  for i in range(len(sigma_scale)):
    blob_coords.append(find_blobs_box(img, sigma_scale[i], r_scale[i], k, theta_corn))
    
  log = []
  for s in (sigma_scale):
    Lxx = convolve_with_box(img, Integral, 'Dxx', s)
    Lyy = convolve_with_box(img, Integral, 'Dyy', s)
    Lxy = convolve_with_box(img, Integral, 'Dxy', s)

    log.append(np.power(s,2) * (Lxx + Lyy))

  true_blobs=[]
  for i in range(len(blob_coords)):
    for j in range(blob_coords[i].shape[0]):
      x = int(blob_coords[i][j][0])
      y = int(blob_coords[i][j][1])
      if i == 0:
        if log[i][x][y] > log[i+1][x][y]:
          true_blobs.append((x, y, blob_coords[i][j][2]))

      if i > 0 and i < (len(blob_coords)-1):
        if log[i][x][y] > log[i-1][x][y] and log[i][x][y] > log[i+1][x][y]:
          true_blobs.append((x, y, blob_coords[i][j][2]))

      if i == (len(blob_coords)-1):
        if log[i][x][y] > log[i-1][x][y]:
          true_blobs.append((x, y, blob_coords[i][j][2]))

  return np.array(true_blobs,dtype=object)


# In[19]:


import os
os.chdir('D:\\Επιφάνεια εργασίας\\cv23_lab1_part3_material')


# In[101]:


detect_fun = lambda I: corner_coordinates(I, 2, 2.5, 0.05, 0.005)

desc_fun = lambda I, kp: featuresSURF(I,kp)

#desc_fun = lambda I, kp: featuresHOG(I,kp)


avg_scale_errors, avg_theta_errors = matching_evaluation(detect_fun, desc_fun)
print('Avg. Scale Error for Image 1 for Harris Detector : {:.3f}'.format(avg_scale_errors[0]))
print('Avg. Theta Error for Image 1 for Harris Detector : {:.3f}'.format(avg_theta_errors[0]))

print('Avg. Scale Error for Image 2 for Harris Detector : {:.3f}'.format(avg_scale_errors[1]))
print('Avg. Theta Error for Image 2 for Harris Detector : {:.3f}'.format(avg_theta_errors[1]))

print('Avg. Scale Error for Image 3 for Harris Detector : {:.3f}'.format(avg_scale_errors[2]))
print('Avg. Theta Error for Image 3 for Harris Detector : {:.3f}'.format(avg_theta_errors[2]))


# In[102]:


detect_fun = lambda I: reject_corners(I, 2, 2.5, 0.05, 0.005)

desc_fun = lambda I, kp: featuresSURF(I,kp)

#desc_fun = lambda I, kp: featuresHOG(I,kp)


avg_scale_errors, avg_theta_errors = matching_evaluation(detect_fun, desc_fun)
print('Avg. Scale Error for Image 1 for Harris-Laplacian : {:.3f}'.format(avg_scale_errors[0]))
print('Avg. Theta Error for Image 1 for Harris-Laplacian : {:.3f}'.format(avg_theta_errors[0]))

print('Avg. Scale Error for Image 2 for Harris-Laplacian : {:.3f}'.format(avg_scale_errors[1]))
print('Avg. Theta Error for Image 2 for Harris-Laplacian : {:.3f}'.format(avg_theta_errors[1]))

print('Avg. Scale Error for Image 3 for Harris-Laplacian : {:.3f}'.format(avg_scale_errors[2]))
print('Avg. Theta Error for Image 3 for Harris-Laplacian : {:.3f}'.format(avg_theta_errors[2]))


# In[104]:


detect_fun = lambda I: find_blobs(I, 2, 2.5, 0.05, 0.005)

desc_fun = lambda I, kp: featuresSURF(I,kp)

#desc_fun = lambda I, kp: featuresHOG(I,kp)


avg_scale_errors, avg_theta_errors = matching_evaluation(detect_fun, desc_fun)
print('Avg. Scale Error for Image 1 for Hessian detector : {:.3f}'.format(avg_scale_errors[0]))
print('Avg. Theta Error for Image 1 for Hessian detector : {:.3f}'.format(avg_theta_errors[0]))

print('Avg. Scale Error for Image 2 for Hessian detector : {:.3f}'.format(avg_scale_errors[1]))
print('Avg. Theta Error for Image 2 for Hessian detector : {:.3f}'.format(avg_theta_errors[1]))

print('Avg. Scale Error for Image 3 for Hessian detector : {:.3f}'.format(avg_scale_errors[2]))
print('Avg. Theta Error for Image 3 for Hessian detector : {:.3f}'.format(avg_theta_errors[2]))


# In[105]:


detect_fun = lambda I: reject_blobs(I, 2, 2.5, 0.05, 0.005)

desc_fun = lambda I, kp: featuresSURF(I,kp)

#desc_fun = lambda I, kp: featuresHOG(I,kp)


avg_scale_errors, avg_theta_errors = matching_evaluation(detect_fun, desc_fun)
print('Avg. Scale Error for Image 1 for Hessian-Laplacian : {:.3f}'.format(avg_scale_errors[0]))
print('Avg. Theta Error for Image 1 for Hessian-Laplacian : {:.3f}'.format(avg_theta_errors[0]))

print('Avg. Scale Error for Image 2 for Hessian-Laplacian : {:.3f}'.format(avg_scale_errors[1]))
print('Avg. Theta Error for Image 2 for Hessian-Laplacian : {:.3f}'.format(avg_theta_errors[1]))

print('Avg. Scale Error for Image 3 for Hessian-Laplacian : {:.3f}'.format(avg_scale_errors[2]))
print('Avg. Theta Error for Image 3 for Hessian-Laplacian : {:.3f}'.format(avg_theta_errors[2]))


# In[106]:


detect_fun = lambda I: multiscale_blobs_box(I, 2, 2.5, 0.05, 0.005)

desc_fun = lambda I, kp: featuresSURF(I,kp)

#desc_fun = lambda I, kp: featuresHOG(I,kp)


avg_scale_errors, avg_theta_errors = matching_evaluation(detect_fun, desc_fun)
print('Avg. Scale Error for Image 1 for Multiscale_Box_Filters : {:.3f}'.format(avg_scale_errors[0]))
print('Avg. Theta Error for Image 1 for Multiscale_Box_Filters : {:.3f}'.format(avg_theta_errors[0]))

print('Avg. Scale Error for Image 2 for Multiscale_Box_Filters : {:.3f}'.format(avg_scale_errors[1]))
print('Avg. Theta Error for Image 2 for Multiscale_Box_Filters : {:.3f}'.format(avg_theta_errors[1]))

print('Avg. Scale Error for Image 3 for Multiscale_Box_Filters : {:.3f}'.format(avg_scale_errors[2]))
print('Avg. Theta Error for Image 3 for Multiscale_Box_Filters : {:.3f}'.format(avg_theta_errors[2]))


# In[55]:


detect_fun = lambda I: corner_coordinates(I, 3, 2.5, 0.05, 0.005)

#desc_fun = lambda I, kp: featuresSURF(I,kp)

desc_fun = lambda I, kp: featuresHOG(I,kp)


avg_scale_errors, avg_theta_errors = matching_evaluation(detect_fun, desc_fun)
print('Avg. Scale Error for Image 1 for Harris Detector with HOG : {:.3f}'.format(avg_scale_errors[0]))
print('Avg. Theta Error for Image 1 for Harris Detector with HOG: {:.3f}'.format(avg_theta_errors[0]))

print('Avg. Scale Error for Image 2 for Harris Detector with HOG: {:.3f}'.format(avg_scale_errors[1]))
print('Avg. Theta Error for Image 2 for Harris Detector with HOG: {:.3f}'.format(avg_theta_errors[1]))

print('Avg. Scale Error for Image 3 for Harris Detector with HOG: {:.3f}'.format(avg_scale_errors[2]))
print('Avg. Theta Error for Image 3 for Harris Detector with HOG: {:.3f}'.format(avg_theta_errors[2]))


# **3.2. Κατηγοριοποίηση Εικόνων**

# 3.2.1

# In[36]:


# Change with your own detectors here!
detect_fun = lambda I: reject_corners(I, 2, 2.5, 0.05, 0.005)

desc_fun = lambda I, kp: featuresSURF(I,kp)

# Extract features from the provided dataset.
feats = FeatureExtraction(detect_fun, desc_fun)

# If the above code takes too long, you can use the following extra parameters of Feature extraction:
#   saveFile = <filename>: Save the extracted features in a file with the provided name.
#   loadFile = <filename>: Load the extracted features from a given file (which MUST exist beforehand).


accs = []
for k in range(5):
    # Split into a training set and a test set.
    data_train, label_train, data_test, label_test = createTrainTest(feats, k)

    # Perform Kmeans to find centroids for clusters.
    BOF_tr, BOF_ts = BagOfWords(data_train, data_test)

    # Train an svm on the training set and make predictions on the test set
    acc, preds, probas = svm(BOF_tr, label_train, BOF_ts, label_test)
    accs.append(acc)

print('Mean accuracy for Harris-Laplacian with SURF descriptors: {:.3f}%'.format(100.0*np.mean(accs)))


# In[37]:


# Change with your own detectors here!
detect_fun = lambda I: reject_blobs(I, 2, 2.5, 0.05, 0.005)

desc_fun = lambda I, kp: featuresSURF(I,kp)

# Extract features from the provided dataset.
feats = FeatureExtraction(detect_fun, desc_fun)

# If the above code takes too long, you can use the following extra parameters of Feature extraction:
#   saveFile = <filename>: Save the extracted features in a file with the provided name.
#   loadFile = <filename>: Load the extracted features from a given file (which MUST exist beforehand).


accs = []
for k in range(5):
    # Split into a training set and a test set.
    data_train, label_train, data_test, label_test = createTrainTest(feats, k)

    # Perform Kmeans to find centroids for clusters.
    BOF_tr, BOF_ts = BagOfWords(data_train, data_test)

    # Train an svm on the training set and make predictions on the test set
    acc, preds, probas = svm(BOF_tr, label_train, BOF_ts, label_test)
    accs.append(acc)

print('Mean accuracy for Hessian-Laplacian with SURF descriptors: {:.3f}%'.format(100.0*np.mean(accs)))


# In[38]:


# Change with your own detectors here!
detect_fun = lambda I: multiscale_blobs_box(I, 2, 2.5, 0.05, 0.005)

desc_fun = lambda I, kp: featuresSURF(I,kp)

# Extract features from the provided dataset.
feats = FeatureExtraction(detect_fun, desc_fun)

# If the above code takes too long, you can use the following extra parameters of Feature extraction:
#   saveFile = <filename>: Save the extracted features in a file with the provided name.
#   loadFile = <filename>: Load the extracted features from a given file (which MUST exist beforehand).


accs = []
for k in range(5):
    # Split into a training set and a test set.
    data_train, label_train, data_test, label_test = createTrainTest(feats, k)

    # Perform Kmeans to find centroids for clusters.
    BOF_tr, BOF_ts = BagOfWords(data_train, data_test)

    # Train an svm on the training set and make predictions on the test set
    acc, preds, probas = svm(BOF_tr, label_train, BOF_ts, label_test)
    accs.append(acc)

print('Mean accuracy for Multiscale box filters detection with SURF descriptors: {:.3f}%'.format(100.0*np.mean(accs)))


# In[ ]:




