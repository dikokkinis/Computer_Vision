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

# ## ***Μέρος 1: Παρακολούθηση Προσώπου και Χεριών με Χρήση της Μεθόδου Οπτικής Ροής των Lucas-Kanade***

# ### 1.1  Ανίχνευση Δέρματος Προσώπου και Χεριών

# In[ ]:


pip install seaborn


# In[1]:


#imports
import sys
import os

import numpy as np
import cv2

import scipy.io
from scipy.stats import multivariate_normal
from scipy.ndimage import label
from scipy import ndimage

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sn

from numpy.linalg import matrix_power


# In[2]:


sys.path.insert(0,'D:\\Επιφάνεια εργασίας\\OY-Lab2\\cv23_lab2_material\\part1 - GreekSignLanguage')


# In[3]:


os.chdir('D:\\Επιφάνεια εργασίας\\OY-Lab2\\cv23_lab2_material\\part1 - GreekSignLanguage')


# In[7]:


mat = scipy.io.loadmat("skinSamplesRGB.mat")
samples = mat.get('skinSamplesRGB')
print(samples)


# In[5]:


print(samples.shape)


# Βλέπουμε πως έχουμε 22 στοιχεία, διαστάσεων 81x3, δηλαδή κάθε στοιχείο αποτελείται απο 81 τριάδες RGB συντεταγμένων. Αρχικά, θα μετατρέψουμε τα δεδομένα αυτά απο το χώρο RGB στο χώρο YCbCr. Στη συνέχεια θα υπολογίσουμε το διάνυσμα της μέσης τιμής μ = [μCb, μCr] και τον πίνακα συνδιακύμανσης Σ. Υπολογίζοντας αυτά, έχουμε κάνει fit μια γκαουσιανή κατανομή γύρω απο τα δεδομένα τα οποία αποτελούν σημεία στον χώρο YCbCr που ανήκουν σε δέρμα. Τέλος, υπολογίζουμε τη πιθανότητα για κάθε pixel του να ανήκει στη κατανομή αυτή με τη συνάρτηση multivariate normal και κατωφλιοποιούμε με βάση κάποιες τιμές. 

# In[8]:


samples_YCbCr = cv2.cvtColor(samples, cv2.COLOR_RGB2YCR_CB)


# In[6]:


plt.imshow(samples_YCbCr)
plt.axis('off')


# In[4]:


BGRImage = cv2.imread('1.png')
img = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2YCR_CB)

plt.imshow(img)
plt.axis('off')


# In[10]:


#Computing mean values and covariance matrix from samples

data1 = samples_YCbCr[:,:,1].flatten()
meanCb = np.mean(data1)

data2 = samples_YCbCr[:,:,2].flatten()
meanCr = np.mean(data2)

data = np.array([data1,data2])
cov_matrix = np.cov(data, bias=True)


# In[167]:


# Initializing the random seed
random_seed=1000
mean = np.array([meanCb,meanCr])
     
# Generating a Gaussian bivariate distribution
# with given mean and covariance matrix
distr = multivariate_normal(cov = cov_matrix, mean = mean,
                                seed = random_seed)
     
# Generating 5000 samples out of the
# distribution
data = distr.rvs(size = 5000)
     
# Plotting the generated samples
plt.plot(data[:,0],data[:,1], 'o', c='skyblue', markeredgewidth = 0.5, markeredgecolor = 'black')
plt.title('scatter plot')
plt.xlabel('Cb')
plt.ylabel('Cr')
plt.axis('equal')
     
plt.show()


# In[169]:


fig = plt.figure(figsize=(20,20))

# Generating a Gaussian bivariate distribution
# with given mean and covariance matrix
distr = multivariate_normal(cov = cov_matrix, mean = mean, seed = random_seed)
     
# Generating a meshgrid complacent with
# the 3-sigma boundary
mean_1, mean_2 = mean[0], mean[1]
sigma_1, sigma_2 = cov_matrix[0,0], cov_matrix[1,1]
     
x = np.linspace(0, 255, num=1000)
y = np.linspace(0, 255, num=1000)
X, Y = np.meshgrid(x,y)
     
# Generating the density function
# for each point in the meshgrid
pdf = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pdf[i,j] = distr.pdf([X[i,j], Y[i,j]])
     
# Plotting the density function values
key = 131
ax = fig.add_subplot(key, projection = '3d')
ax.plot_surface(X, Y, pdf, cmap = 'viridis')
plt.xlabel("Cb")
plt.ylabel("Cr")
plt.title('Gaussian PDF')
ax.axes.zaxis.set_ticks([])
plt.tight_layout()
plt.show()


# In[11]:


#Creating the probabilistic image
x = img.shape[0]
y = img.shape[1]

p = np.zeros((x,y))

mean = [meanCb,meanCr]
cov = cov_matrix

for i in range(x):
    for j in range(y):
        p[i][j] = multivariate_normal.pdf([img[i][j][1],img[i][j][2]], mean, cov, True)


# In[12]:


#Normalize image to [0,1]
norm_img = (p - p.min()) / (p.max()- p.min())

#Thresholding
_, binary = cv2.threshold(norm_img, 0.08, 1, cv2.THRESH_BINARY)

plt.imshow(binary,cmap='viridis')
plt.axis('off')


# Δοκιμάζοντας διάφορες τιμές κατωφλίου, τα αποτελέσματα φάνηκαν να προσεγγίζουν το επιθυμητό αποτέλεσμα στο διάστημα [0.02-0.12] και θεωρήσαμε ως καλύτερη επιλογή την τιμή 0.08 καθώς κρατάει την άκρως απαραίτητη πληροφορία του δέρματος χωρίς θόρυβο. 
# 
# Παρακάτω, κλείνοντας το κεφάλαιο της ανίχνευσης περιοχών του δέρματος, θα προχωρήσουμε σε μια μορφολογική επεξεργασία της εικόνας και συγκεκριμένα θα κάνουμε opening την εικόνα με ένα μικρό δομικό στοιχείο και closing με ένα μεγάλο, ώστε να εξαλειφθούν τα μικρά κενά εντός των περιοχών.

# In[13]:


structure_open = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
structure_close = cv2.getStructuringElement(cv2.MORPH_RECT,(30,30))

opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, structure_open)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, structure_close)

plt.imshow(closing,cmap='viridis')
plt.axis('off')


# Δοκιμάσαμε διάφορες τιμές για το μέγεθος των δομικών στοιχείων και ο συνδυασμός που είχε το πιο ευδιάκριτο και λεπτομερές αποτέλεσμα ήταν για το μικρό στοιχείο 2x2 και για το μεγάλο 30x30. 
# 
# Στη συνέχεια, αναθέτουμε label σε κάθε περιοχή της εικόνας για να διαχωρίσουμε τις τρείς διαφορετικές περιοχές οι οποίες είναι:
# 
# 
# * αριστερό χέρι
# * δεξί χέρι
# * κεφάλι
# 
# Τέλος θα δημιουργήσουμε τα bounding boxes που θα περιβάλλουν τις περιοχές ενδιαφέροντος

# In[14]:


labeled_img, num_features = label(closing)

plt.imshow(labeled_img,cmap='viridis')
plt.axis('off')


# Οι 3 περιοχές διαχωρίζονται μεταξύ τους με διαφορετικό χρώμα

# In[15]:


labeled_img = labeled_img.astype(np.uint8)

# get external contours
contours = cv2.findContours(labeled_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

rgb_img = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2RGB)

# draw contours
res = rgb_img.copy()

colors = [[0,255,0],
          [0,0,255],
          [255,0,0]]
i = 0
for cntr in contours:
    # get bounding boxes
    pad = 6
    x,y,w,h = cv2.boundingRect(cntr)
    cv2.rectangle(res, (x-pad, y-pad), (x+w+pad, y+h+pad), colors[i], 2)
    i = i+1
    
# save result
#cv2.imwrite("res.png",res)

plt.imshow(res)
plt.axis('off')


# Παρακάτω, θα συνοψίσουμε όλα τα παραπάνω βήματα σε μια συνάρτηση find_bounding(I,mu,cov) η οποία θα δέχεται ως εισόδους την εικόνα, τη μέση τιμή και τη συνδιακύμανση της γκαουσιανής κατανομής και θα επιστρέφει τις συντεταγμένες και τις διαστάσεις του bounding box που περιβάλλει τις περιοχές ενδιαφέροντος της εικόνας.

# In[56]:


def find_bounding(I, mu, cov):
    
    
    #Dimensions
    x = I.shape[0]
    y = I.shape[1]
    
    #probabilistic image
    prob = np.zeros((x,y))
    for i in range(x):
        for j in range(y):
            prob[i][j] = multivariate_normal.pdf([I[i][j][1],I[i][j][2]], mu, cov, True)
            
    
    #Normalize image to [0,1]
    normal_img = (prob - prob.min()) / (prob.max()- prob.min())

    #Thresholding
    _, binary_img = cv2.threshold(normal_img, 0.08, 1, cv2.THRESH_BINARY)
    
    #Morphological operations
    structure_open = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    structure_close = cv2.getStructuringElement(cv2.MORPH_RECT,(30,30))

    morph_open = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, structure_open)
    morph_close = cv2.morphologyEx(morph_open, cv2.MORPH_CLOSE, structure_close)
    
    #Labeling image
    label_img, num_features = label(morph_close)
    
    label_img = label_img.astype(np.uint8)

    #Get external contours
    cont = cv2.findContours(label_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont = cont[0] if len(cont) == 2 else cont[1]

    res = []
    for cntr in cont:
        # get bounding boxes
        pad = 6
        x,y,w,h = cv2.boundingRect(cntr)
        res.append((x, y, w, h))
        
    return res


# In[57]:


boxes = find_bounding(img, mean, cov)
print(boxes)


# Παραπάνω φαίνονται οι συντεταγμένες (x,y), το πλάτος (width) και ύψος (height) για κάθε ένα απο τα τρία bounding_boxes της εικόνας μας με σειρά: 
# 
# 1) Δεξί χέρι
# 2) Αριστερό χέρι
# 3) Πρόσωπο

# ## **1.2 Παρακολούθηση Προσώπου και Χεριών**

# ### 1.2.1 Υλοποίηση του Αλγόριθμου των Lucas-Kanade

# In[59]:


def lucas_kanade(I1, I2, features, rho, epsilon, d_x0, d_y0):
        
        '''
        Iterative algorithm for finding optical flow of 
        interest points (features) between two frames of a video
        
        '''
        max_iterations = 20 # till convergence  
            
        ########### Normalize images to [0,1] #############
        gray1 = cv2.cvtColor(I1,cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(I2,cv2.COLOR_BGR2GRAY)

        I1 = (gray1- gray1.min()) / (gray1.max()- gray1.min())
        I2 = (gray2 - gray2.min()) / (gray2.max()- gray2.min())
            
        #############     Gaussian Kernel   ###############
        ksize = int(2*np.ceil(3*rho)+1) 
        gauss_kernel = cv2.getGaussianKernel(ksize, rho)
        gauss_2d = gauss_kernel @ gauss_kernel.T
        
        
        ############   Gradient of the image  ##############
        grad = np.gradient(I1)
        gradientx = grad[0]
        gradienty = grad[1]
        
        ############ Create meshgrid of the image ##########
        x_0, y_0 = np.meshgrid(np.arange(0,I1.shape[1]),np.arange(0, I1.shape[0]))
        
        
        #Array of d vector for every pixel of interest
        d_vectors = []
        
        for corner in (features):
            x = corner[1]
            y = corner[0]
            x = int(np.round(x))
            y = int(np.round(y))
                
            d = [d_x0, d_y0]
            
            for i in range(max_iterations):
                
                #########  Computing In-1(x + d)  #########
                In_1 = ndimage.map_coordinates(I1,[np.ravel(y_0 + d[1]),np.ravel(x_0 + d[0])], order=1)
                In_1 = In_1.reshape(I1.shape[0],I1.shape[1])
                
                #E(x) = In(x) - In-1(x+di)
                E = I2 - In_1
                
            
                #########  Computing dIn-1(x + d)/dx ########
                A1 = ndimage.map_coordinates(gradientx,[np.ravel(y_0 + d[1]), np.ravel(x_0 + d[0])], order=1)
                A1 = A1.reshape(gradientx.shape[0], gradientx.shape[1])
                
            
                #########  Computing dIn-1(x + d)/dy ########
                A2 = ndimage.map_coordinates(gradienty,[np.ravel(y_0 + d[1]), np.ravel(x_0 + d[0])], order=1)
                A2 = A2.reshape(gradienty.shape[0], gradienty.shape[1])
                
                #########   Define 1st Matrix  ##########
                A11 = cv2.filter2D(A1**2,-1,gauss_2d)[x,y] + epsilon
                A12 = cv2.filter2D(A1*A2,-1,gauss_2d)[x,y]
                A21 = cv2.filter2D(A1*A2,-1,gauss_2d)[x,y]
                A22 = cv2.filter2D(A2**2,-1,gauss_2d)[x,y] + epsilon
                
                ########   Define 2nd Matrix  ##########
                B1 = cv2.filter2D(A1*E,-1,gauss_2d)[x,y]
                B2 = cv2.filter2D(A2*E,-1,gauss_2d)[x,y]
                
                det = (A11*A22 - A12*A21) #Determinant
                
                ux = (A22*B1 - A12*B2) / det
                uy = (A11*B2 - A21*B1) / det
                
                dx = d[0] + ux
                dy = d[1] + uy
                
                d = [dx, dy]
                
            d_vectors.append((d[0],d[1]))

                
        return d_vectors      


# Παραδείγματα χρησιμοποίησης του αλγορίθμου Lucas-Kanade

# In[5]:


BGRImage2 = cv2.imread('2.png')
img2 = cv2.cvtColor(BGRImage2, cv2.COLOR_BGR2YCR_CB)

plt.imshow(img2)
plt.axis('off')


# In[11]:


#cutting image from bounding boxes
#left hand coordinates: (94, 274, 64, 78)
rgb_img = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2RGB)
rgb_img2 = cv2.cvtColor(BGRImage2, cv2.COLOR_BGR2RGB)

#left hand
cuttI1 = rgb_img[274:352, 94:158]
cuttI2 = rgb_img2[274:352, 94:158]

fig = plt.figure()

fig.add_subplot(1,2,1)
plt.imshow(cuttI1)
plt.axis('off')
plt.title('Frame 1')

fig.add_subplot(1,2,2)
plt.imshow(cuttI2)
plt.axis('off')
plt.title('Frame 2')


# In[67]:


#Finding corners in the left hand area
gray = cv2.cvtColor(cuttI1,cv2.COLOR_BGR2GRAY)

features = cv2.goodFeaturesToTrack(gray, 20, 0.2, 2)
features = np.int0(features)

color = (255,0,0)
radius = 1
thickness = 1
image = cuttI1.copy()
for i in features:
    x,y = i.ravel()
    cv2.circle(image,(x,y), radius, color, thickness )
    
plt.imshow(image)
plt.axis('off')


# In[68]:


print(features.shape)


# In[69]:


#######  Parameters  ########
rho = 5
epsilon = 0.1
d_x0 = 0
d_y0 = 0


# In[70]:


#Finding optic flow arrows from image features

features_left = features.reshape(len(features),2)

d_vectors_l = lucas_kanade(cuttI1,cuttI2, features_left, rho, epsilon, d_x0, d_y0)


# In[27]:


#Plotting the optic flow arrows on the second frame of left hand

d_x = [-row[1] for row in d_vectors_l]
d_y = [-row[0] for row in d_vectors_l]

x_pos = features_left[:,0]
y_pos = features_left[:,1]

# Creating plot
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.quiver(x_pos, y_pos, d_x,d_y, angles='xy',scale=100,color='mediumspringgreen')
ax1.set_aspect('equal')
ax1.imshow(cuttI2)

ax2.quiver(x_pos, y_pos, d_x,d_y, angles='xy',scale=100)
ax2.set_aspect('equal')
plt.ylim(max(plt.ylim()), min(plt.ylim()))

plt.show()


# In[28]:


#Parameter analysis
r = [1,2,3,4,5]
e = [0.01,0.03,0.05,0.08,0.1]
d_x0 = 0
d_y0 = 0

feat = []
vectors = []
for ro in r:
    for eps in e:
        
        feat = cv2.goodFeaturesToTrack(gray, 50, 0.01, 2)
        feat = features.reshape(len(features),2)

        vectors = lucas_kanade(cuttI1,cuttI2, feat, ro, eps, d_x0, d_y0)
        
        d_x = [-row[1] for row in vectors]
        d_y = [-row[0] for row in vectors]

        x_pos = feat[:,0]
        y_pos = feat[:,1]
        
        plt.quiver(x_pos, y_pos, d_x,d_y, angles='xy',scale=100, color='yellow')
        plt.title("rho = " + str(ro) + ", e = " + str(eps))
        plt.ylim(max(plt.ylim()), min(plt.ylim()))
        plt.imshow(cuttI2)
        
        plt.show()


# Ακολούθως θα εφαρμόσουμε τον αλγόριθμο και για τα μέρη του προσώπου και δεξιού χεριού.

# In[29]:


#face coordinates: (156, 104, 70, 116) for image 1
cuttI1_face = rgb_img[104:220, 156:226]
cuttI2_face = rgb_img2[104:220, 156:226]

#right hand: (208, 270, 52, 80) for image 1
cuttI1_right = rgb_img[270:350, 208:260]
cuttI2_right = rgb_img2[270:350, 208:260]

fig = plt.figure(figsize=(10, 4))

fig.add_subplot(1,4,1)
plt.imshow(cuttI1_face)
plt.axis('off')

fig.add_subplot(1,4,2)
plt.imshow(cuttI2_face)
plt.axis('off')

fig.add_subplot(1,4,3)
plt.imshow(cuttI1_right)
plt.axis('off')

fig.add_subplot(1,4,4)
plt.imshow(cuttI2_right)
plt.axis('off')


# In[30]:


#Finding corners in the face area
gray_face = cv2.cvtColor(cuttI1_face,cv2.COLOR_BGR2GRAY)

features_face = cv2.goodFeaturesToTrack(gray_face, 20, 0.2, 2)
features_face = np.int0(features_face)

color = (255,0,0)
radius = 1
thickness = 1

imagef = cuttI1_face.copy()
for i in features_face:
    x,y = i.ravel()
    cv2.circle(imagef,(x,y), radius, color, thickness)
    
plt.imshow(imagef)
plt.axis('off')


# In[33]:


#Finding corners in the right hand area
gray_right = cv2.cvtColor(cuttI1_right,cv2.COLOR_BGR2GRAY)

features_right = cv2.goodFeaturesToTrack(gray_right, 20, 0.05, 2)
features_right = np.int0(features_right)

color = (255,0,0)
radius = 1
thickness = 1

imager = cuttI1_right.copy()
for i in features_right:
    x,y = i.ravel()
    cv2.circle(imager,(x,y),radius,color,thickness)
    
plt.imshow(imager)
plt.axis('off')


# In[34]:


#Finding optic flow arrows from image features on face

f_face = features_face.reshape(len(features_face),2)

d_vectors_f = lucas_kanade(cuttI1_face,cuttI2_face, f_face, rho, epsilon, d_x0, d_y0)


# In[35]:


#Plotting the optic flow arrows on the second frame of the face

d_x_f = [-row[1] for row in d_vectors_f]
d_y_f = [-row[0] for row in d_vectors_f]

x_pos = f_face[:,0]
y_pos = f_face[:,1]

# Creating plot
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.quiver(x_pos, y_pos, d_x_f,d_y_f, angles='xy',scale=100,color='mediumspringgreen')
ax1.set_aspect('equal')
ax1.imshow(cuttI2_face)


ax2.quiver(x_pos, y_pos, d_x_f,d_y_f, angles='xy',scale=100)
ax2.set_aspect('equal')
plt.ylim(max(plt.ylim()), min(plt.ylim()))

plt.show()


# In[36]:


#Finding optic flow arrows from image features on right hand

f_right = features_right.reshape(len(features_right),2)

d_vectors_r = lucas_kanade(cuttI1_right,cuttI2_right, f_right, rho, epsilon, d_x0, d_y0)


# In[38]:


#Plotting the optic flow arrows on the second frame of right hand

d_x_r = [-row[1] for row in d_vectors_r]
d_y_r = [-row[0] for row in d_vectors_r]

x_pos = f_right[:,0]
y_pos = f_right[:,1]

# Creating plot
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.quiver(x_pos, y_pos, d_x_r,d_y_r, angles='xy',scale=100,color='mediumspringgreen')
ax1.set_aspect('equal')
ax1.imshow(cuttI2_right)


ax2.quiver(x_pos, y_pos, d_x_r,d_y_r, angles='xy',scale=100)
ax2.set_aspect('equal')
plt.ylim(max(plt.ylim()), min(plt.ylim()))

plt.show()


# ### 1.2.2 Υπολογισμός της Μετατόπισης των Παραθύρων από τα Διανύσματα Οπτικής Ροής

# In[63]:


import math

def displ(d_x, d_y):
    
    threshold = 0.2
    d_x = np.array(d_x)
    d_y = np.array(d_y)
    energy = (d_x**2+d_y**2)
    mask = (energy > threshold).astype('int32')
    
    if np.sum(mask)==0:
        return 0,0
    
    dx_mean = np.sum(d_x*mask)/np.sum(mask)
    dy_mean = np.sum(d_y*mask)/np.sum(mask)

    return int(np.round(dx_mean)), int(np.round(dy_mean))


# In[66]:


#Computing final optic flow displacement arrow for each bounding box

displ_x, displ_y = displ(d_x,d_y)

print("Final vector displacement of bounding box for left hand: " + str(displ_x) + ", " + str(displ_y))

displ_x_f, displ_y_f = displ(d_x_f,d_y_f)

print("Final vector displacement of bounding box for face: " + str(displ_x_f) + ", " + str(displ_y_f))

displ_x_r, displ_y_r = displ(d_x_r,d_y_r)

print("Final vector displacement of bounding box for right hand: " + str(displ_x_r) + ", " + str(displ_y_r))


# Παρακάτω θα απεικονίσουμε τα τελικά διανύσματα μετατόπισης των ορθογωνίων bounding boxes. Αυτά τα διανύσματα μας δείχνουν σύμφωνα με τους υπολογισμούς μας τη μετατόπιση που συμβαίνει για κάθε περιοχή μεταξύ δύο frame του video.

# In[43]:


#Plotting the optic flow arrows on the first frame of the left hand

x_pos = cuttI2.shape[1]/2
y_pos = cuttI2.shape[0]/2

# Creating plot
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.quiver(x_pos, y_pos, displ_x, displ_y, angles='xy',scale=80,color='red')
ax1.set_aspect('equal')
ax1.imshow(cuttI2)


ax2.quiver(x_pos, y_pos, displ_x, displ_y, angles='xy',scale=80)
ax2.set_aspect('equal')
plt.ylim(max(plt.ylim()), min(plt.ylim()))

plt.show()


# In[46]:


#Plotting the optic flow arrows on the first frame of the right hand

x_pos_r = cuttI2_right.shape[1]/2
y_pos_r = cuttI2_right.shape[0]/2

# Creating plot
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.quiver(x_pos_r, y_pos_r, displ_x_r, displ_y_r, angles='xy',scale=80,color='red')
ax1.set_aspect('equal')
ax1.imshow(cuttI2_right)


ax2.quiver(x_pos_r, y_pos_r, displ_x_r, displ_y_r, angles='xy',scale=80)
ax2.set_aspect('equal')
plt.ylim(max(plt.ylim()), min(plt.ylim()))

plt.show()


# In[49]:


#Plotting the optic flow arrows on the first frame of the face

x_pos_f = cuttI2_face.shape[1]/2
y_pos_f = cuttI2_face.shape[0]/2

# Creating plot
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.quiver(x_pos_f, y_pos_f, displ_x_f, displ_y_f, angles='xy',scale=20,color='red')
ax1.set_aspect('equal')
ax1.imshow(cuttI2_face)


ax2.quiver(x_pos_f, y_pos_f, displ_x_f, displ_y_f, angles='xy',scale=20)
ax2.set_aspect('equal')
plt.ylim(max(plt.ylim()), min(plt.ylim()))

plt.show()


# Απο τα διανύσματα αυτά συμπεραίνουμε πως οι κινήσεις που αντιλαμβάνεται ο υπολογιστής είναι:
# 
# * Αριστερό χέρι -> Κάτω αριστερά
# * Δεξί χέρι -> Κάτω αριστερά
# * Πρόσωπο -> Κάτω δεξιά
# 
# Παρατηρώντας τις εικόνες 1 και 2 (διαδοχικά καρέ του βίντεο) βλέπουμε πως όντως οι κινήσεις που κάνουν οι περιοχές αυτές στη πραγματικότητα ταιριάζουν σε μεγάλο βαθμό με τις προσεγγίσεις των κινήσεων που κάνει ο υπολογιστής.
# 
# Παρακάτω, αρχικά θα διαβάσουμε όλες τις εικόνες(καρέ) του βίντεο και έπειτα θα εφαρμόσουμε στο σύστημα παρακολούθησης.

# In[54]:


from PIL import Image
import os, os.path
from collections import OrderedDict

imgs = {}
path = os.getcwd()
valid_images = [".jpg",".gif",".png",".tga"]
for f in os.listdir(path):
    name = os.path.splitext(f)[0]
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    
    BGRImage = cv2.imread(f)
    img = cv2.cvtColor(BGRImage, cv2.COLOR_BGRA2RGB)
    
    i = int(name)
    #imgs[i] = Image.open(os.path.join(path,f))  
    imgs[i] = img
    
IMAGES = OrderedDict(sorted(imgs.items()))


# In[55]:


#Function for croping images based on bounding boxes and applying feature extraction from frame 2
def crop_and_feature(I1,I2, x, y, w, h, num_of_corn,quality, dist):
    width = w
    height = h
    x = int(np.round(x))
    y = int(np.round(y))
    cuttI1 = I1[y:y + height, x:x+width]
    cuttI2 = I2[y:y + height, x:x+width]
    
    gray = cv2.cvtColor(cuttI1,cv2.COLOR_BGR2GRAY)
    features = cv2.goodFeaturesToTrack(gray,  num_of_corn, quality, dist)
    if features is None:
        features = np.zeros((2,2))
    
    return cuttI1, cuttI2, features


# In[61]:


def tracking(images, dis_x_l, dis_y_l, dis_x_r, dis_y_r, dis_x_f, dis_y_f):
    
    #heights and widths for left hand,right hand and face
    width = [64, 52, 70]
    height = [78, 80, 116]
    
    #Tracking coords keep coordinates for the 3 bounding boxes
    tracking_coords = np.zeros((3,70,4))
    tracking_coords[0][0] = (94, 274, 64, 78)
    tracking_coords[1][0] = (208,270, 52, 80)
    tracking_coords[2][0] = (156,104, 70, 116)
    
    #initializing displacements
    x_l = dis_x_l
    y_l = dis_y_l
    x_r = dis_x_r
    y_r = dis_y_r
    x_f = dis_x_f
    y_f = dis_y_f
    
    x = [x_l, x_r, x_f]
    y = [y_l, y_r, y_f]
    
    #Parameters
    rho = 5
    epsilon = 0.1
    d_x0 = 0
    d_y0 = 0
    
    #for every image
    for i in range(1,70):
        if i%10 == 0:
            print(i)
        #for each bounding box area
        for j in range(3):
            I1, I2, features = crop_and_feature(images[i], images[i+1], x[j], y[j], width[j], height[j], 20, 0.05, 1)
        
            features = features.reshape(len(features),2)

            d_vectors = lucas_kanade(I1,I2, features, rho, epsilon, d_x0, d_y0)
        
            d_x = [-row[1] for row in d_vectors]
            d_y = [-row[0] for row in d_vectors]
        
            displ_x, displ_y = displ(d_x,d_y)
        
            x[j] = x[j] + displ_x 
            y[j] = y[j] + displ_y 
        
            tracking_coords[j][i] = ((x[j], y[j], width[j], height[j]))
            
    return tracking_coords  


# In[62]:


tracking_coords = tracking(IMAGES, 94, 274, 208, 270, 156, 104)


# In[34]:


print(tracking_coords.shape)


# In[64]:


def sh(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
#Drawing bounding box
for i in range(1,71):
    res = IMAGES[i]
    res_rgb = cv2.cvtColor(res, cv2.COLOR_RGBA2BGR)
    res = res_rgb.copy()
    
    '''
    #RGB
    colors = [[0,0,255],
              [0,255,0],
              [255,0,0]]
    '''
    
    #YCbCr
    colors = [[41, 240, 110],
              [145, 54, 34],
              [82, 90, 240]]
    
    pad = 6
    for j in range(3):
        x = tracking_coords[j][i-1][0]
        y = tracking_coords[j][i-1][1]
        w = tracking_coords[j][i-1][2]
        h = tracking_coords[j][i-1][3]
    
        x = int(np.round(x))
        y = int(np.round(y)) 
        w = int(w)
        h = int(h)
     
        print("coords: " + str(x) + " " + str(y))
        cv2.rectangle(res, (x-pad, y-pad), (x+w+pad, y+h+pad), colors[j], 2)
        
    cv2.imwrite("res_" + str(i) + ".png", res)

    sh(res)


# ### 1.2.3 Πολυ-Κλιμακωτός Υπολογισμός Οπτικής Ροής

# Παρακάτω υλοποιούμε την πολυκλιμακωτή έκδοση του Lucas-Kanade σε μια συνάρτηση που θα δέχεται ως είσοδο δύο εικόνες (παράθυρα απο διαδοχικά frames), το ρ, ε και την αρχική εκτίμηση για το διάνυσμα μετατόπισης και θα έχει ως έξοδο τη τελική τιμή του διανύσματος μετατόπισης.

# In[81]:


import math

def downsample(I1,I2):
    
    #Perform downsampling
    I1_down = cv2.pyrDown(I1)
    I2_down = cv2.pyrDown(I2)
    
    return I1_down, I2_down
    

def pyramid(I1,I2, pyramid_levels):
    '''
    Make gaussian pyramid by downsampling
    i1 and i2
    '''
    
    G_Pyramid1 = []
    G_Pyramid2 = []
    G_Pyramid1.append(I1)
    G_Pyramid2.append(I2)
    
    #######   Gaussian kernel  ########
    sigma = 3
    gauss_kernel = cv2.getGaussianKernel(3, sigma)
    gauss_2d = gauss_kernel @ gauss_kernel.T
    
    #######     Pyramid     ########
    for level in range(pyramid_levels-1):
        
        img1 = cv2.filter2D(G_Pyramid1[level], -1, gauss_2d)
        img2 = cv2.filter2D(G_Pyramid2[level], -1, gauss_2d)
    
        img1, img2 = downsample(img1, img2)
        G_Pyramid1.append(img1)
        G_Pyramid2.append(img2)
    
    return G_Pyramid1, G_Pyramid2


def multiscale_LK(I1, I2, rho, epsilon, dx_0, dy_0, pyramid_levels):
        
        #Guassian pyramid
        g_pyramid1, g_pyramid2 = pyramid(I1, I2, pyramid_levels)
        
        #Initial estimation
        dx = dx_0
        dy = dy_0
        
        #for every pyramid level
        for j in reversed(range(pyramid_levels)): 
            
            #Features between levels
            gray = cv2.cvtColor(g_pyramid1[j],cv2.COLOR_BGR2GRAY)
            features = cv2.goodFeaturesToTrack(gray, 20, 0.05, 1)
            if features is None:
                features = np.zeros((2,2))
            features = features.reshape(len(features),2)
            
            d_vect = lucas_kanade(I1, I2, features, rho, epsilon, dx, dy)
            
            d_x = [-row[1] for row in d_vect]
            d_y = [-row[0] for row in d_vect]
            
            #Averaging the displacement vectors
            displ_x, displ_y = displ(d_x,d_y)
            
            if j > 0:
                dx = 2*displ_x
                dy = 2*displ_y
            else:
                dx = displ_x
                dy = displ_y
                
            
        d_vector_multiscale = [dx,dy] #final displacement vector
        
        return d_vector_multiscale


# In[61]:


#Tracking bounding boxes for every frame

def multiscale_tracking(images):
    
    #heights and widths for left hand,right hand and face
    width = [64, 52, 70]
    height = [78, 80, 116]

    
    #Tracking coords keep coordinates for the 3 bounding boxes
    tracking_coords = np.zeros((3,70,4))
    tracking_coords[0][0] = (94, 274, 64, 78)
    tracking_coords[1][0] = (208,270, 52, 80)
    tracking_coords[2][0] = (156,104, 70, 116)
    
    x = [94, 208, 156]
    y = [274, 270, 104]
    
    #Parameters
    rho = 5
    epsilon = 0.1
    d_x0 = 0
    d_y0 = 0
    pyr_levels = 3
    
    #for every image
    for i in range(1,70):
        if i%5 == 0:
            print(i)
        #for each bounding box area
        for j in range(3):
            
            I1, I2, _ = crop_and_feature(images[i], images[i+1], x[j], y[j], width[j], height[j], 20, 0.05, 2)
            
            d_vectors = multiscale_LK(I1,I2, rho, epsilon, d_x0, d_y0, pyr_levels)
        
            d_x = d_vectors[0]
            d_y = d_vectors[1]
                  
            x[j] = x[j] + d_x 
            y[j] = y[j] + d_y 
        
            tracking_coords[j][i] = ((x[j], y[j], width[j], height[j]))
        
    return tracking_coords 


# In[64]:


tracking_coords_mult = multiscale_tracking(IMAGES)


# In[66]:


def sh(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
#Drawing bounding box
for i in range(1,71):
    res = IMAGES[i]
    res_rgb = cv2.cvtColor(res, cv2.COLOR_RGBA2BGR)
    res = res_rgb.copy()
    
    '''
    #RGB
    colors = [[0,0,255],
              [0,255,0],
              [255,0,0]]
    '''
    
    #YCbCr
    colors = [[41, 240, 110],
              [145, 54, 34],
              [82, 90, 240]]
    
    pad = 6
    for j in range(3):
        x = tracking_coords_mult[j][i-1][0]
        y = tracking_coords_mult[j][i-1][1]
        w = tracking_coords_mult[j][i-1][2]
        h = tracking_coords_mult[j][i-1][3]
    
        x = int(np.round(x))
        y = int(np.round(y)) 
        w = int(w)
        h = int(h)
     
        print("coords: " + str(x) + " " + str(y))
        cv2.rectangle(res, (x-pad, y-pad), (x+w+pad, y+h+pad), colors[j], 2)
        
    cv2.imwrite("res_" + str(i) + ".png", res)

    sh(res)
 


# Εξέταση επίδρασης του αριθμού των επιπέδων

# In[93]:


d_vectors_test_mult = multiscale_LK(cuttI1,cuttI2, 2, 0.1, 0, 0, 4)

d_xx = d_vectors_test_mult[0]
d_yy = d_vectors_test_mult[1]

# Creating plot
fig, (ax1) = plt.subplots(1)

ax1.quiver(d_xx,d_yy, angles='xy',scale=100,color='red')
ax1.set_aspect('equal')
plt.title('4 Levels, e = 0.1, rho = 2')

plt.show()

