#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/JimmyLiu8351/liulabtracking/blob/main/eye_test_py.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[20]:


import cv2
import numpy as np
import math
import datetime
import time

from tqdm import tqdm
from multiprocessing import Pool, Process, Queue, Manager


# In[2]:


def createLineIterator(P1, P2, img):
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])     
    """
    #define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    #difference and absolute difference between points
    #used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    #predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
    itbuffer.fill(np.nan)

    #Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X: #vertical line segment
        itbuffer[:,0] = P1X
        if negY:
            itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
        else:
            itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)              
    elif P1Y == P2Y: #horizontal line segment
        itbuffer[:,1] = P1Y
        if negX:
            itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
        else:
            itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
    else: #diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32)/dY.astype(np.float32)
            if negY:
                itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
            else:
                itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
            itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int) + P1X
        else:
            slope = dY.astype(np.float32)/dX.astype(np.float32)
            if negX:
                itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
            else:
                itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
            itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int) + P1Y

    #Remove points outside of image
    colX = itbuffer[:,0]
    colY = itbuffer[:,1]
    itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

    #Get intensities from img ndarray
    itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]

    return itbuffer


# In[3]:


def gradx(img):
    img = img.astype('int')
    rows, cols = img.shape
    # Use hstack to add back in the columns that were dropped as zeros
    return np.hstack( (np.zeros((rows, 1)), (img[:, 2:] - img[:, :-2])/2.0, np.zeros((rows, 1))) )


# In[4]:


def grady(img):
    img = img.astype('int')
    rows, cols = img.shape
    # Use vstack to add back the rows that were dropped as zeros
    return np.vstack( (np.zeros((1, cols)), (img[2:, :] - img[:-2, :])/2.0, np.zeros((1, cols))) )


# In[5]:


#Performs fast radial symmetry transform
#img: input image, grayscale
#radii: integer value for radius size in pixels (n in the original paper); also used to size gaussian kernel
#alpha: Strictness of symmetry transform (higher=more strict; 2 is good place to start)
#beta: gradient threshold parameter, float in [0,1]
#stdFactor: Standard deviation factor for gaussian kernel
#mode: BRIGHT, DARK, or BOTH
def frst(img, radii, alpha, beta, stdFactor, mode='BOTH'):
    mode = mode.upper()
    assert mode in ['BRIGHT', 'DARK', 'BOTH']
    dark = (mode == 'DARK' or mode == 'BOTH')
    bright = (mode == 'BRIGHT' or mode == 'BOTH')

    workingDims = tuple((e + 2*radii) for e in img.shape)

    #M and O working matrices
    O_n = np.zeros(workingDims, np.int16)
    M_n = np.zeros(workingDims, np.int16)

    #Calculate gradients
    gx = gradx(img)
    gy = grady(img)

    #Find gradient vector magnitude
    gnorms = np.sqrt( np.add( np.multiply(gx, gx) , np.multiply(gy, gy) ) )

    #Use beta to set threshold - speeds up transform significantly
    gthresh = np.amax(gnorms)*beta

    #Find x/y distance to affected pixels
    gpx = np.multiply(np.divide(gx, gnorms, out=np.zeros(gx.shape), where=gnorms!=0), radii).round().astype(int);
    gpy = np.multiply(np.divide(gy, gnorms, out=np.zeros(gy.shape), where=gnorms!=0), radii).round().astype(int);

    #Iterate over all pixels (w/ gradient above threshold)
    for coords, gnorm in np.ndenumerate(gnorms):
        if gnorm > gthresh:
            i, j = coords
            #Positively affected pixel
            if bright:
                ppve = (i+gpy[i,j]+radii, j+gpx[i,j]+radii)
                O_n[ppve] += 1
                M_n[ppve] += gnorm
            #Negatively affected pixel
            if dark:
                pnve = (i-gpy[i,j]+radii, j-gpx[i,j]+radii)
                O_n[pnve] -= 1
                M_n[pnve] -= gnorm

    

    #Abs and normalize O matrix
    O_n = np.abs(O_n)
    O_n = O_n / float(np.amax(O_n))

    #Normalize M matrix
    M_max = float(np.amax(np.abs(M_n)))
    M_n = M_n / M_max

    #Elementwise multiplication
    F_n = np.multiply(np.power(O_n, alpha), M_n)

    #Gaussian blur
    kSize = int( np.ceil( radii / 2 ) )
    kSize = kSize + 1 if kSize % 2 == 0 else kSize

    S = cv2.GaussianBlur(F_n, (kSize, kSize), int( radii * stdFactor ))
    
    return S[radii:-radii,radii:-radii]


# In[6]:


def rough_corneal_remove(gray_frame, replace_val):
    # roughly removing corneal reflection
    blur_frame = cv2.blur(gray_frame, (5, 5))
    ret, thresh_frame = cv2.threshold(blur_frame, replace_val, 255, cv2.THRESH_TRUNC)

    '''
    for (x, y), value in np.ndenumerate(thresh_frame):
        if value > 0:
            gray_frame[x, y] = replace_val
    '''
    
    return thresh_frame


# In[7]:


def estimate(frame, radius_lst):
    # convert frame to grayscale
    if len(frame.shape) != 2:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame

    gray_frame = rough_corneal_remove(gray_frame, np.mean(gray_frame))

    #frst
    frst_sum = np.zeros(gray_frame.shape)
    for i in radius_lst:
        result = frst(gray_frame, i, 2, 0.2, 0, mode='DARK')
        frst_sum = np.add(frst_sum, result)

    # argmin finds the min index as if frst_sum is flattened,
    # so we have to reverse that using unravel_index
    eye_estimate = np.unravel_index(frst_sum.argmin(), frst_sum.shape)[::-1]
    
    return eye_estimate


# In[8]:


def starburst(frame, eye):
    PIVOT_ANGLE = 15
    LINE_LENGTH = 20
    DETECTION_THRESHOLD = 5

    # convert frame to grayscale
    if len(frame.shape) != 2:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame

    gray_frame = cv2.blur(gray_frame, (3, 3))

    feature_points = []

    for degree in range(0, 360, PIVOT_ANGLE):

        radian = degree * math.pi / 180
        edge = (int(eye[0] + LINE_LENGTH * math.cos(radian)), 
                int(eye[1] + LINE_LENGTH * math.sin(radian)))
        
        # [(x, y, intensity), ...]
        line_points = [tupl for tupl in createLineIterator(eye, edge, gray_frame)]
        derivative = []

        prev_intensity = None
        for tupl in line_points:
            if prev_intensity is not None:
                derivative.append(tupl[2] - prev_intensity)            
            prev_intensity = tupl[2]
        
        info = []
        for i in range(len(derivative)):

            # skip ahead to next positive slope
            if len(info) > 0:
                if info[-1]['end_idx'] > i:
                    continue

            if derivative[i] > DETECTION_THRESHOLD:

                info.append({'peak_idx': i, 'end_idx': -1, 'corneal': False})

                for j in range(i + 1, len(derivative)):
                    
                    # going through first slope
                    if info[-1]['end_idx'] < 0:

                        if derivative[j] > derivative[i]:
                            info[-1]['peak_idx'] = j

                        elif derivative[j] < DETECTION_THRESHOLD:
                            info[-1]['end_idx'] = j

                    # finding if corneal reflection
                    else:
                        if derivative[j] < -1 * DETECTION_THRESHOLD:
                            info[-1]['corneal'] = True
                        
                        # hit another positive slope
                        elif derivative[j] > DETECTION_THRESHOLD:
                            break

        # filter out corneal reflections
        info = [peak for peak in info if not peak['corneal']]

        # sort peaks by intensity, pick lowest one
        info = sorted(info, key=lambda peak: derivative[peak['peak_idx']])

        if len(info) > 0:
            feature_points.append([line_points[info[0]['peak_idx']][0], 
                                line_points[info[0]['peak_idx']][1]])
                        
        

    return np.array(feature_points)


# In[14]:


def single_thread_placeholder():
    file_path = "fixedSFblink.avi"
    vidcap = cv2.VideoCapture(file_path)

    ret, frame = vidcap.read()
    ret, frame = vidcap.read()
    crop_frame = frame[80:175, 100:200]
    blur_frame = cv2.blur(crop_frame, (3, 3))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vidwrite = cv2.VideoWriter('track_test' + str(datetime.datetime.now()) + '.avi', fourcc, 100, (crop_frame.shape[1], crop_frame.shape[0]))

    progress_bar = tqdm(total=int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)))

    while ret:
        eye = estimate(crop_frame, [12, 14])

        for i in range(5):
            contour = starburst(crop_frame, (eye[0], eye[1]))

            fitted_ellipse = None
            if len(contour) >= 5:
                fitted_ellipse = cv2.fitEllipse(contour)

            # drawing
            for point in contour:
                cv2.circle(blur_frame, tuple(point), 1, (0, 255, 0))
            cv2.circle(blur_frame, tuple((eye[0], eye[1])), 5, (0, 0, 255))

            if fitted_ellipse:
                cv2.ellipse(blur_frame, fitted_ellipse, (255, 255, 255), 1)
            vidwrite.write(blur_frame)

            ret, frame = vidcap.read()
            if not ret:
                break
            crop_frame = frame[80:175, 100:200]
            blur_frame = cv2.blur(crop_frame, (3, 3))
            progress_bar.update(n=1)

    vidcap.release()
    vidwrite.release()

# In[ ]:
import os
os.environ['DISPLAY'] = ':0'

file_path = "fixedSFblink.avi"
vidcap = cv2.VideoCapture(file_path)

ret, frame = vidcap.read()
ret, frame = vidcap.read()

cv2.imshow('test window', frame)
