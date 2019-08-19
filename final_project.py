# IMPORTING THE REQUIRED LIBRARIES
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import os

#GETTING CURRENT DIRECTORIES AND FOLDER PATH REQUIRED
folpath=os.getcwd()
img_data = "room"
imgpaths=folpath+"/input_images/"+img_data+"/"

#READING IMAGES
img1 = cv2.imread(imgpaths+img_data+'-00.png')

def warpImages(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0,0], [0,rows1], [cols1,rows1], [cols1,0]]).reshape(-1,1,2)
    temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)
    

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
    translation_dist = [-x_min,-y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]])
       
    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
    FrameSize = output_img.shape
    NewImage = img2.shape
    output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1
    
    OriginR = int(list_of_points_2[0][0][1])
    OriginC = int(list_of_points_2[0][0][0])
    
    # if the origin of projected image is out of bounds, then mapping to ()
    if OriginR < 0:
        OriginR = 0
    if OriginC < 0:
        OriginC = 0
        
    # Clipping the new image, if it's size is more than the frame    
    if NewImage[0] > FrameSize[0]-OriginR:
        img2 = img2[0:FrameSize[0]-OriginR,:]
        
    if NewImage[1] > FrameSize[1]-OriginC:
        img2 = img2[:,0:FrameSize[1]-OriginC]    
            
    output_img[OriginR:NewImage[0]+OriginR, OriginC:NewImage[1]+OriginC] = img2    
    
    return output_img

def giveMosaic(FirstImage, no):
    
    ImgList = []          # No of images stitched
    Matches = []         # this stores the number of good matches at every stage
    i = 1
    
    heightM, widthM = FirstImage.shape[:2]
    RecMosaic = FirstImage
    
    for name in images[1:]:
        
        print (name)
        image = cv2.imread(name) 
        height, width = image.shape[:2]
    
        ######Feature detection and matches######
        orb = cv2.ORB_create(nfeatures=10000)
        kp1, des1 = orb.detectAndCompute(RecMosaic, None)
        kp2, des2 = orb.detectAndCompute(image, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des1, des2,k=2)
        
        ## store all the good matches as per Lowe's ratio test ##
        good = []
        allPoints = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        
        Matches.append(len(good))
        print ("Good_Matches:", len(good))
       
        #### Finding the homography #########
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,6.0)
        
        i+=1
        RecMosaic = warpImages(RecMosaic, image, M)
        print (i)
        ImgList.append(i)
        if i==40:
            break
        
    cv2.imwrite("FinalMosaic"+"_"+img_data+".jpg", RecMosaic)
    return ImgList, Matches


images = sorted(glob.glob(imgpaths+'/*.png'))    # for reading images
n = 10000;                                      # no of features to extract
ImgNumbers, GoodMatches = giveMosaic(img1, 10000)







