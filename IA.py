import cv2
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def normal_orb(img1,img2):
    # Initiate SIFT detector
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:50], None,flags=0)
    plt.imshow(img3),plt.show()
    '''good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)'''
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    im_out=cv2.warpPerspective(img1,M,(img2.shape[1],img2.shape[0])) #img2 is destination image.
    cv2.imshow("Warped img",im_out)
    cv2.imwrite('warped.jpg',im_out)
    cv2.waitKey(0)
    return M,mask
    #M,mask=cv2.findHomography(np.array(kp1),np.array(kp2),cv2.RANSAC, 5.0)


#def warp_image(img1,h,

def findROI(img, face_radii, face_centers, image_no):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    radius=face_radii[image_no]
    centerx=face_centers[image_no][0]
    centery=face_centers[image_no][1]
    radius=face_radii[image_no]
    x=max(centerx - radius*3/2,0)
    y=max(centery - radius*3/2,0)
    row,col=img.shape[0],img.shape[1]
    x_max = min(col-1, x+radius*3)
    y_max = min(row-1, x+radius*3)
    w=x_max-x
    h=y_max-y 
    #rectimg=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    #Tryna crop the image out.
    roi_gray1 = gray[y:y+h, x:x+w]
    roi_color1 = img[y:y+h, x:x+w]
    cv2.imshow("cropped",roi_color1)
    cv2.waitKey(0)
    return roi_color1,roi_gray1

def detect_face(pic):
    #Face detection for first image.
    alldetect=[]
    face_radii=[]
    face_centers=[]
    center=[]           #dummy array
    person_list=0
    gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        pic = cv2.rectangle(pic,(x,y),(x+w,y+h),(255,0,0),2)
        #roi_gray = gray[y:y+h, x:x+w]
        #roi_color = img[y:y+h, x:x+w]
        face_radii.append(h/2)
        center=[]
        center.append(x+w/2)
        center.append(y+w/2)
        face_centers.append(center)
        #person_list.append(faces)
        person_list+=1
    cv2.imshow('imgdet',pic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return face_radii, face_centers, person_list        #Person_list is the no. of people.

import cv2
import numpy as np,sys

# function takes in paramters path to both images to be appended
# below u have to change some code for append according to the homography code

def blend_image(A,B):
    #A = cv2.imread(imagepath1)
    #B = cv2.imread(imagepath2)

    # generate Gaussian pyramid for A
    G = A.copy()
    gpA = [G]
    for i in range(6):
        G = cv2.pyrDown(gpA[i])
        gpA.append(G)
        
    # generate Gaussian pyramid for B
    G = B.copy()
    gpB = [G]
    for i in range(6):
        G = cv2.pyrDown(gpB[i])
        gpB.append(G)

    lpA = [gpA[5]]
    for i in range(5,0,-1):
        size = (gpA[i-1].shape[1], gpA[i-1].shape[0])
        GE = cv2.pyrUp(gpA[i], dstsize = size)
        L = cv2.subtract(gpA[i-1],GE)
        lpA.append(L)

    lpB = [gpB[5]]
    for i in range(5,0,-1):
        size = (gpB[i-1].shape[1], gpB[i-1].shape[0])
        GE = cv2.pyrUp(gpB[i], dstsize = size)
        L = cv2.subtract(gpB[i-1],GE)
        lpB.append(L)

    # Now add left and right halves of images in each level
    #here write your own function of appending one part of image into another
    #like for appending half of 1st image into half of second bottom demo code below
    
    LS = []
    for la,lb in zip(lpA,lpB):
        rows,cols,dpt = la.shape
        ls = np.hstack((la[:,0:int(cols/2)], lb[:,int(cols/2):]))
        LS.append(ls)


    ls_ = LS[0]
    for i in range(1,6):
        size = (LS[i].shape[1], LS[i].shape[0])
        ls_ = cv2.pyrUp(ls_, dstsize = size)
        ls_ = cv2.add(ls_, LS[i])


    # image with direct connecting each half
    # write same code from above instead of below line
    
    real = np.hstack((A[:,:int(3*cols/4)],B[:,int(3*cols/4):]))

    # function returns pyramid blended and irect blended as 2 returns
    toret=[ls_,real]
    return toret


#imagePath="C:/Users/Anannya Uberoi/Downloads/Khwaja_Ghosh/PerfectMoments-code/images/images/3/photo1.jpg"
cascPath = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascPath)

img = cv2.imread('test_images/3/photo1.jpg')
img1=cv2.imread('test_images/3/photo2.jpg')

# img = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)

#Face detection for first image.
pic=deepcopy(img)
pic1=deepcopy(img1)
face_radii, face_centers, person_list=detect_face(pic)
face_radii1, face_centers1, person_list1=detect_face(pic1)
cv2.imshow('after face',img)


cv2.namedWindow('img',cv2.WINDOW_NORMAL)
cv2.resizeWindow('img', 100,100)

face_radii.sort(key=dict(zip(face_radii, face_centers)).get)
face_centers=sorted(face_centers)
face_radii1.sort(key=dict(zip(face_radii1, face_centers1)).get)
face_centers1=sorted(face_centers1)



for x in range (0,1):
    roi_color=findROI(img,face_radii,face_centers,x)[0]
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

for x in range (0,1):
    roi_color1=findROI(img1,face_radii1,face_centers1,x)[0]
    cv2.imshow('img1',img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cv2.imshow('a',roi_color1)
cv2.imshow('b',roi_color)
M,mask=normal_orb(roi_color1,roi_color)
toret=blend_image(img,img1)
#int x_max = min((images.at(0)).cols-1, x+radius*3);
