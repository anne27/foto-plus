import cv2
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import sys
import os
import scipy
from scipy.stats import norm
from scipy.signal import convolve2d
import math
import copy
import operator




'''split rgb image to its channels'''
def split_rgb(image):
    red = None
    green = None
    blue = None
    (blue, green, red) = cv2.split(image)
    return red, green, blue
 
'''generate a 5x5 kernel'''
def generating_kernel(a):
    w_1d = np.array([0.25 - a/2.0, 0.25, a, 0.25, 0.25 - a/2.0])
    return np.outer(w_1d, w_1d)
 
'''reduce image by 1/2'''
def ireduce(image):
    out = None
    kernel = generating_kernel(0.4)
    outimage = scipy.signal.convolve2d(image,kernel,'same')
    out = outimage[::2,::2]
    return out
 
'''expand image by factor of 2'''
def iexpand(image):
    out = None
    kernel = generating_kernel(0.4)
    outimage = np.zeros((image.shape[0]*2, image.shape[1]*2), dtype=np.float64)
    outimage[::2,::2]=image[:,:]
    out = 4*scipy.signal.convolve2d(outimage,kernel,'same')
    return out
 
'''create a gaussain pyramid of a given image'''
def gauss_pyramid(image, levels):
    output = []
    output.append(image)
    tmp = image
    for i in range(0,levels):
        tmp = ireduce(tmp)
        output.append(tmp)
    return output
 
'''build a laplacian pyramid'''
def lapl_pyramid(gauss_pyr):
    output = []
    k = len(gauss_pyr)
    for i in range(0,k-1):
        gu = gauss_pyr[i]
        egu = iexpand(gauss_pyr[i+1])
        if egu.shape[0] > gu.shape[0]:
             egu = np.delete(egu,(-1),axis=0)
        if egu.shape[1] > gu.shape[1]:
            egu = np.delete(egu,(-1),axis=1)
        output.append(gu - egu)
    output.append(gauss_pyr.pop())
    return output

'''Blend the two laplacian pyramids by weighting them according to the mask.'''
def blend(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask):
    blended_pyr = []
    k= len(gauss_pyr_mask)
    for i in range(0,k):
     p1= gauss_pyr_mask[i]*lapl_pyr_white[i]
     p2=(1 - gauss_pyr_mask[i])*lapl_pyr_black[i]
     blended_pyr.append(p1 + p2)
    return blended_pyr

'''Reconstruct the image based on its laplacian pyramid.'''
def collapse(lapl_pyr):
    output = None
    output = np.zeros((lapl_pyr[0].shape[0],lapl_pyr[0].shape[1]), dtype=np.float64)
    for i in range(len(lapl_pyr)-1,0,-1):
        lap = iexpand(lapl_pyr[i])
        lapb = lapl_pyr[i-1]
        if lap.shape[0] > lapb.shape[0]:
            lap = np.delete(lap,(-1),axis=0)
        if lap.shape[1] > lapb.shape[1]:
            lap = np.delete(lap,(-1),axis=1)
        tmp = lap + lapb
        lapl_pyr.pop()
        lapl_pyr.pop()
        lapl_pyr.append(tmp)
        output = tmp
    return output

#functions takes in the 2 images to be blended along with a mask matrix which should have 0 where not to be blended and 1 where img2 should replace img1

def blendimg(image1,image2,mask):
  mask=mask
  r1=None
  g1=None
  b1=None
  r2=None
  g2=None
  b2=None
  rm=None
  gm=None
  bm=None

  (r1,g1,b1)=split_rgb(image1)
  (r2,g2,b2)=split_rgb(image2)
  (rm,gm,bm)=split_rgb(mask)

  r1=r1.astype(float)
  g1=g1.astype(float)
  b1=b1.astype(float)

  r2=r2.astype(float)
  g2=g2.astype(float)
  b2=b2.astype(float)

  rm=rm.astype(float)/255
  gm=gm.astype(float)/255
  bm=bm.astype(float)/255

  #Automaticallyfigureoutthesize
  min_size=min(r1.shape)
  depth=int(math.floor(math.log(min_size,2)))-4#atleast16x16atthehighestlevel.

  gauss_pyr_maskr=gauss_pyramid(rm,depth)
  gauss_pyr_maskg=gauss_pyramid(gm,depth)
  gauss_pyr_maskb=gauss_pyramid(bm,depth)

  gauss_pyr_image1r=gauss_pyramid(r1,depth)
  gauss_pyr_image1g=gauss_pyramid(g1,depth)
  gauss_pyr_image1b=gauss_pyramid(b1,depth)

  gauss_pyr_image2r=gauss_pyramid(r2,depth)
  gauss_pyr_image2g=gauss_pyramid(g2,depth)
  gauss_pyr_image2b=gauss_pyramid(b2,depth)

  lapl_pyr_image1r=lapl_pyramid(gauss_pyr_image1r)
  lapl_pyr_image1g=lapl_pyramid(gauss_pyr_image1g)
  lapl_pyr_image1b=lapl_pyramid(gauss_pyr_image1b)

  lapl_pyr_image2r=lapl_pyramid(gauss_pyr_image2r)
  lapl_pyr_image2g=lapl_pyramid(gauss_pyr_image2g)
  lapl_pyr_image2b=lapl_pyramid(gauss_pyr_image2b)

  outpyrr=blend(lapl_pyr_image2r,lapl_pyr_image1r,gauss_pyr_maskr)
  outpyrg=blend(lapl_pyr_image2g,lapl_pyr_image1g,gauss_pyr_maskg)
  outpyrb=blend(lapl_pyr_image2b,lapl_pyr_image1b,gauss_pyr_maskb)

  outimgr=collapse(blend(lapl_pyr_image2r,lapl_pyr_image1r,gauss_pyr_maskr))
  outimgg=collapse(blend(lapl_pyr_image2g,lapl_pyr_image1g,gauss_pyr_maskg))
  outimgb=collapse(blend(lapl_pyr_image2b,lapl_pyr_image1b,gauss_pyr_maskb))
  #blendingsometimesresultsinslightlyoutofboundnumbers.
  outimgr[outimgr<0]=0
  outimgr[outimgr>255]=255
  outimgr=outimgr.astype(np.uint8)

  outimgg[outimgg<0]=0
  outimgg[outimgg>255]=255
  outimgg=outimgg.astype(np.uint8)

  outimgb[outimgb<0]=0
  outimgb[outimgb>255]=255
  outimgb=outimgb.astype(np.uint8)

  result=np.zeros(image1.shape,dtype=image1.dtype)
  tmp=[]
  tmp.append(outimgb)
  tmp.append(outimgg)
  tmp.append(outimgr)
  result=cv2.merge(tmp,result)
  cv2.imshow("blendedimg",result)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  # blendimg("img1.jpg","img2.jpg",mask)
  cv2.imwrite('blendedpro.jpg',result)
  return result









def normal_orb(img1,img2,face_centers,face_centers1,face_radii):
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
    # cv2.imshow("img1 isssss",img1)
    # cv2.waitKey(0)
    # cv2.imshow("img2 isssss",img2)


    cv2.waitKey(0)
    imge1 = cv2.imread('photo1.jpg')
    imge2=cv2.imread('photo2.jpg')

    im_out=cv2.warpPerspective(imge2,M,(imge2.shape[1],imge2.shape[0])) #img2 is destination image.
    # cv2.imshow("Warped img",im_out)

    # cv2.imwrite('warped.jpg',im_out)
    # cv2.waitKey(0)
    fccntr=[]
    fccntr.append(face_centers1[0])
    fccntr=np.array(fccntr)
    print (fccntr)
    print (M)


    dst = cv2.perspectiveTransform(fccntr[None,:,:],M)
    print (dst)

    msk=np.empty(imge1.shape)
    cv2.circle(msk,((int)(dst[0,0,0]),(int)(dst[0,0,1])),(int)(1.2*face_radii[0]),(255,255,255),-1)

    blendimg(imge1,im_out,msk)

    return M,mask
    #M,mask=cv2.findHomography(np.array(kp1),np.array(kp2),cv2.RANSAC, 5.0)


#def warp_image(img1,h,

def findROI(img, face_radii, face_centers, image_no):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    radius=face_radii[image_no]
    centerx=face_centers[image_no][0]
    centery=face_centers[image_no][1]
    radius=face_radii[image_no]
    x=(int)(max(centerx - radius*3/2,0))
    y=(int)(max(centery - radius*3/2,0))
    row,col=img.shape[0],img.shape[1]
    x_max = min(col-1, x+radius*3)
    y_max = min(row-1, x+radius*3)
    w=(int)(x_max-x)
    h=(int)(y_max-y) 
    #rectimg=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    #Tryna crop the image out.
    roi_gray1 = gray[y:y+h, x:x+w]
    roi_color1 = img[y:y+h, x:x+w]
    # cv2.imshow("cropped",roi_color1)
    # cv2.waitKey(0)
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
    # cv2.imshow('imgdet',pic)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
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

img = cv2.imread('photo1.jpg')
img1=cv2.imread('photo2.jpg')

# img = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)

#Face detection for first image.
pic=deepcopy(img)
pic1=deepcopy(img1)
face_radii, face_centers, person_list=detect_face(pic)
face_radii1, face_centers1, person_list1=detect_face(pic1)
# cv2.imshow('after face',img)


cv2.namedWindow('img',cv2.WINDOW_NORMAL)
cv2.resizeWindow('img', 100,100)

face_radii.sort(key=dict(zip(face_radii, face_centers)).get)
face_centers=sorted(face_centers)
face_radii1.sort(key=dict(zip(face_radii1, face_centers1)).get)
face_centers1=sorted(face_centers1)



for x in range (0,1):
    roi_color=findROI(img,face_radii,face_centers,x)[0]
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

for x in range (0,1):
    roi_color1=findROI(img1,face_radii1,face_centers1,x)[0]
    # cv2.imshow('img1',img1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# cv2.imshow('a',roi_color1)
# cv2.imshow('b',roi_color)
M,mask=normal_orb(img1,img,face_centers,face_centers1,face_radii)
toret=blend_image(img,img1)
#int x_max = min((images.at(0)).cols-1, x+radius*3);
