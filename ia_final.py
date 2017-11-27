import sys
import os
import numpy as np
import cv2
import scipy
from scipy.stats import norm
from scipy.signal import convolve2d
import math
import copy
import operator


'''find ROI from face'''
def findROI(img, face_radii, face_centers, image_no):
    print('here')
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
    print('hereeee')
    rectimg=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    #Tryna crop the image out.
    roi_gray1 = gray[y:y+h, x:x+w]
    roi_color1 = img[y:y+h, x:x+w]
    print(x,y,w,h)
    cv2.imshow("cropped",roi_color1)
    cv2.imwrite('cropped.jpg',roi_color1)
    cv2.waitKey(0)
    return roi_color1,x,y,w,h

def detect_face(img):
    print('hereeeeeeeeee')
    #Face detection for first image.
    alldetect=[]
    face_radii=[]
    face_centers=[]
    center=[]           #dummy array
    person_list=0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #roi_gray = gray[y:y+h, x:x+w]
        #roi_color = img[y:y+h, x:x+w]
        face_radii.append(h/2)
        center=[]
        center.append(x+w/2)
        center.append(y+w/2)
        face_centers.append(center)
        #person_list.append(faces)
        person_list+=1
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return face_radii, face_centers, person_list        #Person_list is the no. of people.

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

if __name__=='__main__':
  cascPath = "haarcascade_frontalface_default.xml"
  image1_path="photo1.jpg"        #This is the base image.
  #image2_path="test_images/6/photo2.jpg"       #This is the image from which face is pasted.
  image2_path="warped.png"
  image1=cv2.imread(image1_path)
  image2=cv2.imread(image2_path)
  image1_copy=copy.deepcopy(image1)
  image2_copy=copy.deepcopy(image2)
  
  face_cascade = cv2.CascadeClassifier(cascPath)
  face_radii, face_centers, person_list=detect_face(image1)
  
  keydict = dict(zip(face_radii, face_centers))
  face_radii.sort(key=keydict.get)
  face_centers=sorted(face_centers)

  print (face_centers)

  for x in range (0,1):
    roi_color,x,y,w,h=findROI(image1,face_radii,face_centers,x)
    cv2.imshow('img',image1)
    cv2.imshow('roi',roi_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

  mask=np.empty(image1.shape)
  
  mask[y:y+h,x:x+w]=255
  #mask[:,int(image1.shape[1]/2):,:]=255
  cv2.imshow("a",mask)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
  image3=np.empty(image1.shape)
  image2=cv2.resize(image2,(w,h), interpolation = cv2.INTER_AREA)
  image3[y:y+h,x:x+w]=image2
  cv2.imshow('this image',image3)

  blendimg(image1_copy,image3,mask)
