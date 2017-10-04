import cv2
import numpy as np,sys

# function takes in paramters path to both images to be appended
# below u have to change some code for append according to the homography code

def blend_image(imagepath1,imagepath2):
    A = cv2.imread(imagepath1)
    B = cv2.imread(imagepath2)

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

    # LS = []
    # for la,lb in zip(lpA,lpB):
    #     rows,cols,dpt = la.shape
    #     ls = np.hstack((la[:,0:int(cols/2)], lb[:,int(cols/2):]))
    #     LS.append(ls)


    ls_ = LS[0]
    for i in range(1,6):
        size = (LS[i].shape[1], LS[i].shape[0])
        ls_ = cv2.pyrUp(ls_, dstsize = size)
        ls_ = cv2.add(ls_, LS[i])


    # image with direct connecting each half
    # write same code from above instead of below line
    
    # real = np.hstack((A[:,:int(3*cols/4)],B[:,int(3*cols/4):]))

    # function returns pyramid blended and irect blended as 2 returns
    toret=[ls_,real]

    return toret