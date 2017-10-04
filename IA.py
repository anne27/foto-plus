import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    return M,mask
    #M,mask=cv2.findHomography(np.array(kp1),np.array(kp2),cv2.RANSAC, 5.0)
    

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
    rectimg=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    #Tryna crop the image out.
    roi_gray1 = gray[y:y+h, x:x+w]
    roi_color1 = img[y:y+h, x:x+w]
    cv2.imshow("cropped",roi_color1)
    cv2.waitKey(0)
    return roi_color1,roi_gray1

def detect_face(img):
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

#imagePath="C:/Users/Anannya Uberoi/Downloads/Khwaja_Ghosh/PerfectMoments-code/images/images/3/photo1.jpg"
cascPath = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascPath)

img = cv2.imread('test_images/pic1.jpg')
img1=cv2.imread('test_images/pic2.jpg')
# img = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)

#Face detection for first image.
face_radii, face_centers, person_list=detect_face(img)
face_radii1, face_centers1, person_list1=detect_face(img1)

for x in range (0,(person_list)):
    roi_color=findROI(img,face_radii,face_centers,x)[0]
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

for x in range (0,(person_list1)):
    roi_color1=findROI(img1,face_radii1,face_centers1,x)[0]
    cv2.imshow('img',img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

M,mask=normal_orb(roi_color1,roi_color)

#int x_max = min((images.at(0)).cols-1, x+radius*3);
'''
def find_homography(refImg, curImg, pivot_ref, pivot_cur):
    images=[]
    images.append(curImg)
    images.append(refImg)
    keypoints=[]
    descriptors=[]
    #orb detector(5000)
    orb = cv2.ORB()
    nI=2
    count=0
    for image in images:
        count+=1
        cv2.cvtColor(*img,img_gray,CV_RGB2GRAY)
        keypt=[]                        #Keypoints array
        kp = orb.detect(img_gray,None)  #ORB detects keypoints in grayscale img.
        keypoints.append(kp)            #Add these points in the final list.
        descriptors.append(descriptor)
        #This can be alternatively used:
        #kp1, des1 = orb.detectAndCompute(img1,None)
        #kp2, des2 = orb.detectAndCompute(img2,None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)       #Brute force matcher to match the keypoints.
        matches = bf.match(des1,des2)
        matches = sorted(matches, key = lambda x:x.distance)        #Store the matches in a list.

'''
'''vector<int> queryIdxs( matches.size() ), trainIdxs( matches.size() );
	for( size_t i = 0; i < matches.size(); i++ ) {
		queryIdxs[i] = matches[i].queryIdx;
		trainIdxs[i] = matches[i].trainIdx;
	}
        #Now find keypoints only corresponding to the given ROIs.
Mat find_homography(Mat refImg, Mat curImg, Point pivot_ref, Point pivot_cur)
{


    int nImage = 0;

	//Find Homography
	vector<int> queryIdxs( matches.size() ), trainIdxs( matches.size() );
	for( size_t i = 0; i < matches.size(); i++ ) {
		queryIdxs[i] = matches[i].queryIdx;
		trainIdxs[i] = matches[i].trainIdx;
	}

	vector<Point2f> points1; KeyPoint::convert(keypoints.at(nImage), points1, queryIdxs);
	vector<Point2f> points2; KeyPoint::convert(keypoints.at(numImages-1), points2, trainIdxs);

	// the following is done since we are finding the homography between the 2 images only in the
	// specified roi. The following bring the coordinate center of each of these image rois to their
	// respective image coordinate centers
	for( int i1 = 0; i1 < points1.size(); i1++ )
	{
		points1.at(i1) += Point2f(pivot_cur.x, pivot_cur.y);
		points2.at(i1) += Point2f(pivot_ref.x, pivot_ref.y);
	}

	// find the homography matrix H
	Mat H = findHomography( Mat(points1), Mat(points2), CV_RANSAC);
	
	/*
		// to visualize the matches
		Mat drawImg;


        vector<char> matchesMask( matches.size(), 0 );
        Mat points1t; perspectiveTransform(Mat(points1), points1t, H);

        double maxInlierDist = 3;
        for( size_t i1 = 0; i1 < points1.size(); i1++ )
        {
            if( norm(points2[i1] - points1t.at<Point2f>((int)i1,0)) <= maxInlierDist ) // inlier
                matchesMask[i1] = 1;
        }
        // draw inliers
        drawMatches( images.at(0), keypoints.at(0), images.at(1), keypoints.at(1), matches, drawImg, CV_RGB(0, 255, 0), CV_RGB(0, 0, 255), matchesMask);
        imshow( "matches", drawImg );
        waitKey(0);
		
		*/
		
	
	return H;
}
'''
