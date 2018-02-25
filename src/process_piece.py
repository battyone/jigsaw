#This script extract features abut puzzle peace
import numpy as np
import cv2
import scipy
from scipy.ndimage import measurements
from matplotlib import pyplot


#Assumes img has only one piece in it. Returns a mask where 1 is the piece, 0 outside
def locate_piece(img):
    #Extract puzzle peace mask by color
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = img[:,:,1]
    relative_g = g/(gray_image+1); #Relative green propotion
    mask = cv2.inRange(relative_g, 1.10, 1.30)
    tmp,mask = cv2.threshold(mask, 100,1, cv2.THRESH_BINARY_INV);

    #Run Morphological Transformations filter out the main peace
    #Main peace is the largest cluster after background
    labeled_array, n_labels = scipy.ndimage.measurements.label(mask)
    hist,tmp = np.histogram(labeled_array,n_labels)
    hist = hist[1:]
    mask = (labeled_array == (np.argmax(hist)+1))

    return mask

#This function preprocess piece data, finds contour and orientation
#Returns: 
#   cnt - contour of the piece (rotated) 
#   mask - piece mask (rotated)
#   img - piece image (rotated)
#   rect - piece rectangular and rotation at the original image(degrees)
#Inputs:
#is_plot=0 - no plotting, 1 - plot rotated pice, 2 - rotate original pice with contour on it
def preprocess_piece(mask,img, is_plot=0):
    
    #Find piece countour 
    tmp,contours,hierarchy = cv2.findContours(image=mask.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    cnt = contours[0].reshape(-1,2); #We assume only one contour
    
    #Find bounding box with orientation
    rect = cv2.minAreaRect(cnt)

    #Find Center of piece and angle
    center = (np.array(rect[0])+np.array(rect[1]))/2;
    angle = rect[2]*np.pi/180

    #Rotate

    #Rotate contour
    rot = np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]]);
    cnt_rotated = (np.matmul(rot,(cnt-center).T).T+center);

    #Get new bounding box
    x,y,w1,h1 = cv2.boundingRect(cnt_rotated.astype(np.int32))

    #Rotate images
    rotMat = cv2.getRotationMatrix2D(tuple(center),angle*180/np.pi, 1.0)
    ros,cols,tmp = img.shape
    img_rotated = cv2.warpAffine(img,rotMat,(ros,cols));

    #Trim image to fit the piece
    img_rotated=img_rotated[y:(y+h1),x:(x+w1),:]
    cnt_rotated = (cnt_rotated-np.array([x,y])).astype(np.int32);
    mask_rotated = np.zeros(np.array([h1,w1]), int);
    cv2.drawContours(mask_rotated,[cnt_rotated],0,1, cv2.FILLED)

    #Draw (optional
    if (is_plot==1):
        #Draw rotated image
        img2 = img_rotated
        cv2.drawContours(img2,[cnt_rotated],-1,(0,0,255), 10)
        pyplot.imshow(img2)
        pyplot.show()
    elif (is_plot==2):
        #Draw original image with new data
        img2 = img;
        cv2.drawContours(img2,[cnt],-1,(0,0,255), 10)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img2,[box],0,(0,0,255),2)
        pyplot.imshow(img)
        pyplot.show()


    return {'cnt':cnt_rotated, 'mask':mask_rotated, 'img':img_rotated, 'rect':rect }


if __name__ == '__main__':
    img = cv2.imread('../data/unicorn/11.jpg',1) #Load color image
    mask = locate_piece(img);

    piece = preprocess_piece(mask,img,1)
    
    import pdb; pdb.set_trace()