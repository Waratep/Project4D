import cv2
import numpy as np
import time
from skimage.io import imshow,imread
from scipy import ndimage as ndi
from skimage import feature
import matplotlib.image as mpimg
from matplotlib import pyplot as plt

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    channel_count = img.shape[2]
    match_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

if __name__ == "__main__":

    url = '../video/video1.mjpeg'
    cap = cv2.VideoCapture(url)
    
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output1.avi',fourcc, 10, (128,72),1)

    while cap.isOpened():
        ret, frame = cap.read()

        frame = rescale_frame(frame,10)
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
    #     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
        height=frame.shape[0]
        width=frame.shape[1]
        region_of_interest_vertices = [   
            (0, height),
            (0, height-320),
            (200, height-500),

            (width-200, height-500),
            (width, height-320),
            (width, height)]


        cropped_image = region_of_interest(frame,np.array([region_of_interest_vertices], np.int32))
        img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

        lowerY = np.array([230,230,110])
        upperY = np.array([250,250,204])    
        maskY = cv2.inRange(img, lowerY, upperY)

        lowerW = np.array([231,231,231])
        upperW = np.array([255,255,255])    
        maskW = cv2.inRange(img, lowerW, upperW)

        lowerR = np.array([220,80,90])
        upperR = np.array([255,150,150])    
        maskR = cv2.inRange(img, lowerR, upperR)

        mask = maskY + maskW + maskR 
        cv2.imshow('frame', mask)

        imshow(img)
        # print(img.shape)
        out.write(cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB))
        
    #     time.sleep(.100)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()