import cv2
import numpy as np
from scipy import ndimage as ndi
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from skimage.morphology import square,dilation
import copy
from scipy import ndimage as ndi
import time

class Imageprocessing:

    def __init__(self,filename = 'test.h264'):

        self.url = '../video/' + filename
        self.cap = cv2.VideoCapture(self.url)
        self.count = 0
        self.left = []
        self.right = []
        self.ret = 0
        self.frame = 0
        self.slope_right = 0
        self.slope_left = 0
        self.font = 0
        
    def _debug(self,debug = False):

        if (debug):

            cv2.putText(self.frame,str(self.slope_right),(100,25), self.font, 0.5,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(self.frame,str(self.slope_left),(10,25), self.font, 0.5,(255,255,255),2,cv2.LINE_AA)
            print('slope_left:', self.slope_left,' slope_right:', self.slope_right,' slope:', self.slope_right + self.slope_left)
            cv2.imshow('frame', self.frame)


    def _run(self,fromvideo = True,debugs = False):

        if(fromvideo):

            while self.cap.isOpened():
                self.ret, self.frame = self.cap.read()

                if not self.ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break

                self._debug(debug = debugs)

                if cv2.waitKey(1) == ord('q'):
                    break
                self._corerun()

        else:
            pass
            
    def _corerun(self):

        self.frame = cv2.resize(self.frame,(300, 200))
        self.height = self.frame.shape[0]
        self.width = self.frame.shape[1]

        self.edges = cv2.Canny(self.frame,100,200)
        self.edges = dilation(self.edges, square(3))
        self.edges[:,self.width - 3:self.width] = 0
        self.edges = ndi.gaussian_filter(self.edges,3)

        self.region_of_interest_vertices = [
            (0, self.height),
            (0, (self.height/3)),
            (self.width, (self.height/3)),
            (self.width, self.height)]

        self.crop = self.region_of_interest(self.edges,np.array([self.region_of_interest_vertices], np.int32))
        self.lines = cv2.HoughLinesP(self.crop,rho = 1,theta = 1*np.pi/180,threshold = 155,minLineLength = 10,maxLineGap = 250)
        # print(lines)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        # Draw lines on the image
        if(self.lines is not None) :
            for line in self.lines:
                self.x1, self.y1, self.x2, self.y2 = line[0]
                if(self.slope((self.x1,self.y1),(self.x2,self.y2),self.width)):
                    if(self.x1 < int(self.width/2)):
                        self.left.append(self.slope((self.x1,self.y1),(self.x2,self.y2),self.width))
                        cv2.line(self.frame, (self.x1, self.y1), (self.x2, self.y2), (255, 0, 0), 3)  
                    else:
                        self.right.append(self.slope((self.x1,self.y1),(self.x2,self.y2),self.width))
                        cv2.line(self.frame, (self.x1, self.y1), (self.x2, self.y2), (0, 0, 255), 3)  

            if(len(self.left)):
                self.slope_left = round(sum(self.left)/len(self.left),2)
            else:
                self.slope_left = 0

            if(len(self.right)):
                self.slope_right = round(sum(self.right)/len(self.right),2)
            else:
                self.slope_right = 0
            


    def region_of_interest(self,img, vertices):
        mask = np.zeros_like(img)
        match_mask_color = (255,)
        cv2.fillPoly(mask, vertices, 255)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def slope(self,start,end,width):
        x1 = start[0]
        x2 = end[0]
        y1 = width-start[1]
        y2 = width-end[1]
        if (x2 - x1) != 0:
            m = (y2-y1)/(x2-x1)
            return m

