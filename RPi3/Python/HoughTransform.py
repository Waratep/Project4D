import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import square,dilation
from skimage.io import imshow
import time
import copy
class HoughTransform:    
    def __init__(self,threshold = 40):
        self.url = '../video/my_video.h264'
        self.cap = cv2.VideoCapture(self.url)
        self.maskR = False
        self.ret = 0
        self.frame = 0
        self.slope_right = 0
        self.slope_left = 0
        self.slope = 0
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.threshold = threshold
    
    def region_of_interest(self,img, vertices):   
        mask = np.zeros_like(img)
        match_mask_color = (255,)
        cv2.fillPoly(mask, vertices, 255)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image
    
    def findSlope(self,start,end,height):
        x1 = start[0]
        x2 = end[0]
        y1 = height-start[1]
        y2 = height-end[1]
        if (x2 - x1) != 0:
            m = (y2-y1)/(x2-x1)
            return m 

    def avgSlope(self,li):             
        x1 = round(sum(li[:,0])/len(li)).astype('int')
        x2 = round(sum(li[:,1])/len(li)).astype('int')
        y1 = round(sum(li[:,2])/len(li)).astype('int')
        y2 = round(sum(li[:,3])/len(li)).astype('int')
        return x1,x2,y1,y2

    def debug(self,debug = False):
        if (debug):            
            #cv2.putText(self.frame,str(self.slope_right),(100,25), self.font, 0.5,(255,255,255),2,cv2.LINE_AA)
            print('slope_right :', self.slope_right)            
            print('slope_left :', self.slope_left)            
            print('slope :', self.slope)
            print()
            
    def run(self,frame=None,fromvideo=False,debugs = False):
        if(fromvideo):
            while self.cap.isOpened():
                self.ret, self.frame = self.cap.read()
                if not self.ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break                
                if cv2.waitKey(1) == ord('q'):
                    break
                self.corerun()                    
                self.debug(debug = debugs)
            self.cap.release()
            cv2.destroyAllWindows()
        else:
            self.frame = frame
            while cv2.waitKey(1) != ord('q'):                
                self.corerun()  
                self.debug(debug = debugs)
                return float(self.slope),self.maskR
            self.cap.release()
            cv2.destroyAllWindows()
            
    def corerun(self):
        self.maskR = False
        self.frame = cv2.resize(self.frame,(100, 75)) 
        height=self.frame.shape[0]
        width=self.frame.shape[1]
        
        self.frame=cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)    
        lowerR=np.array([150,100,100])
        upperR=np.array([255,190,190])    
        maskR = cv2.inRange(self.frame, lowerR, upperR)
        
        countR=0
        for i in range(width):            
            if maskR[30][i] == 255 or maskR[31][i] == 255 or maskR[32][i] == 255:
                countR+=1
#                 print(countR)
            if countR > 10:
                self.maskR = True
                # print('stop')
                break
        
        gray=cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray,100,200)
        edges = dilation(edges, square(3))
        
        # crop image
        region_of_interest_vertices = [   
            (0, height),
            (0, (height/3)+10),
            (width, (height/3)+10),
            (width, height)]
        crop = self.region_of_interest(edges,np.array([region_of_interest_vertices], np.int32))
        crop = dilation(crop, square(3))
        lines = cv2.HoughLinesP(crop,rho = 1,theta = 1*np.pi/180,threshold = 40,minLineLength = 10,maxLineGap = 250)
    
        dframe = copy.deepcopy(self.frame) 
        l=[]
        r=[]        
        # Draw lines on the image
        if(lines is not None) :
            for line in lines:  
                x1, y1, x2, y2 = line[0]
                if(self.findSlope((x1,y1),(x2,y2),height)):            
                    if(x1 < int(width/2)):
                        l.append([x1,x2,y1,y2,self.findSlope((x1,y1),(x2,y2),height)])
                    else:
                        r.append([x1,x2,y1,y2,self.findSlope((x1,y1),(x2,y2),height)])
                        
            l=np.array(l)
            r=np.array(r)

            if(len(l)):                
                x1,x2,y1,y2 = self.avgSlope(l)
                self.slope_left = self.findSlope((x1,y1),(x2,y2),width)
                #print(x1,x2,y1,y2,'slope_left',slope_left)
                cv2.line(dframe, (x1, y1), (x2, y2), (255, 0, 0), 3)

            if(len(r)):                
                x1,x2,y1,y2 = self.avgSlope(r)
                self.slope_right = self.findSlope((x1,y1),(x2,y2),width)
                #print(x1,x2,y1,y2,'slope_right',slope_right)
                cv2.line(dframe, (x1, y1), (x2, y2), (0, 0, 255), 3)

            self.slope = round(self.slope_right + self.slope_left,3)
            if(self.slope > 2):
                self.slope = 2
            elif(self.slope < -2):
                self.slope = -2   

        
        cv2.imshow('frame', dframe)  
            