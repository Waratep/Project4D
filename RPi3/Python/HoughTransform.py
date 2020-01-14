import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import square,dilation
import time
class HoughTransform:
    
    def __init__(self,filename = 'test.h264'):
        self.url = '../../video/' + filename
        self.cap = cv2.VideoCapture(self.url)
        self.countR = 0
        self.left = []
        self.right = []
        self.ret = 0
        self.frame = 0
        self.slope_right = 0
        self.slope_left = 0
        self.font = 0
        self.threshold = 40
        self.count=0
        
    def _debug(self,debug = True):
        if (debug):            
#             cv2.putText(self.frame,str(self.slope_right),(100,25), self.font, 0.5,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(self.frame,str(round(self.slope_right + self.slope_left,3)),(10,25), self.font, 0.5,(255,255,255),2,cv2.LINE_AA)
            print('slope_left:', self.slope_left,' slope_right:', self.slope_right,' slope:', round(self.slope_right + self.slope_left,3))
#             if self.countR > 10:
#                 print()

#                 print('----------------------------stop----------------------------')
#                 cv2.putText(self.frame,'stop',(190,25), self.font, 0.5,(0,0,0),2,cv2.LINE_AA)
#             cv2.imshow('frame', self.frame)
    
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
    def run(self,image=0,fromvideo = False,debugs = False):
        if(fromvideo):
            while self.cap.isOpened():
                self.ret, self.frame = self.cap.read()
                if not self.ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break                
                if cv2.waitKey(1) == ord('q'):
                    break
                self.corerun()                    
                self._debug(debug = debugs)
            self.cap.release()
            cv2.destroyAllWindows()
        else:
            self.frame = image
            self.corerun()  
            self._debug(debug = debugs)
            
            return float(round(self.slope_right+self.slope_left))
    def corerun(self): 
        
        self.frame = cv2.resize(self.frame,(100, 75))    
        height=self.frame.shape[0]
        width=self.frame.shape[1]
        region_of_interest_vertices = [   
            (0, height),
            (0, (height/3)+15),
            (width, (height/3)+15),
            (width, height)]
        
        gray=cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)

        edges = cv2.Canny(gray,100,200)
        edges = dilation(edges, square(4))
        # edges[:,width-3:width] = 0 
        # edges = ndi.gaussian_filter(edges,2)
        
        crop = self.region_of_interest(edges,np.array([region_of_interest_vertices], np.int32))
        self.frame=cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)    
        lowerR=np.array([220,70,70])
        upperR=np.array([255,160,170])    
        maskR = cv2.inRange(self.frame, lowerR, upperR)
        
        for i in range(width):        
            if maskR[int(height/3)][i] == 255: 
                self.countR+=1
            if self.countR > 10:
                print('stop')
                break

        # Hough Transform
        lines = cv2.HoughLinesP(crop,rho = 1,theta = 1*np.pi/180,threshold = self.threshold,minLineLength = 10,maxLineGap = 250)
        font = cv2.FONT_HERSHEY_SIMPLEX
        l=[]
        r=[]

        # Draw lines on the image
        if(lines is not None) :
            for line in lines:  
                x1, y1, x2, y2 = line[0]
                if(self.slope((x1,y1),(x2,y2),width)):
                    if(x1 < int(width/2)):
                        l.append([x1,x2,y1,y2,self.slope((x1,y1),(x2,y2),width)])
#                         cv2.line(self.frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    else:
                        r.append([x1,x2,y1,y2,self.slope((x1,y1),(x2,y2),width)])
#                         cv2.line(self.frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

            l=np.array(l)
            r=np.array(r)

            # slope left
            if(len(l)):
                self.slope_left = round(sum(l[:,4])/len(l),2)  
                x1 = round(sum(l[:,0])/len(l)).astype('int')
                x2 = round(sum(l[:,1])/len(l)).astype('int')
                y1 = round(sum(l[:,2])/len(l)).astype('int')
                y2 = round(sum(l[:,3])/len(l)).astype('int')
                cv2.line(self.frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                if(self.slope_left>2):
                    self.slope_left=2
                elif(self.slope_left<-2):
                    self.slope_left=-2
            else:
                self.slope_left = 0

            # slope right
            if(len(r)):
                self.slope_right = round(sum(r[:,4])/len(r),2)
                x1 = round(sum(r[:,0])/len(r)).astype('int')
                x2 = round(sum(r[:,1])/len(r)).astype('int')
                y1 = round(sum(r[:,2])/len(r)).astype('int')
                y2 = round(sum(r[:,3])/len(r)).astype('int')
                cv2.line(self.frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                if(self.slope_right>2):
                    self.slope_right=2
                elif(self.slope_right<-2):
                    self.slope_right=-2
            else:
                self.slope_right = 0  
        self.count+=1       
#         cv2.putText(self.frame,str(self.count),(50,25), font, 0.5,(255,255,255),2,cv2.LINE_AA)
        self.frame=cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)    
        
        time.sleep(0.02)
        cv2.imshow('frame', self.frame)
#         video_list.append(self.frame)
        
        
        