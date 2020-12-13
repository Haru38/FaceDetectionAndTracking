import cv2
import numpy as np

"""
顔のトラッキングを制御するクラス
"""

class meanShift():
    def __init__(self):
        self.term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

    def my_meanShift(self,frame,roi_hist,track_window):
        self.hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.dst = cv2.calcBackProject([self.hsv],[0],roi_hist,[0,180],1)
        self.ret, track_window = cv2.meanShift(self.dst, track_window, self.term_crit)
        return track_window

    #トラッキングするための前処理
    def preProcessing(self,tf,frame):
        self.r,self.h,self.c,self.w = tf[1],tf[3],tf[0],tf[2]
        self.track_window = (self.c,self.r,self.w,self.h)
        self.roi = frame[self.r:self.r+self.h, self.c:self.c+self.w]
        self.hsv_roi =  cv2.cvtColor(self.roi, cv2.COLOR_BGR2HSV)
        self.mask = cv2.inRange(
            self.hsv_roi, np.array((0., 60.,32.)),
            np.array((180.,255.,255.))
        )
        self.roi_hist = cv2.calcHist(
            [self.hsv_roi],[0],
            self.mask,[180],[0,180]
        )
        cv2.normalize(self.roi_hist,self.roi_hist,0,255,cv2.NORM_MINMAX)

        return self.roi_hist,self.track_window

    def c_init(self):
        return -1,-1,0,0