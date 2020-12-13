import cv2

"""
正面の顔を検出するクラス
"""
class detectFaceFront():
    def __init__(self):
        self.cascade_file = "cascade_file/haarcascade_frontalface_alt.xml"
        self.cascade = cv2.CascadeClassifier(self.cascade_file)

    def solve(self,img):
        self.img = img
        self.img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        self.face_list = self.cascade.detectMultiScale(
            self.img_gray,
            scaleFactor = 1.11, 
            minNeighbors = 3, 
            minSize = (50, 50)
        )
        return self.face_list,self.img_gray

    #指定された顔だけ枠を囲む
    def showOne(self,mf):
        cv2.rectangle(self.img,(mf[0],mf[1]),(mf[0]+mf[2],mf[1]+mf[3]),255,5)
        return self.img

    #リストの中から一番近い顔を検出
    def maxFace(self,face_list):
        maxArea = 0
        maxFace = (0,0,0,0)
        for (x,y,w,h) in face_list:
            if maxArea < w*h:
                maxArea = w*h
                maxFace = (x,y,w,h)
        return maxFace
