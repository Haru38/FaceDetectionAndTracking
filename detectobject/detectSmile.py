import cv2

"""
笑顔検知をするクラス
"""
class detectSmile():
    def __init__(self):
        self.cascade_file = 'cascade_file/haarcascade_smile.xml'
        self.cascade = cv2.CascadeClassifier(self.cascade_file)

    def drawSmileFactor(self,gray,cn,frame):
        #Gray画像から，顔領域を切り出す
        self.roi_gray = gray[cn[1]:cn[1]+cn[3], cn[0]:cn[0]+cn[2]] 
        #笑顔識別
        self.smiles= self.cascade.detectMultiScale(self.roi_gray,scaleFactor= 1.1, minNeighbors=10, minSize=(20, 20))
        if len(self.smiles)>0:
            for(self.sx,self.sy,self.sw,self.sh) in self.smiles:
                cv2.rectangle(
                    frame,
                    (cn[0] + self.sx, cn[1] + self.sy),
                    (cn[0] + self.sx + self.sw, cn[1] + self.sy + self.sh),
                    (0, 0, 255),
                    2
                )
        return frame,len(self.smiles)

    def countSmileFactor(self,gray,cn):
        #Gray画像から，顔領域を切り出す
        self.roi_gray = gray[cn[1]:cn[1]+cn[3], cn[0]:cn[0]+cn[2]] 
        #笑顔識別
        self.smiles= self.cascade.detectMultiScale(
            self.roi_gray,scaleFactor= 1.1, 
            minNeighbors=10, 
            minSize=(20, 20)
        )
        if len(self.smiles)>=5:
            return 1
        else:
            return 0