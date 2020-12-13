from detectFaceFront import *
from detectSmile import *
from meanShift import *
from hand import *

cap = cv2.VideoCapture(0)#open camera
ret,frame = cap.read()
height, width, channels = frame.shape

#df = detectFaceFront()#face detect instance
#ds = detectSmile()#smile detect instance
df = Hand()
ms = meanShift()#meanshift

x,y,w,h = ms.c_init()#init coordinate
target_flag = 0#detecting target (1:true,0:false)
interval_time = 0

while True:
    ret,frame = cap.read()
    #detect face
    if (x == 0 or x+w == width or y == 0 or
        y+h == height) or target_flag == 0 or interval_time == 50:
            if interval_time == 50:
                interval_time = 0
            face_list,gray = df.solve(frame)
            if len(face_list) > 0:
                target_flag = 1
                tf = df.maxFace(face_list)#max face's coordinate
                roi_hist,track_window = ms.preProcessing(tf,frame)
                frame = df.showOne(tf)
            else:
                target_flag = 0
            x,y,w,h = ms.c_init()
    else:
        #tracking face by meanShift
        x,y,w,h = ms.my_meanShift(frame,roi_hist,track_window)
        #frame = cv2.rectangle(frame, (x,y-100), (x+w,y+h), 255,2)
        frame = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        
    # frame,s = ds.drawSmileFactor(gray,[x,y,w,h],frame)
    #s = ds.countSmileFactor(gray,[x,y,w,h])
    # if s == 1:
    #     cv2.putText(
    #         frame,
    #         "SMILE!!!!",
    #         (x+w//3, y-150),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         1.0,
    #         (255, 255, 255),
    #         thickness=2
    #     )
    cv2.imshow("hand", frame)#show frame
    interval_time += 1

    key = cv2.waitKey(33)
    if key == 27:
        break

cv2.destroyAllWindows()
cap.release()