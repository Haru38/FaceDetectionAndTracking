from detectobject import *
from firebase.fireBase import *
from filter.meanshift import *

import threading
from queue import Queue
import time

"""
データベースにデータを追加する関数
mainのなかで別スレッドとして起動する
"""

data_q = Queue()#make queue

def add_database():
    while True:
        time.sleep(0.5)
        if data_q.empty() == False:
            #data = [x,y,w,h,s]
            data = data_q.get()
            cp_x = data[0]+data[2]//2#centerpoint
            cp_y = data[1]+data[3]//2
            smile = data[4]
            print(cp_x,cp_y,smile)
            #add database
            coordinate_ref.push({
                'x': cp_x,
                'y': cp_y
            })
            # if smile == 1:
            #     smile_ref.push({
            #         'smile': 1
            #     })
            data_q.task_done()
        else:
            continue

def main():
    cap = cv2.VideoCapture(0)#open camera
    ret,frame = cap.read()
    height, width, channels = frame.shape
    info_ref.push({#push camera's information
            'width': width,
            'height': height
        })

    df = detectFaceFront()#face detect instance
    ds = detectSmile()#smile detect instance
    ms = meanShift()#meanshift instance
    
    x,y,w,h = ms.c_init()#init coordinate
    target_flag = 0#detecting target (1:true,0:false)
    interval_time = 0
    #make add_database_thread and start
    t = threading.Thread(target=add_database, daemon=True)
    t.start()

    while True:
        ret,frame = cap.read()
        #detect face
        if (x == 0 or x+w == width or y == 0 or
            y+h == height) or target_flag == 0 or interval_time == 30:
                if interval_time == 30:
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
            frame = cv2.rectangle(frame, (x,y-100), (x+w,y+h), 255,2)
        
        #frame = ds.drawSmileFactor(gray,[x,y,w,h],frame)
        #s = ds.countSmileFactor(gray,[x,y,w,h])
        s = 0
        
        cv2.imshow("tracking", frame)#show frame
        interval_time += 1

        data_q.put([x,y,w,h,s])#push data to queue

        key = cv2.waitKey(1)
        if key == 27:
            break

    # waiting thread end
    data_q.join()
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__" :
    main()