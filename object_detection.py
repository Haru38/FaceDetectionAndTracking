# coding: utf-8
# Object Detection Demo
import argparse
import cv2
import numpy as np
import os
import sys
import time
import tensorflow as tf

from detectFaceFront import *
from detectSmile import *
from fireBase import *
from meanShift import *

from distutils.version import StrictVersion

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

# Path to label and frozen detection graph. This is the actual model that is used for the object detection.
parser = argparse.ArgumentParser(description='object detection tester, using webcam or movie file')
parser.add_argument('-l', '--labels', default='./exported_graphs/labels.txt', help="default: './exported_graphs/labels.txt'")
parser.add_argument('-m', '--model', default='./exported_graphs/frozen_inference_graph.pb', help="default: './exported_graphs/frozen_inference_graph.pb'")
parser.add_argument('-d', '--device', default='normal_cam', help="normal_cam, jetson_nano_raspi_cam, jetson_nano_web_cam, raspi_cam, or video_file. default: 'normal_cam'") # normal_cam / jetson_nano_raspi_cam / jetson_nano_web_cam
parser.add_argument('-i', '--input_video_file', default='', help="Input video file")

args = parser.parse_args()

detection_graph = tf.Graph()

mode = 'bbox'

colors = [
  (0, 0, 255),
  (0, 64, 255),
  (0, 128, 255),
  (0, 192, 255),
  (0, 255, 255),
  (0, 255, 192),
  (0, 255, 128),
  (0, 255, 64),
  (0, 255, 0),
  (64, 255, 0),
  (128, 255, 0),
  (192, 255, 0),
  (255, 255, 0),
  (255, 192, 0),
  (255, 128, 0),
  (255, 64, 0),
  (255, 0, 0),
  (255, 0, 64),
  (255, 0, 128),
  (255, 0, 192),
  (255, 0, 255),
  (192, 0, 255),
  (128, 0, 255),
  (64, 0, 255),
]


def load_graph():
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(args.model, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
    return detection_graph

detection_graph = load_graph()



tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
with detection_graph.as_default():
  tf_sess = tf.Session(config = tf_config)
  ops = tf.get_default_graph().get_operations()
  all_tensor_names = {output.name for op in ops for output in op.outputs}
  tensor_dict = {}
  for key in [
      'num_detections', 'detection_boxes', 'detection_scores',
      'detection_classes', 'detection_masks'
  ]:
    tensor_name = key + ':0'
    if tensor_name in all_tensor_names:
      tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
          tensor_name)

  image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')


tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
with detection_graph.as_default():
  tf_sess = tf.Session(config = tf_config)
  ops = tf.get_default_graph().get_operations()
  all_tensor_names = {output.name for op in ops for output in op.outputs}
  tensor_dict = {}
  for key in [
      'num_detections', 'detection_boxes', 'detection_scores',
      'detection_classes', 'detection_masks'
  ]:
    tensor_name = key + ':0'
    if tensor_name in all_tensor_names:
      tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
          tensor_name)

  image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

def run_inference_for_single_image(image, graph):
  # Run inference
  output_dict = tf_sess.run(tensor_dict,
                          feed_dict={image_tensor: image})

  # all outputs are float32 numpy arrays, so convert types as appropriate
  output_dict['num_detections'] = int(output_dict['num_detections'][0])
  output_dict['detection_classes'] = output_dict[
      'detection_classes'][0].astype(np.int64)
  output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
  output_dict['detection_scores'] = output_dict['detection_scores'][0]
  return output_dict

cam = cv2.VideoCapture(0)
ret,frame = cam.read()
height, width, channels = frame.shape
count_max = 0

if __name__ == '__main__':
  count = 0
  df = detectFaceFront()#face detect instance
  ms = meanShift()#meanshift instance
  sx,sy,sw,sh = ms.c_init()#init coordinate
  target_flag = 0#detecting target (1:true,0:false)
  interval_time = 0

  labels = ['blank']
  with open(args.labels,'r') as f:
    for line in f:
      labels.append(line.rstrip())

  while True:
    ret, img = cam.read()
    if not ret:
      print('error')
      break

    key = cv2.waitKey(1)
    if key == 27: # when ESC key is pressed break
      break

    if (sx == 0 or sx+sw == width or sy == 0 or
      sy+sh == height) or target_flag == 0 or interval_time == 30:
        if interval_time == 30:
          interval_time = 0
        face_list,gray = df.solve(img)
        if len(face_list) > 0:
          target_flag = 1
          ftf = df.maxFace(face_list)#max face's coordinate
          roi_hist,track_window = ms.preProcessing(ftf,img)
          frame = df.showOne(ftf)
        else:
          target_flag = 0
          sx,sy,sw,sh = ms.c_init()
    else:
      #tracking face by meanShift
      sx,sy,sw,sh = ms.my_meanShift(img,roi_hist,track_window)
      img = cv2.rectangle(img, (sx,sy-100), (sx+sw,sy+sh), 255,2)

    interval_time += 1

    count += 1
    if count > count_max:
      img_bgr = cv2.resize(img, (300, 300))

      # convert bgr to rgb
      image_np = img_bgr[:,:,::-1]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      start = time.time()
      output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
      elapsed_time = time.time() - start

      for i in range(output_dict['num_detections']):
        class_id = output_dict['detection_classes'][i]
        if class_id < len(labels):
          label = labels[class_id]
        else:
          label = 'unknown'

        detection_score = output_dict['detection_scores'][i]

        if detection_score > 0.5:
            # Define bounding box
            h, w, c = img.shape
            box = output_dict['detection_boxes'][i] * np.array( \
              [h, w,  h, w])
            box = box.astype(np.int)

            speed_info = '%s: %.3f' % ('fps', 1.0/elapsed_time)
            cv2.putText(img, speed_info, (10,50), \
              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

            class_id = class_id % len(colors)
            color = colors[class_id]

            if output_dict['detection_scores'][i] * 100.0 > 97:#ての識別率
              # Draw bounding box
              cv2.rectangle(img, \
                (box[1], box[0]), (box[3], box[2]), color, 3)

              # Put label near bounding box
              information = '%s: %.1f%%' % (label, output_dict['detection_scores'][i] * 100.0)
              cv2.putText(img, information, (box[1] + 15, box[2] - 15), \
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)

      cv2.imshow('detection result', img)
      count = 0
    if args.device == 'raspi_cam':
      stream.seek(0)
      stream.truncate()

  tf_sess.close()
  cam.release()
  cv2.destroyAllWindows()
