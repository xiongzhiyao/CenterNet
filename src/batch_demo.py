from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import json

from opts import opts
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

def load_imgs(order_id, image_id):
  image_folder_path = '/home/ec2-user/data/pose/hover-pose/i2/'
  img_path = 'orders/order_{}/image_{}_order_{}.jpg'.format(order_id,image_id,order_id)
  return image_folder_path + img_path

def load_image_path(annot_path):
  image_names = []
  with open(annot_path) as f:
      annotation = json.load(f)
  orders = annotation['orders']
  for data in orders:
      imgs = data['cameras']
      order_id = data['order_id']
      for img in imgs:
        image_id = img['image_id']
        img_path = load_imgs(order_id,image_id)
        image_names.append(img_path)
  return image_names

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)

  ospath = os.path.abspath(opt.demo)

  if ospath.endswith('json'):
    image_names = load_image_path(ospath)
  else:
    image_names = [opt.demo]
  
  for (image_name) in image_names:
    ret = detector.run(image_name)
    time_str = ''
    for stat in time_stats:
      time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
    print(time_str)

if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
