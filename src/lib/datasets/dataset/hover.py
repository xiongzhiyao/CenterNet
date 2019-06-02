from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data

class HOVERDATA(object):
  def __init__(self,annot_path):
    with open(annot_path) as f:
      d = json.load(f)
    self.anno = d['orders']
    self.gables = {}

  def getImgIds(self):
    ids = []
    for data in self.anno:
      imgs = data['cameras']
      order_id = data['order_id']
      for img in imgs:
        image_id = img['image_id']
        key = (order_id,image_id)
        ids.append(key)
        self.gables[key] = img['gables']
    return ids

  def loadImgs(self,key):
    order_id, image_id = key
    img_path = 'orders/order_{}/image_{}_order_{}.jpg'.format(order_id,image_id,order_id)
    return img_path

  def loadAnns(self,img_id,shape):
    gables = self.gables[img_id]
    all_annotations = []

    h,w,_ = shape
    
    DIA = 0.2 # TODO make this opt

    for gable in gables:
      keypoints = np.array(gable)

      '''calculate the bounding box'''
      left, top = np.min(keypoints[:,0]), np.min(keypoints[:,1])
      right, bottom = np.max(keypoints[:,0]), np.max(keypoints[:,1])
      
      left *= (1. - DIA)
      top *= (1. - DIA)
      right *= (1. + DIA)
      bottom *= (1. + DIA)

      left = max(left,0)
      top = max(top,0)
      right = min(right,w)
      bottom = min(bottom,h)

      width, height = right-left, bottom-top

      '''rearrange expoints to coco format'''
      gable_np = np.array(keypoints)
      gable_with_visibility_flag = np.concatenate([gable_np, 2 * np.ones(shape=(gable_np.shape[0], 1))], axis=1)
      gable_with_visibility_flag = gable_with_visibility_flag.astype(int)
      padded_keypoints = np.zeros((17,3),dtype=int) # TODO change this 17 to argument
      padded_keypoints[0:14,:] = gable_with_visibility_flag

      dict_gable = {
        'category_id': 1,
        'image_id': img_id[0],
        'iscrowd': 0, # not sure it's meaning, though
        'num_keypoints': 14,
        'area': np.round(height*width, decimals=4), # we are not using coco dataset area, but let's keep it here
        'bbox': [left, top, width, height],
        'keypoints': padded_keypoints.flatten().tolist(),
      }

      all_annotations.append(dict_gable)

    return all_annotations


class HOVERHP(data.Dataset):
  num_classes = 1
  num_joints = 17
  default_resolution = [512, 512]
  mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
  flip_idx = [[0, 0],    # apex stays the same
              [1, 2],    # fascia endpoint swap
              [3, 5],    # post tops swap
              [4, 6],    # post bottoms swap
              # back pentagon
              [7, 7],    # apex stays the same
              [8, 9],    # fascia endpoint swap
              [10, 12],  # post tops swap
              [11, 13]]
  def __init__(self, opt, split):
    super(HOVERHP, self).__init__()
    self.edges = [[0,1], [0,2],
                  [3,4], [5,6],
                  [4,6],
                  [7,8], [7,9],
                  [10,11], [12,13],
                  [11,13],
                  [0,7],
                  [1,9], [2,8],
                  [4,13],[6,11]
                  ]

    self.acc_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    self.data_dir = os.path.join(opt.data_dir, 'coco')
    #self.img_dir = os.path.join(self.data_dir, 'images/{}2017/'.format(split))
    self.img_dir = '/home/ec2-user/data/pose/hover-pose/house-pose-estimation.data/'
    if split == 'test':
      self.annot_path = os.path.join(
          "/home/ec2-user/data/pose/hover-pose/house-pose-estimation.data/annotations/gable_all_in_image",
          "test.json"
          )
    elif split == 'train':
      self.annot_path = os.path.join(
          "/home/ec2-user/data/pose/hover-pose/house-pose-estimation.data/annotations/gable_all_in_image",
          "train.json"
          )
    elif split == 'val':
      self.annot_path = os.path.join(
          "/home/ec2-user/data/pose/hover-pose/house-pose-estimation.data/annotations/gable_all_in_image",
          "val.json"
          )
    self.max_objs = 32
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    self.split = split
    self.opt = opt

    print('==> initializing coco 2017 {} data.'.format(split))
    
    #self.coco = coco.COCO(self.annot_path)
    #image_ids = self.coco.getImgIds()

    self.hover = HOVERDATA(self.annot_path)
    image_ids = self.hover.getImgIds()

    train_limit = self.opt.train_data_limit #TODO: only for debugging, limit the number of training data
    counter = 0
    if split == 'train':
      self.images = []
      for img_id in image_ids:
        idxs = img_id
        if len(idxs) > 0: #only for understanding the capacity of the model
          counter += 1
          if train_limit and counter > train_limit:
            break
          self.images.append(img_id)
    else:
      self.images = image_ids
    self.num_samples = len(self.images)
    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    # import pdb; pdb.set_trace()
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = 1
        for dets in all_bboxes[image_id][cls_ind]:
          bbox = dets[:4]
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = dets[4]
          bbox_out  = list(map(self._to_float, bbox))
          keypoints = np.concatenate([
            np.array(dets[5:39], dtype=np.float32).reshape(-1, 2), 
            np.ones((17, 1), dtype=np.float32)], axis=1).reshape(51).tolist()
          keypoints  = list(map(self._to_float, keypoints))

          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(score)),
              "keypoints": keypoints
          }
          detections.append(detection)
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    json.dump(self.convert_eval_format(results), 
              open('{}/results.json'.format(save_dir), 'w'))


  def run_eval(self, results, save_dir):
    # result_json = os.path.join(opt.save_dir, "results.json")
    # detections  = convert_eval_format(all_boxes)
    # json.dump(detections, open(result_json, "w"))
    self.save_results(results, save_dir)
    coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
    coco_eval = COCOeval(self.coco, coco_dets, "keypoints")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()