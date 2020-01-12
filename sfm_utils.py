from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import re
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import traceback
import struct
import json
from tensorflow.compat.v1 import ConfigProto
from scipy.spatial.transform import Rotation
import cv2
from transformations import *

def findCameraSfm(dataset):
  path = "datasets/" + dataset + "/MeshroomCache/StructureFromMotion/"
  if not os.path.exists(path): return ""
  dr = os.listdir(path)
  if len(dr) == 0: return ""
  return path + dr[0] + "/cameras.sfm"

def findExrs(dataset):
  path = "datasets/" + dataset + "/MeshroomCache/PrepareDenseScene/"
  dr = os.listdir(path)
  if len(dr) == 0: return ""
  return path + dr[0]

# https://docs.python.org/3/library/struct.html
def readImagesBinary(path):
  images = {}
  f = open(path, "rb")
  num_reg_images = struct.unpack('Q', f.read(8))[0]
  for i in range(num_reg_images):
    image_id = struct.unpack('I', f.read(4))[0]
    qv = np.fromfile(f, np.double, 4)

    tv = np.fromfile(f, np.double, 3)
    camera_id = struct.unpack('I', f.read(4))[0]

    name = ""
    name_char = -1
    while name_char != b'\x00':
      name_char = f.read(1)
      if name_char != b'\x00':
        name += name_char.decode("ascii")


    num_points2D = struct.unpack('Q', f.read(8))[0]

    for i in range(num_points2D):
      f.read(8 * 2) # for x and y
      f.read(8) # for point3d Iid

    r = Rotation.from_quat([qv[1], qv[2], qv[3], qv[0]]).as_dcm().astype(np.float32)
    t = tv.astype(np.float32)

    R = np.transpose(r)
    center = -R * t
    # storage is scalar first, from_quat takes scalar last.
    images[image_id] = {
      "camera_id": camera_id,
      "r": r,
      "t": t,
      "R": R,
      "center": center,
      "filename": name
    }

  f.close()
  return images

def readCamerasBinary(path, new_height, new_width):
  cams = {}
  f = open(path, "rb")
  num_cameras = struct.unpack('Q', f.read(8))[0]
  for i in range(num_cameras):
    camera_id = struct.unpack('I', f.read(4))[0]
    model_id = struct.unpack('i', f.read(4))[0]

    width = struct.unpack('Q', f.read(8))[0]
    height = struct.unpack('Q', f.read(8))[0]

    fx = struct.unpack('d', f.read(8))[0]
    fy = struct.unpack('d', f.read(8))[0]
    cx = struct.unpack('d', f.read(8))[0]
    cy = struct.unpack('d', f.read(8))[0]

    cams[camera_id] = {
      "width": new_width,
      "height": new_height,
      "fx": fx * new_width/width,
      "fy": fy * new_height/height,
      "cx": cx * new_width/width,
      "cy": cy * new_height/height
    }
    # fx, fy, cx, cy
  f.close()
  return cams

def readCameraColmap(path, new_height, new_width):
  images = readImagesBinary(path + "/images.bin")
  cams = readCamerasBinary(path + "/cameras.bin", new_height, new_width)
  return images, cams

def readCameraDeepview(path, new_height, new_width):
    camera = {}
    #pp = [1, 5, 2, 6, 11, 15, 10, 14, 9, 4, 8, 3, 7, 12, 16, 13]
    c=0
    with open(path +'models.json', "r") as fi:
      js = json.load(fi)
      for i,cam in enumerate(js):
          for j, cam_info in enumerate(cam):
              sfm = {}
              sfm['fx'] = cam_info['focal_length'] * new_width/cam_info['width']
              sfm['fy'] = cam_info['focal_length'] * new_height/cam_info['height'] * cam_info['pixel_aspect_ratio']
              sfm['cx'] = cam_info['principal_point'][0] * new_width/cam_info['width']
              sfm['cy'] = cam_info['principal_point'][1] * new_height/cam_info['height']
              '''
              sfm['center'] = np.array([cam_info['position']]).reshape(3, 1)
              rotation,_= cv2.Rodrigues(np.float32(cam_info['orientation']))
              sfm['r'] = rotation.T
              sfm['t'] = - np.matmul(sfm['r'], sfm['center'])
              print(sfm['r'])
              print(sfm['t'])
              '''
              transform = np.identity(4)
              transform[0:3, 3] = (cam_info['position'][0], cam_info['position'][1], cam_info['position'][2])
              angle_axis = np.array([cam_info['orientation'][0], cam_info['orientation'][1], cam_info['orientation'][2]])
              angle = np.linalg.norm(angle_axis)
              axis = angle_axis / angle
              rot_mat = quaternion_matrix(quaternion_about_axis(-angle, axis))
              transform[0:3, 0:3] = rot_mat[0:3, 0:3]
              transform = np.linalg.inv(transform)
              sfm['r'] = transform[0:3,0:3]
              sfm['t'] = transform[0:3,3:4]

              s = '-'
              #camera[s.join(['time%02d' % (i), cam_info['relative_path'].split('/')[0]])] = sfm
              camera['time%02d-cam_%02d' % (i,j)] = sfm
              #print(i,j,cam_info['relative_path'].split('/')[0])
              #c =c+1
              #if c==3:    exit()
              

    return camera

class SfMData:
  def __init__(self, dataset, ref_img, ref_txt, scale, dmin, dmax):
    if not self.readMeshroom(dataset, ref_img, ref_txt):
        self.readDeepview(dataset, ref_img)
        #self.readColmap(dataset, ref_img)

    self.dmin = dmin
    self.dmax = dmax

    if self.dmin < 0 or self.dmax < 0:
      #with open("/home/heisenberg/dataset/" + dataset + "/planes.txt", "r") as fi:
      with open("/home2/suttisak/datasets/spaces_dataset/data/resize_800/" + dataset + "/planes.txt", "r") as fi:
        self.dmin, self.dmax = [float(x) for x in fi.readline().split(" ")]

    #self.w = int(self.ow * scale)
    #self.h = int(self.oh * scale)

    self.scaleAll(scale)
    '''
    xs = self.w / self.ow
    ys = self.h / self.oh

    self.fx = self.ofx * xs
    self.fy = self.ofy * ys
    self.px = self.opx * xs
    self.py = self.opy * ys
    '''

    #self.k = tf.constant([[self.fx, 0, self.px], [0, self.fy, self.py], [0, 0, 1]])

    #self.ki = tf.linalg.inv(self.k)

  def scaleAll(self, scale):
    self.scale = scale
    self.w = int(self.ow * scale)
    self.h = int(self.oh * scale)
    self.sw = 32*2#int(self.w/4)
    self.sh = 32*2#int(self.w/4)

    self.ref_k = np.array([[self.ref_fx * scale, 0, self.ref_px * scale],
    [0, self.ref_fy * scale, self.ref_py * scale],
    [0, 0, 1]])

  def readDeepview(self, dataset, ref_img):
      #path1  = '/home/heisenberg/dataset/' + dataset +'/image/'
      path1  = "/home2/suttisak/datasets/spaces_dataset/data/resize_800/" + dataset + "/img/"
      resize_img = plt.imread(path1 + os.listdir(path1)[0])
      self.oh = resize_img.shape[0]
      self.ow = resize_img.shape[1]
      #path2  = '/home/heisenberg/dataset/' + dataset +'/'
      path2 = "/home2/suttisak/datasets/spaces_dataset/data/800/" + dataset + "/"
      camera = readCameraDeepview(path2, self.oh, self.ow)
      found = 0
      for name, cam_info in camera.items():
          if ref_img in name:
              self.ref_img = plt.imread(path1 + name+".png")
              self.ref_r = cam_info['r'].astype(np.float32)
              self.ref_t = cam_info['t'].astype(np.float32)
              self.ref_fx = cam_info['fx']
              self.ref_fy = cam_info['fy']
              self.ref_px = cam_info['cx']
              self.ref_py = cam_info['cy']
              features = {"r":[self.ref_r],"t":[self.ref_t],"pxFocalLength":[self.ref_fx],"pyFocalLength":[self.ref_fy]}
              features.update({"principalPoint0":[self.ref_px],"principalPoint1":[self.ref_py]}) 
              self.features = features
              found = 1
              break
      if not found:
        raise Exception("ref_r, ref_t not found")

  def readColmap(self, dataset, ref_img):
    path  = '/home/heisenberg/dataset/' + dataset + '/dense/resize/'
    resize_img = plt.imread(path + os.listdir(path)[0])
    self.oh = resize_img.shape[0]
    self.ow = resize_img.shape[1]
    images, cams = readCameraColmap("/home/heisenberg/dataset/" + dataset + "/dense/sparse", self.oh, self.ow)

    found = 0
    for image_id, image in images.items():
        for cam_id, cam in cams.items():
          if cam_id == image_id and ref_img in image["filename"]:
            self.ref_r = image["r"].astype(np.float32)
            self.ref_t = image["t"].astype(np.float32).reshape(3, 1)
            self.ref_fx = cam['fx']

            self.ref_fy = cam['fy']

            self.ref_px = cam['cx']

            self.ref_py = cam['cy']

            found = 1
            break

    if not found:
      raise Exception("ref_r, ref_t not found")

  def readMeshroom(self, dataset, ref_img, ref_txt):
    path = findCameraSfm(dataset)

    if path == "": return False
    with open(path, "r") as fi:
      js = json.load(fi)

    self.ofx = float(js["intrinsics"][0]["pxFocalLength"])
    self.ofy = float(js["intrinsics"][0]["pxFocalLength"])

    self.opx = float(js["intrinsics"][0]["principalPoint"][0])
    self.opy = float(js["intrinsics"][0]["principalPoint"][1])
    self.ow = int(js["intrinsics"][0]["width"])
    self.oh = int(js["intrinsics"][0]["height"])
 
    st = 0
    if ref_txt != "":
      fi = open("datasets/" + dataset + "/" + ref_txt, "r")
      self.ref_r = np.transpose(np.reshape(np.matrix([float(x) for x in fi.readline().split(" ")], dtype='f'), [3, 3]))
      self.ref_t = -self.ref_r * np.reshape(np.matrix([float(x) for x in fi.readline().split(" ")], dtype='f'), [3, 1])

      fi.close()
    else:
      for view in js["views"]:
        if ref_img in view["path"]:
          for pose in js["poses"]:
            if pose["poseId"] == view["poseId"]:
              self.ref_r = np.transpose(np.reshape(np.matrix(pose["pose"]["transform"]["rotation"], dtype='f'), [3, 3]))
              self.ref_t = -self.ref_r * np.reshape(np.matrix(pose["pose"]["transform"]["center"], dtype='f'), [3, 1])

              st = 1
              break
          break
      if st == 0:
        raise Exception("error: ref img not found!")

    return True