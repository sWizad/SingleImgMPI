### generate tfRecord for Scene project
### make for DeepView's Space Dataset
import numpy as np
import os
import tensorflow as tf
import cv2 as cv
import json
import matplotlib.pyplot as plt

import math
from sfm_utils import *

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("dataset", "penguin", "input_dataset")
tf.app.flags.DEFINE_string("output", "tem", "outcome_name")

tf.app.flags.DEFINE_boolean("skipResize", False, "skip_convert_exr_file")

tf.app.flags.DEFINE_integer("start", 0, "start")
tf.app.flags.DEFINE_integer("end", 15, "end")

tf.app.flags.DEFINE_integer("new_height", 480, "new_height")
tf.app.flags.DEFINE_integer("new_width", 800, "new_width")

tf.app.flags.DEFINE_string("ref_img", 'time01-cam_06', "reference focal length during test set")
#tf.app.flags.DEFINE_string("ref_img", "0040.png", "reference image such that MPI is perfectly parallel to")
tf.app.flags.DEFINE_integer("index", 1, "index of reference image such that MPI is perfectly parallel to")

_EPS = np.finfo(float).eps * 4.0
index = FLAGS.index

def quaternion_from_matrix(matrix):
    matrix = np.r_[np.c_[matrix, np.array([[0],[0],[0]])], np.array([[0,0,0,1]])]
    # Return quaternion from rotation matrix.
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    q = np.empty((4, ))
    t = np.trace(M)
    if t > M[3, 3]:
        q[0] = t
        q[3] = M[1, 0] - M[0, 1]
        q[2] = M[0, 2] - M[2, 0]
        q[1] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
        q = q[[3, 0, 1, 2]]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    if q[0] < 0.0:
        np.negative(q, q)
    return q

def quaternion_matrix(quaternion):
    # Return homogeneous rotation matrix from quaternion.
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])


def unit_vector(data, axis=None, out=None):
    # Return ndarray normalized by length, i.e. Euclidean norm, along axis.
    data = np.array(data, dtype=np.float64, copy=True)
    data /= math.sqrt(np.dot(data, data))
    return data

def quaternion_slerp(quat0, quat1, fraction, spin=0, shortestpath=True):
    # Return spherical linear interpolation between two quaternions.
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < _EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        np.negative(q1, q1)
    angle = math.acos(d) + spin * math.pi
    if abs(angle) < _EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0

def interpolate_rotation(m1, m2, t):
  q1 = quaternion_from_matrix(m1)
  q2 = quaternion_from_matrix(m2)
  return quaternion_matrix(quaternion_slerp(q1, q2, t))[:3, :3]

def generateDeepview():
  def f1(a):
    return tf.train.Feature(float_list=tf.train.FloatList(value=np.array(a).flatten())),
  def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
  def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  camera = readCameraDeepview("/home2/suttisak/datasets/spaces_dataset/data/800/" + FLAGS.dataset +'/', FLAGS.new_height, FLAGS.new_width)

  resize_folder = "/home2/suttisak/datasets/spaces_dataset/data/resize_800/" + FLAGS.dataset + "/"
  if not os.path.exists(resize_folder):
      os.system("mkdir " + resize_folder)
  imgFolder = "/home2/suttisak/datasets/spaces_dataset/data/800/" + FLAGS.dataset + '/'

  resize_folder += "img/"
  if not FLAGS.skipResize:
    if os.path.exists(resize_folder):
        os.system("rm -rf "+resize_folder)
    os.system("mkdir " + resize_folder)
    for folder in os.listdir(imgFolder):
        if 'cam' in folder:
            num = int(folder.split('_')[1])
            for file in os.listdir(imgFolder + folder):
                print('Resizing {} to {}x{}'.format(imgFolder + 'cam_%02d/' %num  + file, FLAGS.new_height, FLAGS.new_width))
                time = int(file.split('.')[0].split('_')[1])
                #os.system("convert " + imgFolder + 'cam_%02d/' %num + file + " -resize " + str(FLAGS.new_width) + "x" + str(FLAGS.new_height) + "! " + resize_folder +'time%02d' % (time) +'-' + 'cam_%02d' %(num)+'.png')
                img = cv.imread(imgFolder+'cam_%02d/' %num + file)
                img = cv.resize(img,(FLAGS.new_width, FLAGS.new_height), interpolation=cv.INTER_AREA)
                cv.imwrite(resize_folder +'time%02d' % (time) +'-' + 'cam_%02d' %(num)+'.png',img)
  
  with tf.python_io.TFRecordWriter("/home2/suttisak/datasets/spaces_dataset/data/resize_800/" + FLAGS.dataset + "/" + FLAGS.output + ".train") as tfrecord_writer:
    with tf.Graph().as_default():
      im0 = tf.compat.v1.placeholder(dtype=tf.uint8)
      encoded0 = tf.image.encode_png(im0)
      with tf.compat.v1.Session() as sess:
        #pp = [7,0,1,6,10,9,8,]
        #pp +=[6,1,2,5,11,10,7,]
        #pp =  [5,2,3,4,12,11,6]
        #pp += [11,6,5,12,13,14,     10,7,6,11,14,15,9]
        #pp += [6,0,3,4,13,15,8]
        pp = [0,3,9,12]
        #pp += [1,2,4,6,8,12,9,13,14,15]
        pp += [i for i in range(16)]

        for i0 in [2]+list(range(14)): #[2,9,0]:#range(6): #[0,4,5,10,11,13]:
          for p in pp:
            name = 'time%02d-cam_%02d' % (i0,p)
            cam = camera[name]
            if FLAGS.ref_img in name:
            #if '000352' in cam['filename']:
                ref_fx = cam['fx']
                ref_fy = cam['fy']
                ref_px = cam['cx']
                ref_py = cam['cy']
            print('GENERATE TFRecordDataset FOR {}'.format(name))
            image = tf.convert_to_tensor(resize_folder +  name+'.png', dtype = tf.string)
            image = tf.io.read_file(image)
            ret = sess.run(image)
            example = tf.train.Example(features=tf.train.Features(feature={
                'img': bytes_feature(ret),
                'name':bytes_feature(str.encode(name)),
                'r': f1(cam["r"]),
                't': f1(cam["t"]),
                'h': _int64_feature(FLAGS.new_height),
                'w': _int64_feature(FLAGS.new_width),
                'pxFocalLength':f1(cam['fx']),
                'pyFocalLength':f1(cam['fy']),
                'principalPoint0':f1(cam['cx']),
                'principalPoint1':f1(cam['cy'])}))
            tfrecord_writer.write(example.SerializeToString())

  with tf.python_io.TFRecordWriter("/home2/suttisak/datasets/spaces_dataset/data/resize_800/" + FLAGS.dataset + "/" + FLAGS.output + ".test") as tfrecord_writer:
    scams = sorted(camera.items(), key=lambda x: x[0])
    n = 20
    for i in range(len(scams)-1):
      for j in range(n):
        tt = (j+0.5) / n # avoid the train image, especially the ref image
        rot = interpolate_rotation(scams[i][1]["r"], scams[i+1][1]["r"], tt)
        t = scams[i][1]["t"] * (1-tt) + scams[i+1][1]["t"] * tt
        example = tf.train.Example(features=tf.train.Features(
          feature={
            'r': f1(rot),
            't': f1(t),
            'pxFocalLength':f1(ref_fx),
            'pyFocalLength':f1(ref_fy),
            'principalPoint0':f1(ref_px),
            'principalPoint1':f1(ref_py)
          }))
        tfrecord_writer.write(example.SerializeToString())

  if False:
    file1 = open("/home2/suttisak/datasets/spaces_dataset/data/resize_800/"+ FLAGS.dataset +"/planes.txt","w")
    file1.write("1.0 20.0\n")
    file1.close

if __name__ == "__main__":

    generateDeepview()
    print("success")
