## no sub render equation
## gradient model
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib import cm
from tensorflow.compat.v1 import ConfigProto
from mpl_toolkits.mplot3d import Axes3D
import cv2
from sfm_utils import SfMData

from tensorflow.keras.models import Model
#from tensorflow.keras.applications.vgg16 import VGG16
#from tensorflow.keras.applications.vgg19 import VGG19

#from view_gen import generateWebGL, generateConfigGL
from utils import *
#from localpath import getLocalPath


slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean("invz", False, "using inverse depth, ignore dmax in this case")
tf.app.flags.DEFINE_boolean("predict", False, "making a video")
tf.app.flags.DEFINE_boolean("restart", False, "making a last video frame")
tf.app.flags.DEFINE_float("scale", 0.75, "scale input image by")
tf.app.flags.DEFINE_integer("offset", 16, "offset size to mpi")
tf.app.flags.DEFINE_integer("subscale", 8, "downscale factor for the sub layer")

tf.app.flags.DEFINE_integer("layers", 25, "number of planes")
tf.app.flags.DEFINE_integer("sublayers", 2, "number of sub planes")
tf.app.flags.DEFINE_integer("epoch", 2000, "Training steps")
tf.app.flags.DEFINE_integer("batch_size", 1, "Size of mini-batch.")

tf.app.flags.DEFINE_integer("index", 0, "index number")

tf.app.flags.DEFINE_string("dataset", "temple0", "which dataset in the datasets folder")
tf.app.flags.DEFINE_string("input", "cen0", "input tfrecord")
tf.app.flags.DEFINE_string("ref_img", "01-cam_06", "reference image")

#tf.app.flags.DEFINE_string("ref_img", "0051.png", "reference image such that MPI is perfectly parallel to")

sub_sam = max(FLAGS.sublayers,1)
num_mpi = FLAGS.layers
offset = FLAGS.offset
dmin, dmax = 1,20
num_ = 6
max_step = 2


def load_data(dataset,input,h_w,batch=1,is_shuff=False):
  def parser(serialized_example):
    fs = tf.parse_single_example(
        serialized_example,
        features={
          "img": tf.FixedLenFeature([], tf.string),
          "r": tf.FixedLenFeature([9], tf.float32),
          "t": tf.FixedLenFeature([3], tf.float32),
          "h": tf.FixedLenFeature([], tf.int64),
          "w": tf.FixedLenFeature([], tf.int64),
          'pxFocalLength':tf.FixedLenFeature([], tf.float32),
          'pyFocalLength':tf.FixedLenFeature([], tf.float32),
          'principalPoint0':tf.FixedLenFeature([], tf.float32),
          'principalPoint1':tf.FixedLenFeature([], tf.float32)
        })

    fs["img"] = tf.to_float(tf.image.decode_png(fs["img"], 3)) / 255.0
    if True:#FLAGS.scale < 1:
      fs["img"] = tf.image.resize(fs["img"], h_w, tf.image.ResizeMethod.AREA)

    fs["r"] = tf.reshape(fs["r"], [3, 3])
    fs["t"] = tf.reshape(fs["t"], [3, 1])
    return fs

  #localpp = "datasets/" + dataset + "/" + input + ".train"
  localpp = "/home2/suttisak/datasets/spaces_dataset/data/resize_800/" + dataset + "/" + input + ".train"
  dataset = tf.data.TFRecordDataset([localpp])
  dataset = dataset.map(parser)
  if(is_shuff):  dataset = dataset.shuffle(5)
  dataset = dataset.repeat().batch(batch)

  return dataset.make_one_shot_iterator().get_next()

def computeHomography(sfm, features, ref_feat, d, ks=1,i=0):
  r = features["r"][i]
  t = features["t"][i]
  fx = features['pxFocalLength'][i]
  fy = features['pyFocalLength'][i]
  px = features['principalPoint0'][i]
  py = features['principalPoint1'][i]

  ref_r = ref_feat["r"][0]
  ref_t = ref_feat["t"][0]
  ref_fx = ref_feat['pxFocalLength'][0]
  ref_fy = ref_feat['pyFocalLength'][0]
  ref_px = ref_feat['principalPoint0'][0]
  ref_py = ref_feat['principalPoint1'][0]

  # equivalent to right multiplying [r; t] by the inverse of [ref_r, r ef_t]
  new_r = tf.matmul(r, tf.transpose(ref_r))
  new_t = tf.matmul(tf.matmul(-r, tf.transpose(ref_r)), ref_t) + t

  n = tf.constant([[0.0, 0.0, 1.0]])
  Ha = tf.transpose(new_r)

  Hb = tf.matmul(tf.matmul(tf.matmul(Ha, new_t), n), Ha)
  Hc = tf.matmul(tf.matmul(n, Ha), new_t)[0]

  k = tf.convert_to_tensor([[fx* sfm.scale, 0, px* sfm.scale], [0, fy* sfm.scale, py* sfm.scale], [0, 0, 1]])
  #ref_k = tf.constant([[ref_fx* sfm.scale, 0, ref_px* sfm.scale], [0, ref_fy* sfm.scale, ref_py* sfm.scale], [0, 0, 1]])
  ref_k = tf.convert_to_tensor([[ref_fx* sfm.scale, 0, ref_px* sfm.scale], [0, ref_fy* sfm.scale, ref_py* sfm.scale], [0, 0, 1]])
  #ref_k = tf.constant(sfm.ref_k)

  ki = tf.linalg.inv(k)

  return tf.matmul(tf.matmul(ref_k, Ha + Hb / (-d-Hc)), ki)

def getPlanes(sfm):
    dmax = sfm.dmax
    dmin = sfm.dmin
    if False:
      return (dmax-dmin)*np.linspace(0, 1, num_mpi)**4+dmin
    elif FLAGS.invz:
      return 1/np.linspace(1, dmin/dmax, num_mpi) * dmin
    else:
      return np.linspace(dmin, dmax, num_mpi)

def HomoWarp(sfm, features, ref_feat,i=0, center=0):
  cxys = []
  planes = getPlanes(sfm)
  x, y = tf.meshgrid(list(range(sfm.w)), list(range(sfm.h)))
  x, y = tf.cast(x, tf.float32), tf.cast(y, tf.float32)
  coords = tf.stack([x, y, tf.ones_like(x)], 2)

  for v in planes:
    H = computeHomography(sfm, features, ref_feat, v,i=i)
    newCoords = tf.matmul(coords, H, transpose_b=True)
    cxy = tf.expand_dims(newCoords[:,:,:2]/newCoords[:,:,2:3],0)+ offset
    cxys.append(cxy)

  warp = tf.concat(cxys,0)
  return warp

def InvzHomoWarp(sfm, features, ref_feat,i=0, center=0):
  cxys = []
  planes = getPlanes(sfm)

  x, y = tf.meshgrid(list(range(sfm.nw)), list(range(sfm.nh)))
  x, y = tf.cast(x, tf.float32), tf.cast(y, tf.float32)
  coords = tf.stack([x, y, tf.ones_like(x)], 2)

  for v in planes:
    H = computeHomography(sfm, features, ref_feat, v, i=i)
    newCoords = tf.matmul(coords, tf.linalg.inv(H), transpose_b=True) #not good way!
    cxy = tf.expand_dims(newCoords[:,:,:2]/newCoords[:,:,2:3],0)- offset
    cxy -= center
    cxys.append(cxy)

  iwarp = tf.concat(cxys,0)

  return iwarp

def network(sfm, features, ref_feat, mpi, center=0):
  #No sublayer
  warp = HomoWarp(sfm, features, ref_feat, center=center)
  samples = tf.contrib.resampler.resampler(mpi[:,:,:,:4],warp)
  weight = tf.math.cumprod(1-samples[:,:,:,3:4],0,exclusive=True)

  output = tf.reduce_sum(weight * samples[:,:,:,:3]* samples[:,:,:,3:4],0,keepdims=True)

  return output

def mask_maker(sfm,features,ref_feat):
  cxys = []
  planes = getPlanes(sfm)
  x, y = tf.meshgrid(list(range(sfm.w)), list(range(sfm.h)))
  x, y = tf.cast(x, tf.float32), tf.cast(y, tf.float32)
  coords = tf.stack([x, y, tf.ones_like(x)], 2)

  for v in [planes[0],planes[-1]]:
    H = computeHomography(sfm, features, ref_feat, v)
    newCoords = tf.matmul(coords, H, transpose_b=True)
    cxy = tf.expand_dims(newCoords[:,:,:2]/newCoords[:,:,2:3],0)+ offset 
    cxys.append(cxy)
  warp = tf.concat(cxys,0)
    
  samples = tf.contrib.resampler.resampler(tf.ones([2, sfm.nh, sfm.nw, 1]),warp)
  mask = samples[0:1]*samples[1:2]

  return mask*0.99 + (1-mask)*0.01

def train_MPI(sfm):
    iter = tf.compat.v1.placeholder(tf.float32, shape=[], name='iter')
    lod_in = tf.compat.v1.placeholder(tf.float32, shape=[], name='lod_in')
    
    features0 = load_data(FLAGS.dataset,FLAGS.input,[sfm.h,sfm.w],1,is_shuff = False)

    real_img = features0['img']

    bh = sfm.h + 2 * offset
    bw = sfm.w + 2 * offset
    sfm.nw = sfm.w + 2 * offset
    sfm.nh = sfm.h + 2 * offset

    sfm.sw = sfm.w
    sfm.sh = sfm.h

    center = np.array([[[[0,0]]]])

    mask = mask_maker(sfm,features0,sfm.features,center)

    int_mpi1 = np.random.uniform(-1, 1,[num_mpi, bh, bw, 3]).astype(np.float32)
    int_mpi2 = np.random.uniform(-5,-3,[num_mpi*sub_sam, bh, bw, 1]).astype(np.float32)
    int_mpi3 = np.zeros([num_mpi*sub_sam, bh, bw, 4]).astype(np.float32)

    tt = True
    with tf.compat.v1.variable_scope("Net%d"%(FLAGS.index)):
      mpic = tf.compat.v1.get_variable("mpi_c", initializer=int_mpi1, trainable=tt)   
      mpia = tf.compat.v1.get_variable("mpi_a", initializer=int_mpi2, trainable=tt)
      mpia = tf.concat([mpia[:-1],tf.ones_like(mpia[0:1])*5],0)
      new_mpi = tf.concat([tf.tile(mpic,[sub_sam,1,1,1]),mpia],-1)

    lr = tf.compat.v1.train.exponential_decay(0.1,iter,1000,0.2)
    optimizer = tf.compat.v1.train.AdamOptimizer(lr)
    fac = 0.1#(1 - iter/(1500*2))# *0.01
    tva = tf.constant(0.1) * fac #*0.01
    tvc = tf.constant(0.005)  * fac #*2.0

    mpi_sig = tf.sigmoid(new_mpi)
    img_out = network( sfm, features0, sfm.features, mpi_sig, center)

    #img_tile = tf.tile(tf.concat([real_img[0:1],img_out],-1),[num_mpi*sub_sam,1,1,1])
    #img_tile = tf.contrib.resampler.resampler(img_tile,InvzHomoWarp(sfm,features0,sfm.features))
    #step_img = tf.reshape(img_tile,(1,num_mpi*sub_sam*bh,bw,6))

    loss = 0
    loss +=  100000 * tf.reduce_mean(tf.square(img_out[0] - real_img[-1])*mask)
    loss += tvc * tf.reduce_mean(tf.image.total_variation(mpi_sig[:, :, :, :3]))
    loss += tva * tf.reduce_mean(tf.image.total_variation (mpi_sig[:, :, :, 3:4]))
    train_op = slim.learning.create_train_op(loss,optimizer)



    image_out = tf.clip_by_value(img_out,0.0,1.0)
    a_long = tf.reshape(mpi_sig[:,:,:,3:4],(1,num_mpi*sub_sam*bh,bw,1))
    c_long = tf.reshape(mpi_sig[:,:,:,:3],(1,num_mpi*sub_sam*bh,bw,3))
    summary = tf.compat.v1.summary.merge([
                tf.compat.v1.summary.scalar("post0/all_loss", loss),
                #tf.compat.v1.summary.image("post1/out0",step_img[:,:,:,:3]),
                tf.compat.v1.summary.image("post0/out1",tf.concat([real_img[-1:],image_out],1)),
                tf.compat.v1.summary.image("post1/o_alpha",a_long),
                tf.compat.v1.summary.image("post1/o_color",c_long*a_long),
                ])

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    localpp = "TensorB/"+FLAGS.dataset+"_s%02d"%FLAGS.subscale
    if FLAGS.index==0:
      if os.path.exists(localpp):
        os.system("rm -rf " +localpp )
    if not os.path.exists(localpp):
      os.makedirs(localpp)
    writer = tf.compat.v1.summary.FileWriter(localpp)
    #writer.add_graph(sess.graph)

    
    sess = tf.compat.v1.Session(config=config)
    
    if not FLAGS.restart:
      sess.run(tf.compat.v1.global_variables_initializer())
      t_vars = slim.get_variables_to_restore()
      localpp = '/home2/suttisak/model/SingleImgMPI/' + FLAGS.dataset +"/"
      var2restore = [var for var in t_vars if 'Net0' in var.name ]
      print(var2restore)
      saver = tf.train.Saver(var2restore)
      ckpt = tf.train.latest_checkpoint(localpp )
      saver.restore(sess, ckpt)
    else:
      sess.run(tf.compat.v1.global_variables_initializer())


    localpp = './model/' + FLAGS.dataset +"/s%02d"%FLAGS.subscale
    if not os.path.exists(localpp):
        os.makedirs(localpp)
    saver = tf.train.Saver()


    los = 0
    for i in range(FLAGS.epoch + 3):
        feedlis = {iter:i}
        _,los = sess.run([train_op,loss],feed_dict=feedlis)

        if i%50==0:
            print(i, "loss = ",los )
        if i%20 == 0:
            summ = sess.run(summary,feed_dict=feedlis)
            writer.add_summary(summ,i)
        if i%200==199:
            saver.save(sess, localpp + '/' + str(000))
    saver.save(sess, localpp + '/mpi')

def predict(sfm):
    def parser(serialized_example):
      fs = tf.parse_single_example(
          serialized_example,
          features={
            "r": tf.FixedLenFeature([9], tf.float32),
            "t": tf.FixedLenFeature([3], tf.float32),
            'pxFocalLength':tf.FixedLenFeature([], tf.float32),
            'pyFocalLength':tf.FixedLenFeature([], tf.float32),
            'principalPoint0':tf.FixedLenFeature([], tf.float32),
            'principalPoint1':tf.FixedLenFeature([], tf.float32)
          })
      fs["r"] = tf.reshape(fs["r"], [3, 3])
      fs["t"] = tf.reshape(fs["t"], [3, 1])
      return fs

    lod_in = tf.compat.v1.placeholder(tf.float32, shape=[], name='lod_in')


    #testset = tf.data.TFRecordDataset(["datasets/" + FLAGS.dataset + "/" + FLAGS.input +".test"])
    localpp = "/home2/suttisak/datasets/spaces_dataset/data/resize_800/" + FLAGS.dataset + "/" + FLAGS.input + ".test"
    testset = tf.data.TFRecordDataset([localpp])
    testset = testset.map(parser).repeat().batch(1).make_one_shot_iterator()
    features = testset.get_next()

    rot = features["r"][0]
    tra = features["t"][0]

    nh = sfm.h + offset * 2
    nw = sfm.w + offset * 2
    sfm.nh = nh
    sfm.nw = nw

    int_mpi1 = np.random.uniform(-1, 1,[num_mpi, nh, nw, 3]).astype(np.float32)
    int_mpi2 = np.random.uniform(-5,-3,[num_mpi*sub_sam, nh, nw, 1]).astype(np.float32)
    int_mpi3 = np.zeros([num_mpi*sub_sam, nh, nw, 4]).astype(np.float32)


    with tf.compat.v1.variable_scope("Net%d"%(FLAGS.index)):
        mpic = tf.compat.v1.get_variable("mpi_c", initializer=int_mpi1, trainable=False)   
        mpia = tf.compat.v1.get_variable("mpi_a", initializer=int_mpi2, trainable=False)
        mpie = tf.compat.v1.get_variable("mpi_e", initializer=int_mpi3, trainable=False)
        tf.compat.v1.add_to_collection("mpie",mpie)

        mpi0 = tf.concat([tf.tile(mpic,[sub_sam,1,1,1]),mpia, mpie],-1)


    new_mpi = mpi0

    center = np.array([[[[0,0]]]])
    for i in range(max_step):
      features2 = load_data(FLAGS.dataset,FLAGS.input,[sfm.h,sfm.w],num_,is_shuff = True)
      new_mpi = update_Block(new_mpi,sfm,features2,sfm.features,step=i,center=center)


    name1 = tf.compat.v1.get_collection('mpie')
    gen_mpi = name1[0].assign(new_mpi[:,:,:,:4])
    img_out = network(sfm,features,sfm.features,tf.sigmoid(mpie))

    with tf.compat.v1.variable_scope("post%d"%(FLAGS.index)):
        image_out= tf.clip_by_value(img_out[0],0.0,1.0)

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())
    t_vars = slim.get_variables_to_restore()
    variables_to_restore = [var for var in t_vars if 'Net' not in var.name]
    #variables_to_restore = slim.get_variables_to_restore()
    print(variables_to_restore)
    saver = tf.train.Saver(variables_to_restore)
    localpp = './model/' + FLAGS.dataset +"/s%02d"%FLAGS.subscale
    ckpt = tf.train.latest_checkpoint(localpp )
    saver.restore(sess, ckpt)

    sess.run(gen_mpi)

    if True:  # make sample picture and video
        webpath = "webpath/"  #"/var/www/html/orbiter/"
        if not os.path.exists(webpath + FLAGS.dataset+"_s%02d"%FLAGS.subscale):
            os.system("mkdir " + webpath + FLAGS.dataset+"_s%02d"%FLAGS.subscale)

        for i in range(0,300,1):
          #feed = sess.run(features)
          #out = sess.run(image_out,feed_dict={lod_in:0,rot:feed["r"][0],tra:feed["t"][0]})
          out = sess.run(image_out,feed_dict={lod_in:0})
          if (i%50==0): 
            print(i)
            plt.imsave("webpath/"+FLAGS.dataset+"_s%02d"%FLAGS.subscale+"/%04d.png"%( i),out)
          plt.imsave("result/frame/"+FLAGS.dataset+"_%04d.png"%( i),out)

        cmd = 'ffmpeg -y -i ' + 'result/frame/'+FLAGS.dataset+'_%04d.png -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p webpath/'+FLAGS.dataset+"_s%02d"%FLAGS.subscale+'/moving.mp4'
        print(cmd)
        os.system(cmd)
    exit()


def main(argv):
    sfm = SfMData(FLAGS.dataset,
                FLAGS.ref_img,
                "",
                FLAGS.scale,
                dmin,
                dmax)
    if FLAGS.predict:
        predict(sfm)
    else:
        train_MPI(sfm)
    print("Jub Jub!!")

if __name__ == "__main__":
  sys.excepthook = colored_hook(
      os.path.dirname(os.path.realpath(__file__)))
  tf.compat.v1.app.run()
