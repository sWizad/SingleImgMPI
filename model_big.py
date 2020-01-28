## Make MPI by direct optimization
import os, sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
import cv2
from sfm_utils import SfMData

#from view_gen import generateWebGL, generateConfigGL
from utils import *
#from localpath import getLocalPath


slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean("invz", True, "using inverse depth, ignore dmax in this case")
tf.app.flags.DEFINE_boolean("predict", False, "making a video")
tf.app.flags.DEFINE_boolean("restart", False, "making a last video frame")
tf.app.flags.DEFINE_float("scale", 0.75, "scale input image by")
tf.app.flags.DEFINE_integer("offset", 16, "offset size to mpi")

tf.app.flags.DEFINE_integer("layers", 24, "number of planes")
tf.app.flags.DEFINE_integer("sublayers", 1, "number of sub planes")
tf.app.flags.DEFINE_integer("epoch", 1000, "Training steps")
tf.app.flags.DEFINE_integer("batch_size", 1, "Size of mini-batch.")

tf.app.flags.DEFINE_integer("index", 0, "index number")

tf.app.flags.DEFINE_string("dataset", "temple0", "which dataset in the datasets folder")
tf.app.flags.DEFINE_string("input", "tem", "input tfrecord")
tf.app.flags.DEFINE_string("ref_img", "01-cam_06", "reference image")

#tf.app.flags.DEFINE_string("ref_img", "0051.png", "reference image such that MPI is perfectly parallel to")

sub_sam = max(FLAGS.sublayers,1)
num_mpi = FLAGS.layers
offset = FLAGS.offset
dmin, dmax = -1,-1#0.2,15
MODEL_VERSION = 'BIG/'

def touint8(img):
  return tf.cast(img * 255.0, tf.uint8)

def crop_sample(sfm,features,cxy=[0,0]):
  #center = #tf.concat(cxy,-1)
  center = tf.cast(cxy,tf.float32)
  center = tf.expand_dims(tf.expand_dims(tf.expand_dims(center,0),0),0)
  h2 = sfm.sh//2

  coor = [cxy[0]+sfm.offset+h2,cxy[1]+sfm.offset+h2,1]#tf.concat([cxy[0]+sfm.offset+32,cxy[1]+sfm.offset+32,1],-1)
  coor = tf.cast(coor,tf.float32)
  coor = tf.expand_dims(tf.expand_dims(coor,0),0)

  H = computeHomography(sfm, features, sfm.features, 3.8)
  newCoords = tf.matmul(coor, tf.linalg.inv(H), transpose_b=True)
  ncenter = newCoords[:,:,:2]/newCoords[:,:,2:3]-sfm.offset-h2
  ncx = tf.cast(ncenter[0,0,0],tf.int32)
  ncy = tf.cast(ncenter[0,0,1],tf.int32)
  ncx = tf.clip_by_value(ncx,0,sfm.w - sfm.sw)
  ncy = tf.clip_by_value(ncy,0,sfm.h - sfm.sh)
  ncenter = tf.expand_dims(ncenter,0)

  crop_img = tf.image.crop_to_bounding_box(features['img'],ncy,ncx,sfm.sh,sfm.sw)

  img_tile = tf.tile(crop_img,[num_mpi,1,1,1])
  psv1 = tf.contrib.resampler.resampler(img_tile,InvzHomoWarp(sfm,features,sfm.features,i=0,center=ncenter-center))
  psv1 = tf.reshape(psv1,(1,num_mpi*sfm.nh,sfm.nw,3))
  return crop_img , psv1, ncenter-center*0

def train(sfm,cx,cy):

    sfm.sh = 120 *3
    sfm.sw = 200 *3

    sfm.num_mpi = num_mpi
    sfm.offset = offset
    print(getPlanes(sfm))

    sfm.nh = sfm.sh + 2*offset
    sfm.nw = sfm.sw + 2*offset

    bh = sfm.h + 2*offset
    bw = sfm.w + 2*offset
    #cx, cy = 600, 400

    iter = tf.compat.v1.placeholder(tf.float32, shape=[], name='iter')
    features0 = load_data(FLAGS.dataset,FLAGS.input,[sfm.h,sfm.w],1,is_shuff = True)
    crop_img, psv1, nc = crop_sample(sfm,features0,[cx,cy])

    int_mpi1 = np.random.uniform(-1, 1,[num_mpi, bh, bw, 3]).astype(np.float32)
    int_mpi2 = np.random.uniform(-5,-3,[num_mpi, bh, bw, 1]).astype(np.float32)

    ref_img = -np.log(np.maximum(1/sfm.ref_img-1,0.001))
    int_mpi1[:,offset:sfm.h + offset,offset:sfm.w + offset,:] = np.array([ref_img])
    int_mpi2[-1] = -1
    

    tt = True
    with tf.compat.v1.variable_scope("Net%d"%(FLAGS.index)):
      mpic = tf.compat.v1.get_variable("mpi_c", initializer=int_mpi1, trainable=tt) 
      mpia = tf.compat.v1.get_variable("mpi_a", initializer=int_mpi2, trainable=tt)

    crop_mpic = tf.image.crop_to_bounding_box(mpic,cy,cx,sfm.nh,sfm.nw)
    crop_mpia = tf.image.crop_to_bounding_box(mpia,cy,cx,sfm.nh,sfm.nw)
    #crop_mpic = mpic[:,cy:cy+sfm.nh,cx:cx+sfm.nw]
    #crop_mpia = mpia[:,cy:cy+sfm.nh,cx:cx+sfm.nw]

    mpic_sig = tf.sigmoid(crop_mpic)
    mpia_sig = tf.sigmoid(crop_mpia)

    lr = tf.compat.v1.train.exponential_decay(0.1,iter,1000,0.5)
    optimizer = tf.compat.v1.train.AdamOptimizer(lr)


    fac = 1.0 #tf.maximum(1 - iter/(1500),0.2)
    tva = tf.constant(0.1) * fac
    tvc = tf.constant(0.005) * fac

    mpi_sig = tf.concat([mpic_sig,mpia_sig],-1)
    #mpi_sig = tf.Print(mpi_sig,[nc])
    img_out = network( sfm, features0, sfm.features, mpi_sig,center = nc*0)
    
    #TODO use normailze before tv
    loss = 0
    loss +=  100000 * tf.reduce_mean(tf.square(img_out[0] - crop_img[-1]))
    loss += tvc * tf.reduce_mean(tf.image.total_variation(mpic_sig))
    loss += tva * tf.reduce_mean(tf.image.total_variation (mpia_sig))
    
    train_op =optimizer.minimize(loss)



    image_out = tf.clip_by_value(img_out,0.0,1.0)
    a_long = tf.reshape(mpi_sig[:,:,:,3:4],(1,num_mpi*sfm.nh,sfm.nw,1))
    c_long = tf.reshape(mpi_sig[:,:,:,:3],(1,num_mpi*sfm.nh,sfm.nw,3))
    
    summary = tf.compat.v1.summary.merge([
                tf.compat.v1.summary.scalar("post0/all_loss", loss),
                tf.compat.v1.summary.image("post0/out1",touint8(tf.concat([crop_img[-1:],image_out],1))),
                tf.compat.v1.summary.image("post0/out1",touint8(tf.concat([image_out,crop_img[-1:]],1))),
                tf.compat.v1.summary.image("post1/o_alpha",touint8(a_long)),
                tf.compat.v1.summary.image("post1/o_color",touint8(c_long)),
                tf.compat.v1.summary.image("post1/o_acolor",touint8(c_long*a_long)),
                tf.compat.v1.summary.image("post2/p1",touint8(psv1)),
                ])
    
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    localpp = "TensorB/"+MODEL_VERSION+FLAGS.dataset
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
      localpp = './model/'+MODEL_VERSION + FLAGS.dataset
      saver = tf.train.Saver()
      ckpt = tf.train.latest_checkpoint(localpp)
      saver.restore(sess, ckpt)
    else: 
        sess.run(tf.compat.v1.global_variables_initializer())


    localpp = './model/'+MODEL_VERSION + FLAGS.dataset 
    if not os.path.exists(localpp):
        os.makedirs(localpp)
    saver = tf.train.Saver()

    #print("Var=",np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    los = 0
    n = num_mpi//20
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
    localpp = "/home2/suttisak/datasets/spaces_dataset/data/resize_2k/" + FLAGS.dataset + "/" + FLAGS.input + ".test"
    testset = tf.data.TFRecordDataset([localpp])
    testset = testset.map(parser).repeat().batch(1).make_one_shot_iterator()
    features = testset.get_next()

    rot = features["r"][0]
    tra = features["t"][0]

    sfm.sh = 120
    sfm.sw = 200

    sfm.num_mpi = num_mpi
    sfm.offset = offset
    print(getPlanes(sfm))

    sfm.nh = sfm.sh + 2*offset
    sfm.nw = sfm.sw + 2*offset

    bh = sfm.h + 2*offset
    bw = sfm.w + 2*offset

    sfm.num_mpi = num_mpi
    sfm.offset = offset

    int_mpi1 = np.random.uniform(-1, 1,[num_mpi, bh, bw, 3]).astype(np.float32)
    int_mpi2 = np.random.uniform(-5,-3,[num_mpi, bh, bw, 1]).astype(np.float32)
    int_mpi2[-1] = -1


    with tf.compat.v1.variable_scope("Net%d"%(FLAGS.index)):
        mpic = tf.compat.v1.get_variable("mpi_c", initializer=int_mpi1, trainable=False)   
        mpia = tf.compat.v1.get_variable("mpi_a", initializer=int_mpi2, trainable=False)
        new_mpi = tf.concat([tf.tile(mpic,[sub_sam,1,1,1]),mpia],-1)

    mpi_sig = tf.sigmoid(new_mpi)
    img_out = network(sfm,features,sfm.features,mpi_sig)

    with tf.compat.v1.variable_scope("post%d"%(FLAGS.index)):
        image_out= tf.clip_by_value(img_out[0],0.0,1.0)

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())
    t_vars = slim.get_variables_to_restore()
    variables_to_restore = [var for var in t_vars if 'Net' in var.name]
    #variables_to_restore = slim.get_variables_to_restore()
    print(variables_to_restore)
    saver = tf.train.Saver(variables_to_restore)
    localpp = './model/'+MODEL_VERSION + FLAGS.dataset  #+ '/mpi'
    ckpt = tf.train.latest_checkpoint(localpp )
    saver.restore(sess, ckpt)

    #webpath = "webpath/"
    webpath = "/var/www/html/suttisak/data/"
    if not os.path.exists(webpath + FLAGS.dataset):
        os.system("mkdir " + webpath + FLAGS.dataset)


    if False:  # make sample picture and video

        for i in range(0,300,1):
          #feed = sess.run(features)
          #out = sess.run(image_out,feed_dict={lod_in:0,rot:feed["r"][0],tra:feed["t"][0]})
          out = sess.run(image_out,feed_dict={lod_in:0})
          if (i%50==0): 
            print(i)
            plt.imsave("webpath/"+FLAGS.dataset+"/%04d.png"%( i),out)
          plt.imsave("result/frame/"+FLAGS.dataset+"_%04d.png"%( i),out)

        cmd = 'ffmpeg -y -i ' + 'result/frame/'+FLAGS.dataset+'_%04d.png -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p webpath/'+FLAGS.dataset+'/moving.mp4'
        print(cmd)
        os.system(cmd)
 
    if True:  # make web viewer

      ret = sess.run(mpi_sig,feed_dict={lod_in:0})
      maxcol = 12
      mpis = []
      cols = []
      for i in range(num_mpi):
        cols.append(ret[i,:,:,:4])
        if len(cols) == maxcol:
          mpis.append(np.concatenate(cols,1))
          cols = []
          #mpis.append(ret[i,:,:,:4])
      if len(cols):
        while len(cols)<maxcol:
          cols.append(np.zeros_like(cols[-1]))
        mpis.append(np.concatenate(cols,1))

      plt.imsave(webpath + FLAGS.dataset+ "/sublayer.png", np.ones_like(mpis[0][:, :, :3]))
      mpis = np.concatenate(mpis, 0)
      plt.imsave(webpath + FLAGS.dataset+ "/mpi%02d.png"%(FLAGS.index), mpis)
      plt.imsave("webpath/" + FLAGS.dataset+ "/mpi%02d.png"%(FLAGS.index), mpis)
      #plt.imsave(webpath + FLAGS.dataset+ "/mpi.png", mpis)
      #plt.imsave(webpath + FLAGS.dataset+ "/mpi_alpha.png", np.tile(mpis[:, :, 3:], (1, 1, 3)))

      namelist = "["
      for ii in range(FLAGS.index+1):
        namelist += "\"%02d\","%(ii)
      namelist += "]"

      ref_r = sfm.ref_r
      ref_t = sfm.ref_t
      with open(webpath + FLAGS.dataset+ "/extrinsics%02d.txt"%(FLAGS.index), "w") as fo:
        for i in range(3):
          for j in range(3):
            fo.write(str(ref_r[ i, j]) + " ")
        fo.write(" ".join([str(x) for x in np.nditer(ref_t)]) + "\n")
      generateConfigGL(webpath + FLAGS.dataset+ "/config.js", sfm.w, sfm.h, getPlanes(sfm),namelist,sub_sam, sfm.ref_fx, sfm.ref_px, sfm.ref_py,FLAGS.scale,FLAGS.offset)
      generateConfigGL("webpath/" + FLAGS.dataset+ "/config.js", sfm.w, sfm.h, getPlanes(sfm),namelist,sub_sam, sfm.ref_fx, sfm.ref_px, sfm.ref_py,FLAGS.scale,FLAGS.offset)


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
      #for cx in [200,600]:
        #for cy in [400,800]:
      train(sfm,0,0)
    print("Jub Jub!!")

if __name__ == "__main__":
  sys.excepthook = colored_hook(
      os.path.dirname(os.path.realpath(__file__)))
  tf.compat.v1.app.run()
