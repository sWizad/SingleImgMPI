## Make MPI by direct optimization
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

tf.app.flags.DEFINE_integer("layers", 24, "number of planes")
tf.app.flags.DEFINE_integer("sublayers", 1, "number of sub planes")
tf.app.flags.DEFINE_integer("epoch", 2000, "Training steps")
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

def train_MPI(sfm):
    iter = tf.compat.v1.placeholder(tf.float32, shape=[], name='iter')
    lod_in = tf.compat.v1.placeholder(tf.float32, shape=[], name='lod_in')
    ratio = tf.compat.v1.placeholder(tf.float32, shape=[num_mpi,1,1,1], name='lod_in')
    
    features0 = load_data(FLAGS.dataset,FLAGS.input,[sfm.h,sfm.w],1,is_shuff = True)
    real_img = features0['img']

    bh = sfm.h + 2 * offset
    bw = sfm.w + 2 * offset
    sfm.nw = sfm.w + 2 * offset
    sfm.nh = sfm.h + 2 * offset

    sfm.sw = sfm.w
    sfm.sh = sfm.h

    sfm.num_mpi = num_mpi
    sfm.offset = offset
    print(getPlanes(sfm))

    center = np.array([[[[0,0]]]])

    mask = mask_maker(sfm,features0,sfm.features)

    int_mpi1 = np.random.uniform(-1, 1,[num_mpi, bh, bw, 3]).astype(np.float32)
    int_mpi2 = np.random.uniform(-5,-3,[num_mpi*sub_sam, bh, bw, 1]).astype(np.float32)

    ref_img = -np.log(np.maximum(1/sfm.ref_img-1,0.001))
    int_mpi1[:,offset:sfm.h + offset,offset:sfm.w + offset,:] = np.array([ref_img])
    int_mpi2[-1] = -1
    

    tt = True
    with tf.compat.v1.variable_scope("Net%d"%(FLAGS.index)):
      mpic = tf.compat.v1.get_variable("mpi_c", initializer=int_mpi1, trainable=tt) 
      mpia = tf.compat.v1.get_variable("mpi_a", initializer=int_mpi2, trainable=tt)
      white_plane = tf.ones_like(mpia)#tf.concat([tf.zeros_like(mpia[:-1]),tf.ones_like(mpia[0:1])*5],0)
      mpia += white_plane * ratio #last_al * tf.maximum(1-iter/100,0)
      new_mpi = tf.concat([tf.tile(mpic,[sub_sam,1,1,1]),mpia],-1)
      tf.compat.v1.add_to_collection("mpia",mpia)

    lr = tf.compat.v1.train.exponential_decay(0.1,iter,1000,0.2)
    optimizer = tf.compat.v1.train.AdamOptimizer(lr)

    fac = tf.maximum(1 - iter/(1500),0)# *0.01
    tva = tf.constant(0.1) * fac #*0.01
    tvc = tf.constant(0.005)  * fac *0.01

    mpi_sig = tf.sigmoid(new_mpi)
    img_out = network( sfm, features0, sfm.features, mpi_sig, center)

    #img_tile = tf.tile(tf.concat([real_img[0:1],img_out],-1),[num_mpi*sub_sam,1,1,1])
    #img_tile = tf.contrib.resampler.resampler(img_tile,InvzHomoWarp(sfm,features0,sfm.features))
    #step_img = tf.reshape(img_tile,(1,num_mpi*sub_sam*bh,bw,6))

    loss = 0
    loss +=  100000 * tf.reduce_mean(tf.square(img_out[0] - real_img[-1])*mask)
    loss += tvc * tf.reduce_mean(tf.image.total_variation(mpi_sig[:, :, :, :3]))
    loss += tva * tf.reduce_mean(tf.image.total_variation (mpi_sig[:, :, :, 3:4]))
    varmpi = [var for var in slim.get_variables_to_restore() if 'mpi_a' in var.name ]
    train_op0 = slim.learning.create_train_op(loss,optimizer,variables_to_train=varmpi)
    train_op = slim.learning.create_train_op(loss,optimizer)



    image_out = tf.clip_by_value(img_out,0.0,1.0)
    a_long = tf.reshape(mpi_sig[:,:,:,3:4],(1,num_mpi*sub_sam*bh,bw,1))
    c_long = tf.reshape(mpi_sig[:,:,:,:3],(1,num_mpi*sub_sam*bh,bw,3))
    summary = tf.compat.v1.summary.merge([
                tf.compat.v1.summary.scalar("post0/all_loss", loss),
                #tf.compat.v1.summary.image("post1/out0",step_img[:,:,:,:3]),
                tf.compat.v1.summary.image("post0/out1",tf.concat([real_img[-1:],image_out],1)),
                tf.compat.v1.summary.image("post1/o_alpha",a_long),
                tf.compat.v1.summary.image("post1/o_color",c_long),
                tf.compat.v1.summary.image("post1/o_acolor",c_long*a_long),
                ])

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    localpp = "TensorB/"+FLAGS.dataset
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
      localpp = './model/space/' + FLAGS.dataset
      var2restore = [var for var in t_vars if 'Net0' in var.name ]
      print(var2restore)
      saver = tf.train.Saver(var2restore)
      ckpt = tf.train.latest_checkpoint(localpp )
      saver.restore(sess, ckpt)
    else:
      sess.run(tf.compat.v1.global_variables_initializer())


    localpp = './model/space/' + FLAGS.dataset 
    if not os.path.exists(localpp):
        os.makedirs(localpp)
    saver = tf.train.Saver()


    los = 0
    for i in range(FLAGS.epoch + 3):
      rr = np.zeros((num_mpi,1,1,1))
      #rr[-1] = 3-i/100
      #rr = np.clip(rr,0,5)
      ii = i//100 
      #Sub Back2Front
      rr[39-ii*2+2:39-ii*2] = -6 * (i-ii*100)/100
      #if i<5000:
      #  rr[39-ii] = 6 * (1.2+ii - i/100)#

      #if i>500 and i<600: rr[8]= -3
      #if i>600 and i<700: rr[10]= -3
      #if i>700 and i<300: rr[12]= -3
      #if i>800 and i<900: rr[-1]= -3
      feedlis = {iter:i,ratio:rr}
      if i<0:
        _,los = sess.run([train_op0,loss],feed_dict=feedlis)
      else:
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
    sfm.num_mpi = num_mpi
    sfm.offset = offset

    int_mpi1 = np.random.uniform(-1, 1,[num_mpi, nh, nw, 3]).astype(np.float32)
    int_mpi2 = np.random.uniform(-5,-3,[num_mpi*sub_sam, nh, nw, 1]).astype(np.float32)

    ref_img = -np.log(np.maximum(1/sfm.ref_img-1,0.001))
    int_mpi1[:,offset:sfm.h + offset,offset:sfm.w + offset,:] = np.array([ref_img])
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
    localpp = './model/space/' + FLAGS.dataset  #+ '/mpi'
    ckpt = tf.train.latest_checkpoint(localpp )
    saver.restore(sess, ckpt)

    webpath = "webpath/"  #"/var/www/html/orbiter/"
    if not os.path.exists(webpath + FLAGS.dataset):
        os.system("mkdir " + webpath + FLAGS.dataset)


    if True:  # make sample picture and video

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
      mpis = []
      for i in range(num_mpi):
          mpis.append(ret[i,:,:,:4])

      mpis = np.concatenate(mpis, 1)
      plt.imsave(webpath + FLAGS.dataset+ "/mpi%02d.png"%(FLAGS.index), mpis)
      plt.imsave(webpath + FLAGS.dataset+ "/mpi.png", mpis)
      plt.imsave(webpath + FLAGS.dataset+ "/mpi_alpha.png", np.tile(mpis[:, :, 3:], (1, 1, 3)))

      namelist = "["
      for ii in range(FLAGS.index+1):
        namelist += "\"%02d\","%(ii)
      namelist += "]"
      print(namelist)

      ref_r = sfm.ref_r
      ref_t = sfm.ref_t
      with open(webpath + FLAGS.dataset+ "/extrinsics%02d.txt"%(FLAGS.index), "w") as fo:
        for i in range(3):
          for j in range(3):
            fo.write(str(ref_r[ i, j]) + " ")
        fo.write(" ".join([str(x) for x in np.nditer(ref_t)]) + "\n")

      #generateWebGL(webpath + FLAGS.dataset+ "/index.html", sfm.w, sfm.h, getPlanes(sfm),namelist,sub_sam, sfm.ref_fx, sfm.ref_px, sfm.ref_py)
      generateConfigGL(webpath + FLAGS.dataset+ "/config.js", sfm.w, sfm.h, getPlanes(sfm),namelist,sub_sam, sfm.ref_fx, sfm.ref_px, sfm.ref_py)


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
