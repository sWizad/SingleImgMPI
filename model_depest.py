## Make MPI by optimizate from plane sweet volume
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
max_step = 0
MODEL_VERSION = "DepthEst/"

def create_list(sfm,mpia_sig,num_psv=4):
    #make psv
    psv_mpi = []
    iwarps = []
    warps = []
    with tf.compat.v1.variable_scope("psv_list",reuse=tf.compat.v1.AUTO_REUSE):
      for i in range(num_psv):
        mpic = tf.compat.v1.get_variable("psv%d"%(i), shape=[num_mpi, sfm.nh, sfm.nw, 3], trainable=False)
        tf.compat.v1.add_to_collection("psv",mpic)
        psv_mpi.append(mpic)
        iwarp = tf.compat.v1.get_variable("iw%d"%(i), shape=[num_mpi, sfm.nh, sfm.nw, 2], trainable=False)
        tf.compat.v1.add_to_collection("iwarp",iwarp)
        iwarps.append(iwarp)
        warp = tf.compat.v1.get_variable("w%d"%(i), shape=[num_mpi, sfm.h, sfm.w, 2], trainable=False)
        tf.compat.v1.add_to_collection("warp",warp)
        warps.append(warp)

    #make mpic_sig
    nomi = 0
    deci = 1e-8
    for i in range(num_psv):
      samples = tf.contrib.resampler.resampler(mpia_sig,warps[i])
      Transmit = tf.math.cumprod(1-samples,0,exclusive=True)
      Highlight = samples * Transmit
      Highlight = tf.contrib.resampler.resampler(Highlight,iwarps[i])
      deci = deci + Highlight
      nomi = nomi + psv_mpi[i] * Highlight

    return deci, nomi

def initial_psv(sfm,num_psv=4):
    features = load_data(FLAGS.dataset,FLAGS.input,[sfm.h,sfm.w],num_psv,is_shuff = True)
    #initial psv
    name1 = tf.compat.v1.get_collection("psv")
    name2 = tf.compat.v1.get_collection("iwarp")
    name3 = tf.compat.v1.get_collection("warp")
    make_psv = []
    for i in range(num_psv):
      img_tile = tf.tile(features['img'][i:i+1],[num_mpi,1,1,1])
      iwarp = InvzHomoWarp(sfm, features, sfm.features,i=i)
      make_psv.append(name1[i].assign(tf.contrib.resampler.resampler(img_tile,iwarp)))
      make_psv.append(name2[i].assign(iwarp))
      make_psv.append(name3[i].assign(HomoWarp(sfm,features,sfm.features,i=i)))
    return make_psv


@tf.custom_gradient
def sigmoid_hack(mpi):
  ex = tf.math.exp(-mpi)
  def grad(dy):
    dx = ex/tf.square(ex+1)
    dx = tf.where(tf.abs(mpi)>1,0.196611933*tf.sqrt(dx),dx)
    #dx = tf.sqrt(dx)
    return dx * dy
  return 1/(1+ex), grad

def DepthEst(sfm,depth=5):
    int_mpi1 = np.zeros([1, sfm.nh, sfm.nw, 3],dtype=np.float32)
    int_mpi1[:,sfm.offset:sfm.h + sfm.offset,sfm.offset:sfm.w +sfm. offset,:] = np.array([sfm.ref_img])
    ref_img = tf.convert_to_tensor(int_mpi1)
    next = ref_img
    chanels = [6*2**i for i in range(depth)]
    layer = []
    for i,c in enumerate(chanels):
        next = lrelu(conv2d(next,c,stride=[1,2,2,1],name="downA"+str(i)))
        layer.append(next)

    next = lrelu(conv2d(next,chanels[-1],name="mid"))
    for i,c in enumerate(reversed(chanels)):
        next = tf.concat([next,layer[depth-1-i]],-1)
        next = lrelu(conv2d(next,c,name="up"+str(i)))
        next = upscale2d(next)

    next = tf.concat([ref_img,next],-1)
    last = conv2d(next,1,name="clast")
    last = tf.maximum(last,0.0001)
    return last


def Depth2Alpha(sfm,depth,sd=1.0):
  tt = (sfm.dmin/depth-1)*(num_mpi-1)/(sfm.dmin/sfm.dmax-1) + 1
  base = tf.convert_to_tensor(np.array(range(0,num_mpi+1)).reshape(-1,1,1,1).astype(np.float32))
  z = (tt-base)/sd
  cumprob = 0.3535533*tf.math.tanh(1.12838*z*(1+0.08943*z**2))+0.5
  mpia = cumprob[:-1]-cumprob[1:]
  return mpia

def train_MPI(sfm):
    iter = tf.compat.v1.placeholder(tf.float32, shape=[], name='iter')
    features0 = load_data(FLAGS.dataset,FLAGS.input,[sfm.h,sfm.w],1,is_shuff = True)
    real_img = features0['img']

    sfm.num_mpi = num_mpi
    sfm.offset = offset
    print(getPlanes(sfm))
    sfm.nh = sfm.h + 2*offset
    sfm.nw = sfm.w + 2*offset

  
    int_mpi1 = np.random.uniform(-1, 1,[num_mpi, sfm.nh, sfm.nw, 3]).astype(np.float32)
    int_mpi2 = np.random.uniform(-5,-3,[num_mpi, sfm.nh, sfm.nw, 1]).astype(np.float32)
    ref_img = -np.log(np.maximum(1/sfm.ref_img-1,0.001))
    int_mpi1[:,offset:sfm.h + offset,offset:sfm.w + offset,:] = np.array([ref_img])
    with tf.compat.v1.variable_scope("Net%d"%(FLAGS.index)):
      mpic = tf.compat.v1.get_variable("mpi_c", initializer=int_mpi1) 
      mpia = tf.compat.v1.get_variable("mpi_a", initializer=int_mpi2)

    #mpia_sig = tf.sigmoid(mpia)
    mpia_sig = sigmoid_hack(mpia)
    mpic_sig = tf.sigmoid(mpic)
    depth = DepthEst(sfm,6)
    sd = tf.clip_by_value(1 - (iter-1000)/4000,0.5,1.0) 
    mpia_est = Depth2Alpha(sfm,depth,sd)

    num_psv = 4
    deci, nomi = create_list(sfm,mpia_sig,num_psv)
    make_psv = initial_psv(sfm,num_psv)
    weight = mpia_sig #* tf.math.cumprod(1-mpia_sig,0,exclusive=True)
    deci = deci + weight
    nomi = nomi + mpic_sig*weight
    condition = tf.tile(deci>1e-6,[1,1,1,3])
    mpic_psv = tf.where(condition,nomi/deci,mpic_sig)

    lr = tf.compat.v1.train.exponential_decay(0.1,iter,1000,0.5)
    optimizer = tf.compat.v1.train.AdamOptimizer(lr)
    tva = tf.constant(0.1) #* fac
    tvc = tf.constant(0.0005) #* fac
    mpi_sig = tf.concat([mpic_psv,mpia_sig],-1)
    img_out = network( sfm, features0, sfm.features, mpi_sig)
    mask = mask_maker(sfm,features0,sfm.features)
    loss =  100000 * tf.reduce_mean(tf.square(img_out[0] - real_img[-1])*mask)
    loss += tvc * tf.reduce_mean(tf.image.total_variation(mpic_sig))
    loss += tva * tf.reduce_mean(tf.image.total_variation (mpia_sig))
    #train_op = slim.learning.create_train_op(loss,optimizer)
    train_op = optimizer.minimize(loss)

    image_out = tf.clip_by_value(img_out,0.0,1.0)
    a_long = tf.reshape(mpia_sig,(1,num_mpi*sfm.nh,sfm.nw,1))
    c_long = tf.reshape(mpic_sig,(1,num_mpi*sfm.nh,sfm.nw,3))
    
    #2nd model
    deci, nomi = create_list(sfm,mpia_est,num_psv)
    make_psv = initial_psv(sfm,num_psv)
    deci = deci + mpia_est
    nomi = nomi + mpic_sig*mpia_est
    condition = tf.tile(deci>1e-6,[1,1,1,3])
    mpic_psv = tf.where(condition,nomi/deci,mpic_sig)

    optimizer2 = tf.compat.v1.train.AdamOptimizer(0.00015)
    mpi_sig = tf.concat([mpic_psv,mpia_est],-1)
    img_out2 = network( sfm, features0, sfm.features, mpi_sig)
    loss2 =  100000 * tf.reduce_mean(tf.square(img_out2[0] - real_img[-1])*mask)
    loss2 += 1000 * tf.reduce_mean(tf.square(mpia_sig-mpia_est ))
    varmpi = [var for var in slim.get_variables_to_restore() if 'Net' not in var.name ]
    train_op2 = optimizer2.minimize(loss2,var_list=varmpi)
    a2_long = tf.reshape(mpia_est,(1,num_mpi*sfm.nh,sfm.nw,1))
    depth = tf.tile(depth,[1,1,1,3])


    summary = tf.compat.v1.summary.merge([
                tf.compat.v1.summary.scalar("post0/all_loss", loss),
                tf.compat.v1.summary.scalar("post0/all_loss2", loss2),
                tf.compat.v1.summary.image("post0/out1",tf.concat([real_img[-1:],image_out],1)),
                #tf.compat.v1.summary.image("post0/out2",tf.concat([depth,img_out2],1)),
                tf.compat.v1.summary.image("post0/out2",img_out2),
                tf.compat.v1.summary.image("post1/o_alpha",a_long),
                tf.compat.v1.summary.image("post1/o_alpha",a2_long),
                tf.compat.v1.summary.image("post1/o_color",c_long),
                tf.compat.v1.summary.image("post1/o_acolor",c_long*a_long),
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

    localpp = './model/'+ MODEL_VERSION + FLAGS.dataset 
    if not FLAGS.restart:
      sess.run(tf.compat.v1.global_variables_initializer())
      saver = tf.train.Saver()
      ckpt = tf.train.latest_checkpoint(localpp)
      saver.restore(sess, ckpt)
    else: 
        sess.run(tf.compat.v1.global_variables_initializer())


    if not os.path.exists(localpp):
        os.makedirs(localpp)
    saver = tf.train.Saver()

    sess.run(make_psv)
    print("make PSV")

    los = 0
    n = num_mpi//20
    for i in range(FLAGS.epoch + 3):
      feedlis = {iter:i}
      if i<1000:
        _,los = sess.run([train_op,loss],feed_dict=feedlis)
      else:
        _,los = sess.run([train_op2,loss2],feed_dict=feedlis)

      if i%50==0:
          print(i, "loss = ",los )
      if i%20 == 0:
          summ = sess.run(summary,feed_dict=feedlis)
          writer.add_summary(summ,i)
      if i%200==199:
          saver.save(sess, localpp + '/' + str(000))
          sess.run(make_psv)
    #saver.save(sess, localpp + '/mpi')


def main(argv):
    sfm = SfMData(FLAGS.dataset,
                FLAGS.ref_img,
                "",
                FLAGS.scale,
                dmin,
                dmax)

    train_MPI(sfm)
    print("Jub Jub!!")

if __name__ == "__main__":
  sys.excepthook = colored_hook(
      os.path.dirname(os.path.realpath(__file__)))
  tf.compat.v1.app.run()