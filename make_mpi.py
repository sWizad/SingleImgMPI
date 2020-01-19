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
      white_plane = tf.ones_like(mpia)
      mpia += white_plane * ratio
      #new_mpi = tf.concat([tf.tile(mpic,[sub_sam,1,1,1]),mpia],-1)
      tf.compat.v1.add_to_collection("mpia",mpia)

    mpia_sig = tf.sigmoid(mpia)
    mpic0_sig = tf.sigmoid(mpic)


    lr = tf.compat.v1.train.exponential_decay(0.1,iter,1000,0.5)
    optimizer = tf.compat.v1.train.AdamOptimizer(lr)


    fac = 1.0#tf.maximum(1 - iter/(1500),0)# *0.01
    tva = tf.constant(0.1) * fac #*0.01
    tvc = tf.constant(0.005) * 0.05  #* fac *0.1

    mode = 3
    if mode>1:
      features = load_data(FLAGS.dataset,FLAGS.input,[sfm.h,sfm.w],1,is_shuff = True)
      img_tile = tf.tile(features['img'],[num_mpi*sub_sam,1,1,1])
      iwarp = InvzHomoWarp(sfm, features, sfm.features)
      mpic_sig = tf.contrib.resampler.resampler(img_tile,iwarp)
      mask1 = mask_maker(sfm,features0,features)
      if mode==3:
        samples = tf.contrib.resampler.resampler(mpia_sig,HomoWarp(sfm,features,sfm.features))
        Transmit = tf.math.cumprod(1-samples,0,exclusive=True)
        Highlight = samples * Transmit
        Highlight = tf.contrib.resampler.resampler(Highlight,iwarp)
        weight = mpia_sig * tf.math.cumprod(1-mpia_sig,0,exclusive=True)
        condition = tf.tile(Highlight+weight>1e-4,[1,1,1,3])
        mpic_sig = tf.where(condition,(mpic_sig*Highlight + mpic0_sig*weight)/(Highlight+weight),mpic0_sig)


    mpi_sig0 = tf.concat([mpic0_sig,mpia_sig],-1)
    img_out0 = network( sfm, features0, sfm.features, mpi_sig0, center)
    img_out = img_out0


    loss = 0
    loss +=  100000 * tf.reduce_mean(tf.square(img_out0[0] - real_img[-1])*mask)
    loss += tvc * tf.reduce_mean(tf.image.total_variation(mpi_sig0[:, :, :, :3]))
    loss += tva * tf.reduce_mean(tf.image.total_variation (mpi_sig0[:, :, :, 3:4]))
    #loss2 = loss
    if mode>1:
      mpi_sig = tf.concat([mpic_sig,mpia_sig],-1)
      img_out = network( sfm, features0, sfm.features, mpi_sig, center)
      loss2 =  100000 * tf.reduce_mean(tf.square(img_out[0] - real_img[-1])*mask*mask1)
      loss2 += tva * tf.reduce_mean(tf.image.total_variation (mpia_sig))
    varmpi = [var for var in slim.get_variables_to_restore() if 'mpi_a' in var.name ]
    train_op0 = slim.learning.create_train_op(loss2,optimizer,variables_to_train=varmpi)
    varmpi = [var for var in slim.get_variables_to_restore() if 'mpi_c' in var.name ]
    train_op = slim.learning.create_train_op(loss,optimizer,variables_to_train=varmpi)



    #h_long = tf.reshape(Highlight,(1,num_mpi*sub_sam*bh,bw,3))

    image_out = tf.clip_by_value(img_out,0.0,1.0)
    a_long = tf.reshape(mpi_sig0[:,:,:,3:4],(1,num_mpi*sub_sam*bh,bw,1))
    c_long = tf.reshape(mpi_sig[:,:,:,:3],(1,num_mpi*sub_sam*bh,bw,3))
    blue = np.array([0.47,0.77,0.93]).reshape((1,1,1,3)) * tf.cast(a_long<0.01,tf.float32)
    
    summary = tf.compat.v1.summary.merge([
                tf.compat.v1.summary.scalar("post0/all_loss", loss),
                #tf.compat.v1.summary.image("post1/out0",step_img[:,:,:,:3]),
                tf.compat.v1.summary.image("post0/out1",tf.concat([real_img[-1:],image_out],1)),
                tf.compat.v1.summary.image("post1/o_alpha",a_long),
                tf.compat.v1.summary.image("post1/o_color",c_long),
                tf.compat.v1.summary.image("post1/o_acolor",c_long*a_long+blue),
                #tf.compat.v1.summary.image("post2/hL",h_long),
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
    n = num_mpi//20
    for i in range(FLAGS.epoch + 3):
      rr = np.zeros((num_mpi,1,1,1))
      if i<1100:
        ii = i//100 
        f = 1 #ii/20
        #Sub Back2Front
        jj = num_mpi-1-ii*n
        rr[jj:jj+n] = -4*f * (i-ii*100)/100
        #rr[jj:jj+n] = -6 * (1.1+ii - i/100)
        #Plus Front2Back
        jj = ii*n
        #rr[jj:jj+n] = 4 *f* (i-ii*100)/100
        #sub Front2Back
        rr[jj:jj+n] = -4 *f* (i-ii*100)/100

        #last
        #if (i//40)% 5 ==0:
        #  rr[-1] = -12 * (i-(i//40)*40)/40
      elif i<1300:
        rr = rr+4 * (i-ii*100)/100

      feedlis = {iter:i,ratio:rr}
      if i<1100:
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

def network_hack2(sfm, features, ref_feat, mpi, center=0):
  @tf.custom_gradient
  def composit(mpi):
    warp = HomoWarp(sfm, features, ref_feat, center=center)
    samples = tf.contrib.resampler.resampler(mpi,warp)
    colord = samples[:,:,:,:3]
    alphad = samples[:,:,:,3:4]
    Transmit = tf.math.cumprod(1-alphad,0,exclusive=True)
    Rrr  = Transmit * colord * alphad
    Rd = tf.math.cumsum(Rrr,0,exclusive=True)
    output = Rd[-1:]+Rrr[-1:]
    #output = tf.reduce_sum(Transmit * colord* alphad,0,keepdims=True)
    
    def grad(dy):
      dOdc = alphad*Transmit * dy
      dOda = tf.reduce_sum((colord-Rd)*Transmit*dy,-1,keepdims=True)
      pack = tf.concat([dOdc,dOda],-1)
      return tf.contrib.resampler.resampler(pack,InvzHomoWarp(sfm,features,sfm.features)) 
    return output, grad

  return composit(mpi)

def network_hack(sfm, features, ref_feat, mpi, center=0):
  @tf.custom_gradient
  def composit(samples):
    colord = samples[:,:,:,:3]
    alphad = samples[:,:,:,3:4]
    Transmit = tf.math.cumprod(1-alphad,0,exclusive=True)
    Rrr  = Transmit * colord * alphad
    Rd = tf.math.cumsum(Rrr,0,exclusive=True)
    output = Rd[-1:]+Rrr[-1:]
    def grad(dy):
      #Tran = tf.where(alphad>0.95,tf.math.pow(Transmit,4/5),Transmit)
      Tran = tf.concat([tf.ones_like(Transmit[0:1]),Transmit[:-1]],0)
      Tran = tf.sqrt(Tran*Transmit)
      dOdc = alphad* Transmit* dy
      dOda = tf.reduce_sum((colord-Rd)*Tran*dy,-1,keepdims=True)
      return tf.concat([dOdc,dOda],-1)
    return output, grad

  warp = HomoWarp(sfm, features, ref_feat, center=center)
  samples = tf.contrib.resampler.resampler(mpi,warp)
  output = composit(samples)
  
  return output

@tf.custom_gradient
def sigmoid_hack(mpi):
  ex = tf.math.exp(-mpi)
  def grad(dy):
    dx = ex/tf.square(ex+1)
    #dx = tf.where(tf.abs(mpi)>1,0.196611933*tf.sqrt(dx),dx)
    dx = tf.sqrt(dx)
    return dx * dy
  return 1/(1+ex), grad

def UNet(input,depth=3):
  next = input
  chanels = [8*2**i for i in range(depth)]
  layer = []
  for i,c in enumerate(chanels):
    next = lrelu(conv2d(next,c,stride=[1,2,2,1],name="downA"+str(i)))
    #next = lrelu(conv2d(next,c,name="downB"+str(i)))
    layer.append(next)

  next = lrelu(conv2d(next,chanels[-1],name="mid"))
  for i,c in enumerate(reversed(chanels)):
    next = tf.concat([next,layer[depth-1-i]],-1)
    next = lrelu(conv2d(next,c,name="up"+str(i)))
    next = upscale2d(next)

  next = tf.concat([input,next],-1)
  last = conv2d(next,1,name="clast")
  return last

def train_Alpha(sfm):
    iter = tf.compat.v1.placeholder(tf.float32, shape=[], name='iter')
    lod_in = tf.compat.v1.placeholder(tf.float32, shape=[], name='lod_in')
    ratio = tf.compat.v1.placeholder(tf.float32, shape=[num_mpi,1,1,1], name='ratio')
    
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

    int_mpi1 = np.random.uniform(-1, 1,[num_mpi, bh, bw, 3]).astype(np.float32)*0
    int_mpi2 = np.random.uniform(-5,-3,[num_mpi*sub_sam, bh, bw, 1]).astype(np.float32)

    ref_img = -np.log(np.maximum(1/sfm.ref_img-1,0.001))
    int_mpi1[:,offset:sfm.h + offset,offset:sfm.w + offset,:] = np.array([ref_img])
    #int_mpi2[-1] = -1
    

    num_psv = 4
    psv_mpi = []
    iwarps = []
    warps = []
    for i in range(num_psv):
      mpic = tf.compat.v1.get_variable("psv%d"%(i), initializer=int_mpi1, trainable=False)
      tf.compat.v1.add_to_collection("psv",mpic)
      psv_mpi.append(mpic)
      iwarp = tf.compat.v1.get_variable("iw%d"%(i), shape=[num_mpi, bh, bw, 2], trainable=False)
      tf.compat.v1.add_to_collection("iwarp",iwarp)
      iwarps.append(iwarp)
      warp = tf.compat.v1.get_variable("w%d"%(i), shape=[num_mpi, sfm.h, sfm.w, 2], trainable=False)
      tf.compat.v1.add_to_collection("warp",warp)
      warps.append(warp)

    features = load_data(FLAGS.dataset,FLAGS.input,[sfm.h,sfm.w],num_psv,is_shuff = False)
    #initial psv
    name1 = tf.compat.v1.get_collection("psv")
    name2 = tf.compat.v1.get_collection("iwarp")
    name3 = tf.compat.v1.get_collection("warp")
    make_psv = []
    for i in range(num_psv):
      img_tile = tf.tile(features['img'][i:i+1],[num_mpi*sub_sam,1,1,1])
      iwarp = InvzHomoWarp(sfm, features, sfm.features,i=i)
      make_psv.append(name1[i].assign(tf.contrib.resampler.resampler(img_tile,iwarp)))
      make_psv.append(name2[i].assign(iwarp))
      make_psv.append(name3[i].assign(HomoWarp(sfm,features,sfm.features,i=i)))

    tt = True
    with tf.compat.v1.variable_scope("Net%d"%(FLAGS.index)):
      mpic = tf.compat.v1.get_variable("mpi_c", initializer=int_mpi1, trainable=tt) 
      mpia = tf.compat.v1.get_variable("mpi_a", initializer=int_mpi2, trainable=tt)
      mpia = mpia #+ UNet(mpia,4)
      mpia += ratio

    #mpia_sig = tf.sigmoid(mpia)
    mpia_sig = sigmoid_hack(mpia)
    mpic0_sig = tf.sigmoid(mpic)

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
    weight = mpia_sig #* tf.math.cumprod(1-mpia_sig,0,exclusive=True)
    deci = deci + weight
    nomi = nomi + mpic0_sig*weight
    condition = tf.tile(deci>1e-6,[1,1,1,3])
    mpic_sig = tf.where(condition,nomi/deci,mpic0_sig)

    lr = tf.compat.v1.train.exponential_decay(0.1,iter,2000,0.5)
    #lr = 0.00015
    optimizer = tf.compat.v1.train.AdamOptimizer(lr)


    fac = tf.maximum(1 - iter/(1500),0.2)# *0.01
    tva = tf.constant(0.1) * fac #*0.01
    tvc = tf.constant(0.005) * fac #*0.1

    mpi_sig0 = tf.concat([mpic0_sig,mpia_sig],-1)
    img_out0 = network( sfm, features0, sfm.features, mpi_sig0, center)
    img_out = img_out0


    loss = 0
    loss +=  100000 * tf.reduce_mean(tf.square(img_out0[0] - real_img[-1])*mask)
    grad = tf.gradients(loss, [mpi_sig0])
    loss += tvc * tf.reduce_mean(tf.image.total_variation(mpic0_sig))
    loss += tva * tf.reduce_mean(tf.image.total_variation (mpia_sig))
    #varmpi = [var for var in slim.get_variables_to_restore() if 'mpi_c' in var.name ]
    
    mpi_sig = tf.concat([mpic_sig,mpia_sig],-1)
    img_out = network( sfm, features0, sfm.features, mpi_sig, center)
    loss2 =  100000 * tf.reduce_mean(tf.square(img_out[0] - real_img[-1])*mask)
    loss2 += tva * tf.reduce_mean(tf.image.total_variation (mpia_sig))
    varmpi = [var for var in slim.get_variables_to_restore() if 'mpi_a' in var.name ]
    #print("var",varmpi)
    train_op = slim.learning.create_train_op(loss2,optimizer,variables_to_train=varmpi)
    #train_op = slim.learning.create_train_op(loss2,optimizer)
    train_op0 = slim.learning.create_train_op(loss,optimizer)



    image_out = tf.clip_by_value(img_out0,0.0,1.0)
    a_long = tf.reshape(mpia_sig,(1,num_mpi*sub_sam*bh,bw,1))
    c0_long = tf.reshape(mpic0_sig,(1,num_mpi*sub_sam*bh,bw,3))
    #c0_long = tf.reshape(grad[0],(1,num_mpi*sub_sam*bh,bw,4))
    c_long = tf.reshape(mpic_sig,(1,num_mpi*sub_sam*bh,bw,3))
    blue = np.array([0.47,0.77,0.93]).reshape((1,1,1,3)) * tf.cast(a_long<0.01,tf.float32)
    red = np.array([0.77,0.77,0.47]).reshape((1,1,1,3)) * tf.cast(tf.abs(a_long-0.8)<0.1,tf.float32)
    red += np.array([0.95,0.3,0.2]).reshape((1,1,1,3)) * tf.cast(tf.abs(a_long-0.95)<0.05,tf.float32)
    
    summary = tf.compat.v1.summary.merge([
                tf.compat.v1.summary.scalar("post0/all_loss", loss),
                tf.compat.v1.summary.image("post0/out1",tf.concat([real_img[-1:],image_out],1)),
                tf.compat.v1.summary.image("post1/o_alpha",a_long),
                tf.compat.v1.summary.image("post1/o_color",c_long),
                tf.compat.v1.summary.image("post1/o_color0",c0_long),
                tf.compat.v1.summary.image("post1/o_acolor",blue+red),
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

    sess.run(make_psv)
    print("make PSV")

    los = 0
    n = num_mpi//20
    for i in range(FLAGS.epoch + 3):
      rr = np.zeros((num_mpi,1,1,1))

      """
      ii = i//100 
      if i>700 and i<1000:
        rr[26-ii] +=-7 * (i-ii*100)/100 #18
        rr[31-ii] +=-7 * (i-ii*100)/100 #23
        rr[33-ii] +=-7 * (i-ii*100)/100 #25

      ii = i//100 
      if i>200 and i<300:
        rr[::4] = rr[::4]+4 * (i-ii*100)/100
      if i>300 and i<400:
        rr[1::4] = rr[1::4]+4 * (i-ii*100)/100
      if i>400 and i<500:
        rr[2::4] = rr[::4]+4 * (i-ii*100)/100
      if i>500 and i<600:
        rr[3::4] = rr[1::4]+4 * (i-ii*100)/100
      if i>600 and i<700:
        rr[::4] = rr[::4]-6 * (i-ii*100)/100
      if i>700 and i<800:
        rr[1::4] = rr[1::4]-6 * (i-ii*100)/100
      if i>800 and i<900:
        rr[2::4] = rr[::4]-6 * (i-ii*100)/100
      if i>900 and i<1000:
        rr[3::4] = rr[1::4]-6 * (i-ii*100)/100

      if i<1100 and i>200:
        ii = i//100 
        f = 1 #ii/20
        #Sub Back2Front
        jj = num_mpi-1-ii*n
        rr[jj:jj+n] = -4*f * (i-ii*100)/100
        #rr[jj:jj+n] = -6 * (1.1+ii - i/100)
        #Plus Front2Back
        jj = ii*n
        #rr[jj:jj+n] = 4 *f* (i-ii*100)/100
        #sub Front2Back
        rr[jj:jj+n] = -4 *f* (i-ii*100)/100

      elif i<1300:
        rr = rr+4 * (i-ii*100)/100
      """
      feedlis = {iter:i,ratio:rr}
      if i%3>0:
        _,los = sess.run([train_op,loss],feed_dict=feedlis)
      else:
        _,los = sess.run([train_op0,loss],feed_dict=feedlis)

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
      generateConfigGL(webpath + FLAGS.dataset+ "/config.js", sfm.w, sfm.h, getPlanes(sfm),namelist,sub_sam, sfm.ref_fx, sfm.ref_px, sfm.ref_py,FLAGS.offset)


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
        #train_MPI(sfm)
        train_Alpha(sfm)
    print("Jub Jub!!")

if __name__ == "__main__":
  sys.excepthook = colored_hook(
      os.path.dirname(os.path.realpath(__file__)))
  tf.compat.v1.app.run()
