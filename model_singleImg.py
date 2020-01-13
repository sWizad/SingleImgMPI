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
max_step = 1

def create_input(sfm,features,mpi_sig):
  samples = tf.contrib.resampler.resampler(mpi_sig,HomoWarp(sfm,features,sfm.features))
  colord = samples[:,:,:,:3]
  alphad = samples[:,:,:,3:4]
  Transmit = tf.math.cumprod(1-alphad,0,exclusive=True)
  Rrr  = Transmit * colord * alphad
  Rd = tf.math.cumsum(Rrr,0,exclusive=True)
  output = tf.expand_dims(Rd[-1] + Rrr[-1] ,0)
  img = tf.concat([output,features['img'][0:1]],-1)
  img_tile = tf.tile(img,[num_mpi*sub_sam,1,1,1])
  pack = tf.concat([img_tile,alphad*Transmit,colord*Transmit,Rd*Transmit],-1)
  input = tf.contrib.resampler.resampler(pack,InvzHomoWarp(sfm,features,sfm.features))
  return tf.concat([mpi_sig,input],-1), output

    
def UNet(input,depth=3):
  next = input
  chanels = [18*2**i for i in range(depth)]
  layer = []
  for i,c in enumerate(chanels):
    next = lrelu(conv2d(next,c,name="downA"+str(i)))
    next = lrelu(conv2d(next,c,stride=[1,2,2,1],name="downB"+str(i)))
    layer.append(next)

  next = lrelu(conv2d(next,chanels[-1],name="mid"))
  for i,c in enumerate(reversed(chanels)):
    next = tf.concat([next,layer[depth-1-i]],-1)
    next = lrelu(conv2d(next,c,name="up"+str(i)))
    next = upscale2d(next)

  last = conv2d(next,8,name="clast")
  return last

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

    sfm.num_mpi = num_mpi
    sfm.offset = offset

    center = np.array([[[[0,0]]]])

    mask = mask_maker(sfm,features0,sfm.features)

    int_mpi1 = np.random.uniform(-1, 1,[num_mpi, bh, bw, 3]).astype(np.float32)
    int_mpi2 = np.random.uniform(-5,-3,[num_mpi*sub_sam, bh, bw, 1]).astype(np.float32)
    ref_img = -np.log(np.maximum(1/sfm.ref_img-1,0.001))
    int_mpi1[:,offset:sfm.h + offset,offset:sfm.w + offset,:] = np.array([ref_img])
    int_mpi2[-1] += 3

    tt = False
    with tf.compat.v1.variable_scope("Net%d"%(FLAGS.index)):
      mpic = tf.compat.v1.get_variable("mpi_c", initializer=int_mpi1, trainable=tt) 
      mpia = tf.compat.v1.get_variable("mpi_a", initializer=int_mpi2, trainable=tt)
      #last_al = tf.concat([tf.zeros_like(mpia[:-1]),tf.ones_like(mpia[0:1])*5],0)
      #mpia += last_al * tf.maximum(1-iter/100,0)
    
    #mpic = inv_ref(sfm)
    new_mpi = tf.concat([mpic,mpia],-1)
    

    #lr = tf.compat.v1.train.exponential_decay(0.1,iter,1000,0.2)
    lr = 0.00015
    optimizer = tf.compat.v1.train.AdamOptimizer(lr)
    fac = 0.1#(1 - iter/(1500*2))# *0.01
    tva = tf.constant(0.1) * fac #*0.01
    tvc = tf.constant(0.005)  * fac *0.0

    mpi_sig = tf.sigmoid(new_mpi)
    #img_out = network( sfm, features0, sfm.features, mpi_sig, center)
    #img_list = [img_out]
    img_list = []
    train_list = []
    for i in range(max_step):
      with tf.compat.v1.variable_scope("step"+str(i)):
        #features = load_data(FLAGS.dataset,FLAGS.input,[sfm.h,sfm.w],1,is_shuff = False)
        input, img_out = create_input(sfm,features0,mpi_sig)
        img_list.append(img_out)
        if i>0:
          loss =  100000 * tf.reduce_mean(tf.square(img_out[0] - real_img[-1])*mask)
          loss += tvc * tf.reduce_mean(tf.image.total_variation(mpi_sig[:, :, :, :3]))
          train_list.append(slim.learning.create_train_op(loss,optimizer))

        delta = UNet(input,3)
        if i == 0: new_mpi = tf.concat([new_mpi,tf.zeros_like(new_mpi)],-1)
        new_mpi = new_mpi + delta
        mpi_sig = tf.sigmoid(new_mpi[:,:,:,:4])

    img_out = network( sfm, features0, sfm.features, mpi_sig, center)
    img_list.append(img_out)
    loss = 0
    loss +=  100000 * tf.reduce_mean(tf.square(img_out[0] - real_img[-1])*mask)
    loss += tvc * tf.reduce_mean(tf.image.total_variation(mpi_sig[:, :, :, :3]))
    loss += tva * tf.reduce_mean(tf.image.total_variation (mpi_sig[:, :, :, 3:4]))
    train_op = slim.learning.create_train_op(loss,optimizer)
    train_list.append(train_op)
    step_img = tf.concat(img_list, 1)


    image_out = tf.clip_by_value(img_out,0.0,1.0)
    a_long = tf.reshape(mpi_sig[:,:,:,3:4],(1,num_mpi*sub_sam*bh,bw,1))
    c_long = tf.reshape(mpi_sig[:,:,:,:3],(1,num_mpi*sub_sam*bh,bw,3))
    summary = tf.compat.v1.summary.merge([
                tf.compat.v1.summary.scalar("post0/all_loss", loss),
                #tf.compat.v1.summary.image("post1/out0",step_img[:,:,:,:3]),
                tf.compat.v1.summary.image("post0/step",step_img),
                tf.compat.v1.summary.image("post0/out1",tf.concat([real_img[-1:],image_out],1)),
                tf.compat.v1.summary.image("post1/o_alpha",a_long),
                tf.compat.v1.summary.image("post1/o_color",c_long),
                tf.compat.v1.summary.image("post1/oa_color",c_long*a_long),
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

    #t_vars = slim.get_variables_to_restore()
    #var2restore = [var for var in t_vars if 'mpi_a' in var.name and 'Adam' not in var.name ]
    #saver = tf.train.Saver(var2restore)
    #saver.restore(sess, './model/space/'+FLAGS.dataset+'/mpi')



    localpp = './model/space/' + FLAGS.dataset 
    if not os.path.exists(localpp):
        os.makedirs(localpp)
    saver = tf.train.Saver()


    los = 0
    for i in range(FLAGS.epoch + 3):
        feedlis = {iter:i}
        #_,los = sess.run([train_op,loss],feed_dict=feedlis)
        #los = sess.run(loss,feed_dict=feedlis)
        for trainer in train_list:
            _,los = sess.run([trainer,loss],feed_dict=feedlis)
            
        if i%50==0:
            print(i, "loss = ",los )
        if i%20 == 0:
            summ = sess.run(summary,feed_dict=feedlis)
            writer.add_summary(summ,i)
        #if i%200==199:
        #    saver.save(sess, localpp + '/' + str(000))
    #saver.save(sess, localpp + '/mpi')

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
