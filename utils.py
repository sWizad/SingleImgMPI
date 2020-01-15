import os
import sys
import json
import tensorflow as tf
import numpy as np
import traceback
import math


def colored_hook(home_dir):
  """Colorizes python's error message.
  Args:
    home_dir: directory where code resides (to highlight your own files).
  Returns:
    The traceback hook.
  """

  def hook(type_, value, tb):
    def colorize(text, color, own=0):
      """Returns colorized text."""
      endcolor = "\x1b[0m"
      codes = {
          "green": "\x1b[0;32m",
          "green_own": "\x1b[1;32;40m",
          "red": "\x1b[0;31m",
          "red_own": "\x1b[1;31m",
          "yellow": "\x1b[0;33m",
          "yellow_own": "\x1b[1;33m",
          "black": "\x1b[0;90m",
          "black_own": "\x1b[1;90m",
          "cyan": "\033[1;36m",
      }
      return codes[color + ("_own" if own else "")] + text + endcolor

    for filename, line_num, func, text in traceback.extract_tb(tb):
      basename = os.path.basename(filename)
      own = (home_dir in filename) or ("/" not in filename)

      print(colorize("\"" + basename + '"', "green", own) + " in " + func)
      print("%s:  %s" % (
          colorize("%5d" % line_num, "red", own),
          colorize(text, "yellow", own)))
      print("  %s" % colorize(filename, "black", own))

    print(colorize("%s: %s" % (type_.__name__, value), "cyan"))
  return hook

def generateConfigGL(outputFile, w, h, planes,namelist, subplane, f, px, py):
  print("Generating WebGL viewer")
  fo = open(outputFile, "w")

  replacer = {}
  replacer["WIDTH"] = w;
  replacer["HEIGHT"] = h;
  replacer["PLANES"] = "[" + ",".join([str(x) for x in planes]) + "]"
  #replacer["nPLANES"] = len(planes);
  replacer["nSUBPLANES"] = subplane;
  replacer["F"] = f
  replacer["NAMES"] = namelist#"[\"\"]"
  replacer["PX"] = px
  replacer["PY"] = py


  
  st = """
  const w = {WIDTH};
  const h = {HEIGHT};
  const nSubPlanes = {nSUBPLANES};

  const planes = {PLANES};
  const f = {F};
  var names = {NAMES};

  const py = {PY};
  const px = {PX};
  const invz = 0;
  """
  for k in replacer:
      st = st.replace("{" + k + "}", str(replacer[k]))

  fo.write(st + '\n')
  fo.close()

def _blur2d(x, f=[1,2,1], normalize=True, flip=False, stride=1):
    assert x.shape.ndims == 4 and all(dim.value is not None for dim in x.shape[1:])
    assert isinstance(stride, int) and stride >= 1

    # Finalize filter kernel.
    f = np.array(f, dtype=np.float32)
    if f.ndim == 1:
        f = f[:, np.newaxis] * f[np.newaxis, :]
    assert f.ndim == 2
    if normalize:
        f /= np.sum(f)
    if flip:
        f = f[::-1, ::-1]
    #f = f[:, :, np.newaxis, np.newaxis]
    f = f[:, :, np.newaxis, np.newaxis]
    #f = np.tile(f, [1, 1, int(x.shape[1]), 1])
    f = np.tile(f, [1, 1, int(x.shape[3]), 1])

    # No-op => early exit.
    if f.shape == (1, 1) and f[0,0] == 1:
        return x

    # Convolve using depthwise_conv2d.
    orig_dtype = x.dtype
    x = tf.cast(x, tf.float32)  # tf.nn.depthwise_conv2d() doesn't support fp16
    f = tf.constant(f, dtype=x.dtype, name='filter')
    strides = [1, stride, stride, 1]
    x = tf.nn.depthwise_conv2d(x, f, strides=strides, padding='SAME', data_format=None)
    x = tf.cast(x, orig_dtype)
    return x

def _upscale2d(x, factor=2, gain=1):
    assert x.shape.ndims == 4 and all(dim.value is not None for dim in x.shape[1:])
    assert isinstance(factor, int) and factor >= 1

    # Apply gain.
    if gain != 1:
        x *= gain

    # No-op => early exit.
    if factor == 1:
        return x

    # Upscale using tf.tile().
    s = x.shape
    x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
    x = tf.tile(x, [1, 1, factor, 1, factor, 1])
    x = tf.reshape(x, [-1, s[1]* factor, s[2] * factor, s[3] ])
    return x

def _downscale2d(x, factor=2, gain=1):
    assert x.shape.ndims == 4 and all(dim.value is not None for dim in x.shape[1:])
    assert isinstance(factor, int) and factor >= 1

    # 2x2, float32 => downscale using _blur2d().
    if factor == 2 and x.dtype == tf.float32:
        f = [np.sqrt(gain) / factor] * factor
        return _blur2d(x, f=f, normalize=False, stride=factor)

    # Apply gain.
    if gain != 1:
        x *= gain

    # No-op => early exit.
    if factor == 1:
        return x

    # Large factor => downscale using tf.nn.avg_pool().
    # NOTE: Requires tf_config['graph_options.place_pruned_graph']=True to work.
    ksize = [1, 1, factor, factor]
    return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID')

def blur2d(x, f=[1,2,1], normalize=True):
    with tf.variable_scope('Blur2D'):
        @tf.custom_gradient
        def func(x):
            y = _blur2d(x, f, normalize)
            @tf.custom_gradient
            def grad(dy):
                dx = _blur2d(dy, f, normalize, flip=True)
                return dx, lambda ddx: _blur2d(ddx, f, normalize)
            return y, grad
        return func(x)

def upscale2d(x, factor=2):
    with tf.variable_scope('Upscale2D'):
        @tf.custom_gradient
        def func(x):
            y = _upscale2d(x, factor)
            @tf.custom_gradient
            def grad(dy):
                dx = _downscale2d(dy, factor, gain=factor**2)
                return dx, lambda ddx: _upscale2d(ddx, factor)
            return y, grad
        return func(x)

def downscale2d(x, factor=2):
    with tf.variable_scope('Downscale2D'):
        @tf.custom_gradient
        def func(x):
            y = _downscale2d(x, factor)
            @tf.custom_gradient
            def grad(dy):
                dx = _upscale2d(dy, factor, gain=1/factor**2)
                return dx, lambda ddx: _downscale2d(ddx, factor)
            return y, grad
        return func(x)

def conv2d(x,k,filter=3,is_norm=False,stride=[1,1,1,1],name=''):
  w_int = tf.random.truncated_normal([filter, filter, x.shape[3].value, k], stddev=0.1)
  #w = tf.Variable(w_int,name=name+"_w")
  w = tf.compat.v1.get_variable(name+"_w",initializer=w_int, trainable=True)
  out = tf.nn.conv2d(x,w,strides=stride,padding='SAME')
  tf.compat.v1.add_to_collection("checkpoints",out)
  return lay_norm(out,is_norm,name)

def lay_norm(x,is_norm=False,name=''):
  if is_norm:
    a = tf.get_variable(name+"scale",shape=[x.shape[-1]],initializer=tf.initializers.random_normal())
    a = tf.reshape(tf.cast(a,x.dtype),[1,1,1,-1])
    b = tf.compat.v1.get_variable(name+"bias",shape=[x.shape[-1]],initializer=tf.initializers.zeros())
    b = tf.reshape(tf.cast(b,x.dtype),[1,1,1,-1])
    z = x - tf.reduce_mean(x,axis=[1,2],keepdims=True)
    z *= tf.math.rsqrt(tf.reduce_mean(tf.square(z),axis=[1,2],keepdims=True))
    z = a * z + b
  else:
    b = tf.compat.v1.get_variable(name+"bias",shape=[x.shape[-1]],initializer=tf.initializers.zeros())
    b = tf.reshape(tf.cast(b,x.dtype),[1,1,1,-1])
    z = x + b
  return z

def lrelu(x,a = 0.1):
   return tf.math.maximum(x,a*x)

def lerp_clip(a, b, t): return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)




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

def getPlanes(sfm,invz=True):
    dmax = sfm.dmax
    dmin = sfm.dmin
    num_mpi =sfm.num_mpi
    if False:
      return (dmax-dmin)*np.linspace(0, 1, num_mpi)**4+dmin
    elif invz:
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
    cxy = tf.expand_dims(newCoords[:,:,:2]/newCoords[:,:,2:3],0)+ sfm.offset
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
    cxy = tf.expand_dims(newCoords[:,:,:2]/newCoords[:,:,2:3],0)- sfm.offset
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
    cxy = tf.expand_dims(newCoords[:,:,:2]/newCoords[:,:,2:3],0)+ sfm.offset 
    cxys.append(cxy)
  warp = tf.concat(cxys,0)
    
  samples = tf.contrib.resampler.resampler(tf.ones([2, sfm.nh, sfm.nw, 1]),warp)
  mask = samples[0:1]*samples[1:2]

  return mask*0.99 + (1-mask)*0.01

  def inv_ref(sfm):
    iwarp = InvzHomoWarp(sfm, sfm.features, sfm.features)
    img_tile = cv2.resize(sfm.ref_img,(int(sfm.ow*FLAGS.scale),int(sfm.oh*FLAGS.scale)))
    img_tile = tf.expand_dims(img_tile,0)
    img_tile = -tf.math.log(tf.maximum(1/img_tile-1,0.006))
    img_tile = tf.tile(img_tile,[num_mpi*sub_sam,1,1,1])
    elem = tf.contrib.resampler.resampler(img_tile,iwarp)
    return elem