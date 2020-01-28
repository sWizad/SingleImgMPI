import os

codename = "model_op.py"
#codename = "make_mpi.py"
#codename = "model_singleImg.py"
#codename = "model_depest.py"
#codename ="gen_tfrecord.py"

input = "ccc"
dataset = "scene_033"
ref_img = "00-cam_06"

if "63" in dataset:
  ref_img = "09-cam_06"
elif "56" in dataset:
  ref_img = "00-cam_06"
elif "52" in dataset:
  ref_img = "00-cam_06"
elif "23" in dataset:
  ref_img = "00-cam_06"
elif "10" in dataset:
  ref_img = "00-cam_06"
#else:
#  print("dataset not found")
#  exit()

if 0:
  command = "python gen_tfrecord.py -dataset="+dataset+" -ref_img="+ref_img#+" -output="+input
  command += " -new_height=1200 -new_width=2000"
  print(command)
  os.system(command)
  exit()
elif 1:
  command = "python "+codename+" -dataset="+dataset+" -ref_img="+ref_img#+" -input="+input
  command += " -layers=24 -scale=0.3 -offset=16"
  #command += " -layers=12 -scale=0.8 -offset=0 -epoch=20000"
  print(command)
  #os.system(command)
  #os.system(command+ " -restart")
  os.system(command+ " -predict")
  exit()









for i in [9]:
  for j in [63]:
    #dataset = "scene_%03d"%(j)
    #ref_img = "%02d-cam_06"%(i)
    command = "python "+codename+" -dataset="+dataset+" -ref_img="+ref_img#+" -input="+input
    command += " -layers=10 -scale=1.0 -offset=16"
    #command += " -restart"
    #command += " -predict"
    print(command)
    os.system(command+ " -restart")
    #os.system(command+ " -predict")
