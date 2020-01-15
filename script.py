import os


codename = "make_mpi.py"
#codename = "model_singleImg.py"
#codename = "model_depest.py"
#codename ="gen_tfrecord.py"
dataset = "scene_010"
ref_img = "03-cam_06"

input = "ccc"

#command = "python "+codename+" -dataset="+dataset+" -ref_img="+ref_img#+" -output="+input
#print(command)
#os.system(command)

for i in [3]:
  for j in [8]:
    dataset = "scene_%03d"%(j)
    ref_img = "%02d-cam_06"%(i)
    command = "python "+codename+" -dataset="+dataset+" -ref_img="+ref_img#+" -input="+input
    command += " -layers=40 -scale=1.0 -offset=16"
    #command += " -restart"
    #command += " -predict"
    print(command)
    #os.system(command+ " -restart")
    os.system(command+ " -predict")