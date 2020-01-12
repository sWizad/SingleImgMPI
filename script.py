import os


#codename = "make_mpi.py"
codename = "model_singleImg.py"
dataset = "scene_009"
ref_img = "01-cam_06"

input = "ccc"

#command = "python gen_tfrecord.py -dataset="+dataset+" -ref_img="+ref_img#+" -output="+input
#print(command)
#os.system(command)

command = "python "+codename+" -dataset="+dataset+" -ref_img="+ref_img#+" -input="+input
command += " -layers=32"
command += " -restart"
print(command)
os.system(command)