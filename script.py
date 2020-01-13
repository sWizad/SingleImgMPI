import os


#codename = "make_mpi.py"
codename = "model_singleImg.py"
dataset = "scene_009"
ref_img = "05-cam_06"

input = "ccc"

#command = "python gen_tfrecord.py -dataset="+dataset+" -ref_img="+ref_img#+" -output="+input
#print(command)
#os.system(command)

for i in [3]:
    ref_img = "%02d-cam_06"%(i)
    command = "python "+codename+" -dataset="+dataset+" -ref_img="+ref_img#+" -input="+input
    command += " -layers=20 -scale=0.75"
    #command += " -restart"
    command += " -predict"
    print(command)
    os.system(command)