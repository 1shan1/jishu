import os
import torch
import skvideo
import skvideo.io
import numpy as np
import cv2
from src.crowd_count import CrowdCounter
from src import network
from src import utils

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
vis = False
save_output = True

model_path = '/home/shanzhen/try_video/final_models/mcnn_shtechA_68.h5'
net = CrowdCounter()
      
trained_model = os.path.join(model_path)
network.load_net(trained_model, net)
net.cuda()
net.eval()


et_count_list = []
video_list =[]
model_path = '/home/shanzhen/crowdcount-mcnn/final_models/mcnn_shtechA_68.h5'
videogen = skvideo.io.vread('/home/shanzhen/try_video/TrackingPeople.mp4')
for frame in videogen:
  
  frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  frame = frame.astype(np.float32, copy=False)

  ht = frame.shape[0]
  wd = frame.shape[1]
  ht_1 = (ht/4)*4
  wd_1 = (wd/4)*4
  frame = cv2.resize(frame,(wd_1,ht_1))
  frame = frame.reshape((1,1,frame.shape[0],frame.shape[1]))
  density_map = net(frame)
  density_map = density_map.data.cpu().numpy()
  
  
  
  et_count = np.sum(density_map)
  et_count_list.append(et_count)
  density_map = 255*density_map/np.max(density_map)
  density_map= density_map[0][0]
  video_list.append(density_map)


print(et_count_list)
np_video_array = np.array(video_list)
#print(np_video_array)
#print(np_video_array.shape)
outputdata = np_video_array.astype(np.uint8)
skvideo.io.vwrite("output_TrackingPeople.mp4", outputdata)
  #frame = frame.reshape((1,1,frame.shape[0],frame.shape[1],frame.shape[2]))
  #print(frame)  
  #print(frame.shape)
  #exit()   
#print(videodata.shape)

