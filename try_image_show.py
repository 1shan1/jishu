import os
import torch
import numpy as np
from PIL import Image
import os
import cv2
from src.crowd_count import CrowdCounter
from src import network
from src import utils


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
vis = False
save_output = True

data_path = '/home/pengshanzhen/try_video/count_crowd/IMG'

model_path = '/home/pengshanzhen/try_video/final_models/mcnn_shtechB_156.h5'
output_dir = '/home/pengshanzhen/try_video/count_crowd/cout'




net = CrowdCounter()
      
trained_model = os.path.join(model_path)
network.load_net(trained_model, net)
net.cuda()
net.eval()

datafiles = os.listdir(data_path)
for fname in datafiles:
  
  img = cv2.imread(os.path.join(data_path,fname),0)
  
  img = img.astype(np.float32, copy=False)
  ht = img.shape[0]
  wd = img.shape[1]
  ht_1 = (ht/4)*4
  wd_1 = (wd/4)*4
  img = cv2.resize(img,(wd_1,ht_1))
  img1 = cv2.resize(img, ((wd_1/4),(ht_1/4)), interpolation=cv2.INTER_CUBIC)
  #print(frame1)
  #print(frame1.shape)
  #exit()
  img = img.reshape((1,1,img.shape[0],img.shape[1]))
  density_map = net(img)
  density_map = density_map.data.cpu().numpy()
  
  
  
  et_count = np.sum(density_map)
  
  density_map = 255*density_map/np.max(density_map)
  density_map= density_map[0][0]
  #print(density_map)
  #print(density_map.shape)
  #exit()
  
  density_map = cv2.addWeighted(img1, 0.8, density_map, 0.2, 0)
  font=cv2.FONT_HERSHEY_SIMPLEX  
  
  cv2.putText(density_map,str(et_count),(10,150),font,1,(255,0,0),3) 
  
  
  if vis:
        utils.display_results(im_data, gt_data, density_map)
  if save_output:
        cv2.imwrite(os.path.join(output_dir,fname),density_map)

