import numpy as np
import cv2
from PIL import Image

img = Image.open('/home/shanzhen/try_video/exp_B/et_output_IMG_10.png')
#img.save('/home/shanzhen/try_video/thumbnail.jpg', 'jpeg') 

img = np.array(img)
#img=np.zeros((512,512,3),np.uint8)
  
#print(img)
#print(img.shape)
#exit()
#cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)  
#cv2.ellipse(img,(256,256),(100,50),0,0,360,255,-1)  
#cv2.ellipse(img,(256,256),(100,50),0,0,0,255,-1)  
font=cv2.FONT_HERSHEY_SIMPLEX  
opencv = 181
opencl = 185
cv2.putText(img,'gt'+':'+str(opencv),(30,100),font,1,(255,0,0),3)
cv2.putText(img,'et'+':'+str(opencl),(30,150),font,1,(255,0,0),3) 
#cv2.imshow("shi",img)
#img.save('thumbnail.jpg', 'jpeg') 
cv2.imwrite('/home/shanzhen/try_video/result.png',img) 
cv2.waitKey(0)  
cv2.destroyAllWindows()  

