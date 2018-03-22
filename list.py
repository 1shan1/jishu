import os



img_root = '/home/pengshanzhen/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_B_patches_9/train'
label_root = '/home/pengshanzhen/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_B_patches_9/train_den'
#img_root = '/home/pengshanzhen/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_B_patches_9/val'
#label_root = '/home/pengshanzhen/crowdcount-mcnn/data/formatted_trainval/shanghaitech_part_B_patches_9/val_den'



img_list = os.listdir(img_root)
fid = open('partB_train_label.txt', 'w')
for image_name in img_list:
  img_path =os.path.join(img_root ,image_name )
  label_path = os.path.join(label_root ,os.path.splitext(image_name)[0] + '.csv')

  fid.write('%s %s\n'%(img_path, label_path))





