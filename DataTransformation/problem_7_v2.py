
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 16:43:26 2021

@author: wshi
"""

import cv2
import torchvision.transforms.functional as TF
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import os


class FaceGndTruthDataset:
    """ Ground Truth dataset"""
    def __init__(self, csv_file, root_dir):
        """
        Args :
            csv_file ( string ): Path to the csv file with annotations .
            root_dir ( string ): Directory with all the images 
        """
        self.gnd_truth_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        
        
    def __len__(self):
        return len(self.landmarks_frame)
   
    
    def __getitem__(self, idx):
        """
        Return the image sample and its ground truth
        """
        img_name = os.path.join(self.root_dir, self.gnd_truth_frame.iloc[idx, 0])
        
        bgr_img = cv2.imread(img_name)
        rgb_img  = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        
        x = self.gnd_truth_frame.iloc[idx, 2]
        y = self.gnd_truth_frame.iloc[idx, 3]
        w = self.gnd_truth_frame.iloc[idx, 4]
        h = self.gnd_truth_frame.iloc[idx, 5]
        
        ori_gt = [(x, y, w, h)]
        pil_img  = Image.fromarray(rgb_img)
 
        return pil_img, ori_gt



class RandomAffineOccluded:
    
    def __init__(self, img, ori_gt):
                
        self.img           = img
        self.img_ori       = img.copy()
        self.col, self.row = self.img.size
        self.ori_gt        = ori_gt
        
        """ Set parameters for randomized affine transformations """
        self.theta = random.uniform(-45, 45)     # rotate
        self.scale = random.uniform(0.8, 1.2)    # rescale
        self.tx    = random.uniform(-100,100)    # translate
        self.ty    = random.uniform(-100, 100)   # translate
        self.m1    = random.uniform(-45, 45)     # shear
        self.m2    = random.uniform(-45, 45)     # shear
    
    
        self.all_param = [round(self.theta, 2),
                          round(self.scale, 2),
                          (round(self.tx, 2), round(self.ty, 2)),
                          (round(self.m1, 2), round(self.m2, 2))]
        
        
        self.all_param_labels = ['Flip',
                                 'Theta',
                                 'Scale',
                                 'Translate',
                                 'Shear',
                                 '(i, j, MaxRow, MaxCol)']
    
    
               
    def affine_img_mask(self):
        """ 
            This step is aimed to avoid cropping image with 
            black regions
        """
        
        
        """  Generate a mask with the same size of original image """
        mask     = np.ones((self.row, self.col))
        pil_mask = Image.fromarray(mask)
        
        
        """ Do the same transformations"""
        if self.all_param[0] == 'True':
            mask_hflip = TF.hflip(pil_mask)
        else:
            mask_hflip = pil_mask
        
        mask_affine = TF.affine(mask_hflip, 
                                self.all_param[1], 
                                self.all_param[3], 
                                self.all_param[2],
                                self.all_param[4])
            
        np_mask = np.uint8(mask_affine)
        
        """ Select random rectangular regions without black parts """
        while(1):
            occ_i     = int(random.uniform(0, 0.8) * self.row)
            occ_j     = int(random.uniform(0, 0.8) * self.col)
            occ_h     = int(random.uniform(0.2, 0.5) * self.row)
            occ_w     = int(random.uniform(0.2, 0.5) * self.col)
        
            MaxRow = min(self.row - 1, occ_i + occ_h)
            MaxCol = min(self.col - 1, occ_j + occ_w)
        
            mask_crop = np_mask[occ_i:MaxRow, occ_j:MaxCol]
            mask_crop_pixel = (MaxRow-occ_i)*(MaxCol-occ_j)

            mask_crop_sum = sum(sum(mask_crop))

            
            if (mask_crop_sum == mask_crop_pixel):
                break
        
        return occ_i, occ_j, MaxRow, MaxCol
      

      
    def get_new_coor(self, x1, y1, x2, y2, x3, y3, x4, y4):
        """
         Generate new bounding box based on coordinates
         of four new points
        """
        x_min = min(x1, x2, x3, x4)
        y_min = min(y1, y2, y3, y4)
        x_max = max(x1, x2, x3, x4)
        y_max = max(y1, y2, y3, y4)
        
        return int(x_min), int(y_min), int(x_max-x_min), int(y_max-y_min)
    
    
    def flip_gt(self, x, y, w, h):
        """
            Get four new points after flipping
        """
        if self.all_param[0] == 'True':
            x1_new, y1_new = self.col - x - w, y
            x2_new, y2_new = self.col - x,     y
            x3_new, y3_new = self.col - x - w, y + h
            x4_new, y4_new = self.col - x,     y + h
            
        else:
            x1_new, y1_new = x,     y
            x2_new, y2_new = x + w, y
            x3_new, y3_new = x,     y + h
            x4_new, y4_new = x + w, y + h
        
        new_bbox_coor = [x1_new, y1_new,
                         x2_new, y2_new,
                         x3_new, y3_new,
                         x4_new, y4_new]
        new_bbox_coor = [int(i) for i in new_bbox_coor]
        
 
        return new_bbox_coor
    
      
    def rotate_point(self, x, y):
        cen_x, cen_y = self.col/2, self.row/2
        x_new = (x-cen_x) * math.cos(math.radians(self.theta)) - (y-cen_y) * math.sin(math.radians(self.theta))+cen_x
        y_new = (x-cen_x) * math.sin(math.radians(self.theta)) + (y-cen_y) * math.cos(math.radians(self.theta))+cen_y
    
        return x_new, y_new
    
    
    def rotate_gt(self, bbox_coor):   
        """
            Get four new points after rotation
        """
        x1, y1 = bbox_coor[0], bbox_coor[1]
        x2, y2 = bbox_coor[2], bbox_coor[3]
        x3, y3 = bbox_coor[4], bbox_coor[5]
        x4, y4 = bbox_coor[6], bbox_coor[7]
        
        
        x1_new, y1_new = self.rotate_point(x1, y1)
        x2_new, y2_new = self.rotate_point(x2, y2)
        x3_new, y3_new = self.rotate_point(x3, y3)
        x4_new, y4_new = self.rotate_point(x4, y4)
        
        new_bbox_coor = [x1_new, y1_new,
                         x2_new, y2_new,
                         x3_new, y3_new,
                         x4_new, y4_new]
        
        new_bbox_coor = [int(i) for i in new_bbox_coor]
        
        
        return new_bbox_coor

        
                          
    def scale_gt(self, bbox_coor):  
        """
            Get four new points after rescaling
        """
        x1, y1 = bbox_coor[0], bbox_coor[1]
        x2, y2 = bbox_coor[2], bbox_coor[3]
        x3, y3 = bbox_coor[4], bbox_coor[5]
        x4, y4 = bbox_coor[6], bbox_coor[7]
        
        cen_x, cen_y = self.col/2, self.row/2
        
        x1_new, y1_new = (x1-cen_x) * self.scale + cen_x, (y1-cen_y) * self.scale + cen_y                        
        x2_new, y2_new = (x2-cen_x) * self.scale + cen_x, (y2-cen_y) * self.scale + cen_y 
        x3_new, y3_new = (x3-cen_x) * self.scale + cen_x, (y3-cen_y) * self.scale + cen_y  
        x4_new, y4_new = (x4-cen_x) * self.scale + cen_x, (y4-cen_y) * self.scale + cen_y
        
       
        new_bbox_coor = [x1_new, y1_new,
                         x2_new, y2_new,
                         x3_new, y3_new,
                         x4_new, y4_new]
        new_bbox_coor = [int(i) for i in new_bbox_coor]
        
        
        return new_bbox_coor     
    
    
    def translate_gt(self, bbox_coor):
        """
            Get four new points after translating
        """
        x1, y1 = bbox_coor[0], bbox_coor[1]
        x2, y2 = bbox_coor[2], bbox_coor[3]
        x3, y3 = bbox_coor[4], bbox_coor[5]
        x4, y4 = bbox_coor[6], bbox_coor[7]
        
        x1_new, y1_new  = x1 + self.tx, y1 + self.ty
        x2_new, y2_new  = x2 + self.tx, y2 + self.ty
        x3_new, y3_new  = x3 + self.tx, y3 + self.ty
        x4_new, y4_new  = x4 + self.tx, y4 + self.ty
        
        
        return self.get_new_coor(x1_new, y1_new, 
                                 x2_new, y2_new, 
                                 x3_new, y3_new, 
                                 x4_new, y4_new)  
        


    def compare_X_cenX(self, x, y):
        """
            Get new y for shearing
        """
        cen_x = self.col/2
        y_new = y + (cen_x-x)*math.tan(math.radians(self.m2))
       
        return y_new    
    
    
    def compare_Y_cenY(self, x, y):
        """
            Get new x for shearing
        """
        cen_y = self.row/2
        x_new = x + (cen_y-y)*math.tan(math.radians(self.m1)) 
        
        return x_new    
            
    
    def shear_gt(self, bbox_coor):
        """
            Get four new points after shearing
        """
        x1, y1 = bbox_coor[0], bbox_coor[1]
        x2, y2 = bbox_coor[2], bbox_coor[3]
        x3, y3 = bbox_coor[4], bbox_coor[5]
        x4, y4 = bbox_coor[6], bbox_coor[7]
        
    
        x1_new_h, y1_new_h = self.compare_Y_cenY(x1, y1), y1
        x2_new_h, y2_new_h = self.compare_Y_cenY(x2, y2), y2
        x3_new_h, y3_new_h = self.compare_Y_cenY(x3, y3), y3
        x4_new_h, y4_new_h = self.compare_Y_cenY(x4, y4), y4
        
        
        x1_new_v, y1_new_v = x1_new_h, self.compare_X_cenX(x1_new_h, y1_new_h)
        x2_new_v, y2_new_v = x2_new_h, self.compare_X_cenX(x2_new_h, y2_new_h)
        x3_new_v, y3_new_v = x3_new_h, self.compare_X_cenX(x3_new_h, y3_new_h)
        x4_new_v, y4_new_v = x4_new_h, self.compare_X_cenX(x4_new_h, y4_new_h)
        
     
        new_bbox_coor = [x1_new_v, y1_new_v,
                         x2_new_v, y2_new_v,
                         x3_new_v, y3_new_v,
                         x4_new_v, y4_new_v]
        new_bbox_coor = [int(i) for i in new_bbox_coor]
        
        
        return new_bbox_coor
    
        


    def gen_new_gt(self):
        """
            Generate new ground truth after doing all transformations
        """
        new_gt = []
        
        for i in range(len(self.ori_gt)):
            ori_coor = self.ori_gt[i]
            x_ori, y_ori, w_ori, h_ori = (ori_coor[0], 
                                          ori_coor[1], 
                                          ori_coor[2],
                                          ori_coor[3])
            
            bbox_coor_flip = self.flip_gt(x_ori, 
                                          y_ori, 
                                          w_ori, 
                                          h_ori)
            
            
            bbox_coor_scale = self.scale_gt(bbox_coor_flip)  
            bbox_coor_shear = self.shear_gt(bbox_coor_scale)
            bbox_coor_rotate= self.rotate_gt(bbox_coor_shear)
            x_shear, y_shear, w_shear, h_shear = self.translate_gt(bbox_coor_rotate)
            new_gt.append((x_shear, y_shear, w_shear, h_shear))
           
    
        return new_gt 
            

        
    def random_flip(self, img):
        """
            Random horizontal flip
        """
        if random.random() > 0.5:
            img_hflip = TF.hflip(img)
            flip = 'True'
        else:
            img_hflip = img
            flip = 'False'
           
        self.all_param.insert(0, flip)
        
        return img_hflip
        
        
    
    def random_affine(self, img):
        
        """
        Random affine transformations
        
        affine(img:       torch.Tensor, 
               angle:     float, 
               translate: List[float], 
               scale:     float, 
               shear:     List[float]
        """
        
        img_affine = TF.affine(img, 
                               self.theta, 
                               (self.tx, self.ty), 
                               self.scale,
                               (self.m1, self.m2))
      
        return img_affine
    
     
    def crop_resize(self, img, occ_i, occ_j, MaxRow, MaxCol, new_gt):
        """
        resized_crop(img:    torch.Tensor, 
                     top:    int, 
                     left:   int, 
                     height: int, 
                     width:  int, 
                     size:   List[int])
        """
        
        """ Generate the mask of whole frame """
         
        crop_img = TF.crop(img,
                           occ_i,
                           occ_j,
                           MaxRow-occ_i, 
                           MaxCol-occ_j)
                                   
      
        new_gt_crop  = []
        
        
        for i in range(len(new_gt)):
            new_gt_coor = new_gt[i]
            new_gt_crop_coor = (new_gt_coor[0]-occ_j,
                                new_gt_coor[1]-occ_i,
                                new_gt_coor[2],
                                new_gt_coor[3])
            new_gt_crop.append(new_gt_crop_coor)
        
                        
        """ Resize cropped image """
        new_size = 224
        crop_resize_img = TF.resize(crop_img, (new_size, new_size))
                                  
        new_gt_crop_resize = []
        for j in range(len(new_gt_crop)):
           new_gt_coor1 = new_gt_crop[j]
           new_gt_crop_resize_coor = (int(new_gt_coor1[0]*new_size/(MaxCol-occ_j)),
                                      int(new_gt_coor1[1]*new_size/(MaxRow-occ_i)),
                                      int(new_gt_coor1[2]*new_size/(MaxCol-occ_j)),
                                      int(new_gt_coor1[3]*new_size/(MaxRow-occ_i)))
           new_gt_crop_resize.append(new_gt_crop_resize_coor)
        
        np_crop_resize_img = np.uint8(TF.to_pil_image(crop_resize_img))
        
        
        """ Add new bounding box in the cropped image with blue color"""
        for k in range(len(new_gt_crop_resize)):
            
            cv2.rectangle(np_crop_resize_img, new_gt_crop_resize[k], (0, 0, 255), 3)
        
        plt.figure()
        plt.imshow(np_crop_resize_img)
        plt.title('Cropped & Resized gt')
        plt.show()
        
        return crop_resize_img
        
    
    def random_occlusion(self, img, img_nogt, new_gt):
        """
        erase(img: torch.Tensor, 
              i:   int, 
              j:   int, 
              h:   int, 
              w:   int, 
              v:   torch.Tensor, 
              inplace: bool = False)
        """
        occ_i, occ_j, MaxRow, MaxCol = self.affine_img_mask()
        
        
        tensor_img        = TF.pil_to_tensor(img)
        tensor_img_nogt   = TF.pil_to_tensor(img_nogt)
        
        img_crop_resize = self.crop_resize(tensor_img_nogt, 
                                           occ_i, 
                                           occ_j,
                                           MaxRow, 
                                           MaxCol,
                                           new_gt) 
        
        img_erase = TF.erase(tensor_img, 
                             occ_i, 
                             occ_j, 
                             MaxRow-occ_i, 
                             MaxCol-occ_j,
                             0)
    
        
        self.all_param.append((occ_i, occ_j, MaxRow, MaxCol)) 
        
        
        return img_erase, img_crop_resize 
    
    
    
    def all_activities(self):
        """
        Apply all transformations
        """
        
        """ Random horizontal flips """
        img_hflip = self.random_flip(self.img)
        
        
        """ Random affine transformation """
        img_affine = self.random_affine(img_hflip)
        
        
        """ Generate new ground truth """
        new_gt = self.gen_new_gt()
        
        img_affine_backup = np.uint8(img_affine.copy())
        img_affine_backup_nogt = np.uint8(img_affine.copy())
        for i in range(len(new_gt)):
            cv2.rectangle(img_affine_backup, new_gt[i], (0, 255, 0), 3)
        
       
        plt.figure()
        plt.title('New gt')
        plt.imshow(img_affine_backup)
        plt.show()
        
                       
        """ Random occlusion """
        img_erase, img_crop_resize = self.random_occlusion(Image.fromarray(img_affine_backup),
                                                           Image.fromarray(img_affine_backup_nogt),
                                                           new_gt)
                     
        plt.figure()
        plt.imshow(img_erase.permute(1, 2, 0))
        plt.title('Applying random occlusion')
        plt.show()
        
        plt.figure()
        plt.imshow(img_crop_resize.permute(1, 2, 0))
        plt.title('Cropped & Resized')
        plt.show()
        
        return self.img_ori, img_erase, img_crop_resize, new_gt
        
   

    def last_sample_params(self):
        """
        Save all parameters of transformations in a dictionary
        """
        random_param = dict(zip(self.all_param_labels, self.all_param))
        return random_param


def ten_crop(pil_img, ori_gt):
    
    affine_ten_img = ()
    random_ten_param = []
    
    """" Add bbox based on original gt """
    np_img_backup = np.array(pil_img.copy())
    
    for i in range(len(ori_gt)):
        cv2.rectangle(np_img_backup, ori_gt[i], (255, 0, 255), 3)
    
    
    plt.figure()
    plt.title('Ori gt')
    plt.imshow(np_img_backup)
    plt.show()
        
    """ Get 10 samples """          
    for j in range(1):
        
        """ Randomized affine transformations """
        RAO = RandomAffineOccluded(pil_img, ori_gt)
        ori_img, affine_img, cropped_img, new_gt = RAO.all_activities()
        random_param = RAO.last_sample_params()       
        print(random_param)
        
        
        """ Add both original and new gt in the same image """
        pil_img_backup = Image.fromarray(np_img_backup)
        if random_param.get('Flip') == 'True':
            pil_img_backup_flip = TF.hflip(pil_img_backup)
        else:
            pil_img_backup_flip = pil_img_backup    
        
        pil_img_backup_affine = TF.affine(pil_img_backup_flip, 
                                          random_param.get('Theta'), 
                                          random_param.get('Translate'), 
                                          random_param.get('Scale'),
                                          random_param.get('Shear'))
        
        np_img_backup_affine = np.uint8(pil_img_backup_affine)
        
        for i in range(len(new_gt)):
            cv2.rectangle(np_img_backup_affine, new_gt[i], (0, 255, 0), 3)
                    
        plt.figure()
        plt.imshow(np_img_backup_affine, cmap='gray')
        plt.title('Both gt')
        plt.show() 
               
        
        """ Save all ten samples and transformation parameters """
        affine_ten_img   = affine_ten_img + (cropped_img,)
        random_ten_param.append(random_param)
        
    return affine_ten_img, random_ten_param


    
def main():
   
    csv_file = './Data/groundtruth.csv'
    root_dir = './Images/'
    
    """ Get sample and ground truth """
    index = 3 # Select a sample for furture processing
    FGTD  = FaceGndTruthDataset(csv_file, root_dir)
    pil_img, ori_gt = FGTD[index] 
    
    """Randomized affine transformations with ten image samples"""
    affine_ten_img, random_ten_param = ten_crop(pil_img, ori_gt)
    

if __name__ == "__main__":
    main()


