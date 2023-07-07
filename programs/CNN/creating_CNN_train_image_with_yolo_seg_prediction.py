# -*- coding: utf-8 -*-
"""
Created on Tue May 23 00:51:31 2023

@author: Plabon Dibra
"""

from ultralytics import YOLO
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
from glob import glob

model = YOLO("F:/Thesis_CE18030&60/programs/YOLO/runs/segment/train/weights/best.pt")


input_path = 'F:/Thesis_CE18030&60/resource/CNN/RawData'
folders = glob(input_path + '/*/*.jp*g')
#print(len(folders))

ln = len(folders)
#src="D:/yolo-seg/YOLODataset/images/train/IMG_20230513_192409_851.png"
count = 0
for src in folders:
    count +=1
    
    print(count,"/",ln)
    results = model.predict(source=src)
    '''
    res_plotted = results[0].plot()
    plt.imshow(res_plotted[:, :, ::-1])
    plt.show()
    '''
    image = cv2.imread(src)
    
    data = results[0].boxes.boxes.tolist()
    indx = -1
    coords = []
    for mask in results[0].masks.xy:
        indx +=1
        if data[indx][4]>.75:
            
            mask = mask.tolist()
            mask_list = []
            for k in range(len(mask)):
                mask_list.append([round(mask[k][0]),round(mask[k][1])])
            mask = np.array(mask_list)
            
            if data[indx][5] == 0.0:
                
                coords.append([round(data[indx][0]) , round(data[indx][1]), round(data[indx][2])-round(data[indx][0]), round(data[indx][3])-round(data[indx][1])])
                
                tmp = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
                cv2.fillPoly(tmp, [mask], color=(255,255,255))
                
                image = cv2.bitwise_and(image,tmp)
            
            else:   
                cv2.fillPoly(image, [mask], color=(0,0,0))
                
    
    for coord in coords:
        crop = image[coord[1]:coord[1]+coord[3], coord[0]:coord[0]+coord[2]]
        '''
        plt.title("cropped")
        plt.imshow(crop[:, :, ::-1])
        plt.show()
        '''
        file_name = os.path.splitext(os.path.basename(src)) 
        output_path = os.path.dirname(src) + '/train_data/' +file_name[0]+file_name[1]

        resized = cv2.resize(crop, (256, 256))
        status = cv2.imwrite(output_path, resized)
        print(status, output_path)
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    














