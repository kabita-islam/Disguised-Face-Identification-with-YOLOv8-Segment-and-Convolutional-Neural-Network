# -*- coding: utf-8 -*-
"""
Created on Wed May 24 01:10:51 2023

@author: Plabon Dibra
"""

from ultralytics import YOLO
from matplotlib import pyplot as plt
import cv2
import numpy as np
from glob import glob

model = YOLO("D:/CE18060/Research_Done/YOLOv8-Segment/YOLODataset/runs/segment/train/weights/best.pt")
folder = "D:/CE18060/Research_Done/YOLOv8-Segment/YOLODataset/images/test"

image_files = glob(folder + '/*.png')

print(len(image_files))
#src="D:/yolo-seg/IMG20230517204158.jpg"


sm = 0
instance = 0
for src in image_files:    
    results = model.predict(source=src)
    image = cv2.imread(src)
    
    res_plotted = results[0].plot()
    plt.title("Input Image")
    plt.imshow(image[:, :, ::-1])
    plt.show()
    
    
    
    res_plotted = results[0].plot()
    plt.title("YOLO Prediction")
    plt.imshow(res_plotted[:, :, ::-1])
    plt.show()
    
    
    
    data = results[0].boxes.boxes.tolist()
    indx = -1
    coords = []
    for mask in results[0].masks.xy:
        indx +=1
        #print(data[indx])
        
        if data[indx][4]>.5:
            sm=sm+data[indx][4]
            instance +=1
            
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
            
            
                
    #print(coords)
    
    plt.title("Only Face")
    plt.imshow(image[:, :, ::-1]) 
    plt.show()
    
    for coord in coords:
        crop = image[coord[1]:coord[1]+coord[3], coord[0]:coord[0]+coord[2]]
        plt.title("Cropped Face")
        plt.imshow(crop[:, :, ::-1])
        plt.show()
    
print("Test Accuracy: ",sm/instance * 100,"%")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

























