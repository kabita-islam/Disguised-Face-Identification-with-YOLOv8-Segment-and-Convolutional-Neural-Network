# -*- coding: utf-8 -*-
"""
Created on Mon May 22 17:22:17 2023

@author: Plabon Dibra
"""
from ultralytics import YOLO
from matplotlib import pyplot as plt
import cv2
import numpy as np

yolo_model = YOLO("F:/Thesis_CE18030&60/programs/YOLO/runs/segment/train/weights/best.pt")
src="F:/Thesis_CE18030&60/resource/YOLO/YOLO_dataset/YOLODataset/images/test/IMG_20230513_192236_672.png"

image = cv2.imread(src)
plt.imshow(image[:, :, ::-1])
plt.title("Original Image")
plt.show()

results = yolo_model.predict(src)

res_plotted = results[0].plot()
plt.imshow(res_plotted[:, :, ::-1])
plt.title("YOLOv8-Segment Predicted")
plt.show()
                  

data = results[0].boxes.boxes.tolist() 
indx = -1
coords = []
for mask in results[0].masks.xy:
    indx +=1
    #print(data[indx])
    
    if data[indx][4]>.75:
        
         
        mask_list = []
        for k in range(len(mask)):
            mask_list.append([round(mask[k][0]),round(mask[k][1])])
        mask = np.array(mask_list)
        
        if data[indx][5] == 0.0:   #face ID 0.0
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
to_detect = []
for coord in coords:
    crop = image[coord[1]:coord[1]+coord[3], coord[0]:coord[0]+coord[2]]
    
    plt.title("cropped")
    plt.imshow(crop[:, :, ::-1])
    plt.show()
    to_detect.append(crop)











































