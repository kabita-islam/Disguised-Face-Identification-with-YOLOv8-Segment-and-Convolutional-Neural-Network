# -*- coding: utf-8 -*-
"""
Created on Tue May 16 05:47:22 2023

@author: Plabon Dibra
"""


from ultralytics import YOLO
from matplotlib import pyplot as plt
import cv2
import numpy as np

from ultralytics import YOLO
from matplotlib import pyplot as plt
import cv2
import numpy as np
from glob import glob


yolo_model = YOLO("F:/Thesis_CE18030&60/programs/YOLO/runs/segment/train/weights/best.pt")

from tensorflow import keras
cnn_model = keras.models.load_model(r"F:/Thesis_CE18030&60/resource/CNN/resnet50.h5")



folders  = 'F:/Thesis_CE18030&60/resource/CNN/RawData'
image_files = glob(folders + '/*/*.jpg')

match = 0

#src="F:/Thesis_CE18030&60/resource/CNN/RawData/Aynan/IMG20230515162018.jpg"
for src in image_files:
    image = cv2.imread(src)
    h,w = image.shape[0], image.shape[1]
    
    rh = (h)/(max(h,w))
    rw = (w)/(max(h,w))
    
    image =  cv2.resize(image,(int(rw*800),int(rh*800)))
    plt.title("Input Image")
    plt.imshow(image[:, :, ::-1])
    plt.show()
    
    results = yolo_model.predict(image)
    
    '''
    res_plotted = results[0].plot()
    plt.imshow(res_plotted[:, :, ::-1])
    plt.show()
    '''  

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
            
            if data[indx][5] == 0.0:   
                coords.append([round(data[indx][0]) , round(data[indx][1]), round(data[indx][2])-round(data[indx][0]), round(data[indx][3])-round(data[indx][1])])
                
                tmp = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
                cv2.fillPoly(tmp, [mask], color=(255,255,255))
                
                image = cv2.bitwise_and(image,tmp)
            
            else:   
                cv2.fillPoly(image, [mask], color=(0,0,0))
                
    #print(coords)
    '''
    plt.title("face")
    plt.imshow(image[:, :, ::-1])
    plt.show()
    '''
    to_detect = []
    for coord in coords:
        crop = image[coord[1]:coord[1]+coord[3], coord[0]:coord[0]+coord[2]]
        '''
        plt.title("cropped")
        plt.imshow(crop[:, :, ::-1])
        plt.show()
        '''
        to_detect.append(crop)
    
    

      
    ########################### CNN Predict #############################
    
    
    
    for test_image in to_detect:
        plt.imshow(test_image[:, :, ::-1])
        
        test_image = cv2.resize(test_image,(256,256))
        test_input = test_image.reshape((1,256,256,3))
        
        pred = cnn_model.predict(test_input)
        
        #print(pred)
        ind_max = 0
        
        for i in range(9):
            if pred[0][i]> pred[0][ind_max]:
                ind_max = i
            
        if ind_max==0:
            #print("aynan",pred[0][ind_max] )
            plt.title("aynan")
        elif ind_max==1:
            #print("hasib",pred[0][ind_max])
            plt.title("hasib")
        elif ind_max==2:
            print("mim",pred[0][ind_max])
            plt.title("mim")
        elif ind_max==3:
           # print("mohir",pred[0][ind_max])
            plt.title("mohir")
        elif ind_max==4:
            #print("rifat",pred[0][ind_max])
            plt.title("rifat")
        elif ind_max==5:
           # print("rongon",pred[0][ind_max])
            plt.title("rongon")
        elif ind_max==6:
            #print("sukhi",pred[0][ind_max])
            plt.title("sukhi")
        elif ind_max==7:
            #print("tonni",pred[0][ind_max])
            plt.title("tonni")
        elif ind_max==8:
            # print("tuli",pred[0][ind_max])
             plt.title("tuli")
        else:
            print("Null")
                
        plt.show()
        
print(len(image_files))





















