# -*- coding: utf-8 -*-
"""
Created on Tue May 16 05:47:22 2023

@author: Plabon Dibra
"""

import numpy as np
from glob import glob
from matplotlib import pyplot as plt
import cv2
import os
folders = "F:/Thesis_CE18030&60/resource/CNN/Dataset/test"
image_files = glob(folders + '/*/*.jpg')

########################### CNN Predict #############################
from tensorflow import keras
#new_model = keras.models.load_model(r"F:/Thesis_CE18030&60/resource/CNN/vgg19.h5")
new_model = keras.models.load_model(r"F:/Thesis_CE18030&60/resource/CNN/resnet50.h5")

imgsz = 256
match = 0
for src in image_files:
    #print(src)
    parent_directory = os.path.dirname(src)
    name = os.path.basename(parent_directory)
    
    test_image = cv2.imread(src)
    plt.imshow(test_image)
    
    test_image = cv2.resize(test_image,(imgsz,imgsz))
    test_input = test_image.reshape((1,imgsz,imgsz,3))
    
    pred = new_model.predict(test_input)
    
    #print(pred)
   
    #pred = pred[1]
    ind_max = 0
    
    for i in range(9):
        if pred[0][i]> pred[0][ind_max]:
            ind_max = i
    
    
    target=-1
    
    if name=="aynan":
        target=0
    elif name=="hasib":
        target=1
    elif name=="mim":
        target=2
    elif name=="mohir":
        target=3
    elif name=="rifat":
        target=4
    elif name=="rongon":
        target=5
    elif name=="sukhi":
        target=6
    elif name=="tonni":
        target=7
    elif name=="tuli":
        target=8
    
        
    
    
    if ind_max==target:
        match +=1
        
    
    
    
    if ind_max==0:
       # print("aynan",pred[0][ind_max] )
        plt.title("aynan")
    elif ind_max==1:
        #print("hasib",pred[0][ind_max])
        plt.title("hasib")
    elif ind_max==2:
       # print("mim",pred[0][ind_max])
        plt.title("mim")
    elif ind_max==3:
       # print("mohir",pred[0][ind_max])
        plt.title("mohir")
    elif ind_max==4:
        #print("rifat",pred[0][ind_max])
        plt.title("rifat")
    elif ind_max==5:
        #print("rongon",pred[0][ind_max])
        plt.title("rongon")
    elif ind_max==6:
       # print("sukhi",pred[0][ind_max])
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
    
    
    


print("match: ",match)
print(len(image_files))
print("Accuracy: ",match/len(image_files))   
    
    
    
    
    














