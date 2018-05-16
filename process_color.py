#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 13:12:18 2018

@author: liushenghui
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 

import xml.etree.ElementTree as ET
from os import getcwd
import os
import pickle
import pdb
import cv2
with open('class.pkl', 'rb') as f:
    classes = pickle.load(f)
data = []
label = []
def convert_annotation(file):
    in_file = open(file)
    img = cv2.imread(img_file)
    tree=ET.parse(in_file)
    root = tree.getroot()
    for obj in root.iter('subcomponent'):
        cls = obj.find('name').text
        try:
            color = int(obj.find('color').text)
        except:
            continue
        if 'shoes' in cls:
            xmlbox = obj
            try:
                b = (int(xmlbox.find('xmin_l').text), int(xmlbox.find('ymin_l').text), int(xmlbox.find('xmax_l').text), int(xmlbox.find('ymax_l').text))
                sub_img = cv2.resize(img[b[1]:b[3],b[0]:b[2]],(50,50))
                data.append(sub_img)
                label.append(color)
            except:
                pass
            try:
                b = (int(xmlbox.find('xmin_r').text), int(xmlbox.find('ymin_r').text), int(xmlbox.find('xmax_r').text), int(xmlbox.find('ymax_r').text))
                sub_img = cv2.resize(img[b[1]:b[3],b[0]:b[2]],(50,50))
                data.append(sub_img)
                label.append(color)
            except:
                pass
        else:
            if 'bag' in cls:
                xmlbox = obj.find('id_1')
                xmlbox = xmlbox.find('bndbox')
            try:
                b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
                sub_img = cv2.resize(img[b[1]:b[3],b[0]:b[2]],(50,50))
                data.append(sub_img)
                label.append(color)
            except:
                pass
wd = getcwd()
anno_path = 'TRAIN/ANNOTATIONS_TRAIN'
anno_files = os.listdir(anno_path)
anno_files.remove('.DS_Store')
for file in anno_files:
    path = anno_path+'/'+file
    img_file = 'TRAIN/IMAGES_TRAIN'+'/'+file.split('.')[0]+'.jpg'
    convert_annotation(path)
    
import numpy as np
from keras.utils import to_categorical
import pickle
data = np.array(data)
label = to_categorical(label)
dataset = {'data':data,'label':label}
with open('color_data.pkl', 'wb') as f:
    pickle.dump(dataset,f)
    
#for year, image_set in sets:
#    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
#    list_file = open('%s_%s.txt'%(year, image_set), 'w')
#    for image_id in image_ids:
#        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, year, image_id))
#        convert_annotation(year, image_id, list_file)
#        list_file.write('\n')
#    list_file.close()

