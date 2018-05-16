#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 23:30:29 2018

@author: liushenghui
"""

import xml.etree.ElementTree as ET
from os import getcwd
import os
import pickle
import pdb
with open('class.pkl', 'rb') as f:
    classes = pickle.load(f)
cc = []
def convert_annotation(file):
    in_file = open(file)
    tree=ET.parse(in_file)
    root = tree.getroot()
    for obj in root.iter('subcomponent'):
        cls = obj.find('name').text
        try:
            category = obj.find('category').text
            cls = cls+category
        except:
            pass       
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        if 'shoes' in cls:
            xmlbox = obj
            try:
                b = (int(xmlbox.find('xmin_l').text), int(xmlbox.find('ymin_l').text), int(xmlbox.find('xmax_l').text), int(xmlbox.find('ymax_l').text))
                list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
            except:
#                pdb.set_trace()
                pass
            try:
                b = (int(xmlbox.find('xmin_r').text), int(xmlbox.find('ymin_r').text), int(xmlbox.find('xmax_r').text), int(xmlbox.find('ymax_r').text))
                list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
            except:
                pass
            if cls_id not in cc:
                cc.append(cls_id)
        else:
            if 'bag' in cls:
                xmlbox = obj.find('id_1')
                xmlbox = xmlbox.find('bndbox')
            try:
                b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
                list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
                if cls_id not in cc:
                    cc.append(cls_id)
            except:
                pass
wd = getcwd()

anno_path = 'TRAIN/ANNOTATIONS_TRAIN'
anno_files = os.listdir(anno_path)
anno_files.remove('.DS_Store')
label = []
list_file = open('train.txt', 'w')
for file in anno_files:
    path = anno_path+'/'+file
    list_file.write('TRAIN/IMAGES_TRAIN'+'/'+file.split('.')[0]+'.jpg')
    convert_annotation(path)
    list_file.write('\n')
list_file.close()
#for year, image_set in sets:
#    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
#    list_file = open('%s_%s.txt'%(year, image_set), 'w')
#    for image_id in image_ids:
#        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, year, image_id))
#        convert_annotation(year, image_id, list_file)
#        list_file.write('\n')
#    list_file.close()

