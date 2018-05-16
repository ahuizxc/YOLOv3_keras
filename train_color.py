#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 13:03:26 2018

@author: liushenghui
"""

"""
Retrain the YOLO model for your own dataset.
"""
import os

import numpy as np
from PIL import Image
from keras.layers import Input, Lambda, Flatten, Dense, GlobalAveragePooling2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, yolo_loss
from yolo3.utils import letterbox_image
import pdb
import pickle
# Default anchor boxes
YOLO_ANCHORS = np.array(((10,13), (16,30), (33,23), (30,61),
    (62,45), (59,119), (116,90), (156,198), (373,326)))

def _main():
#    annotation_path = '2007_train.txt'
#    data_path = '2007_train.npz'
    output_path = 'model_data/yolo.h5'
    log_dir = 'logs/000/'
    classes_path = 'model_data/voc_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    with open('color_data.pkl','rb') as f:
        dataset = pickle.load(f)
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    input_shape = (128,128) # multiple of 32
    infer_model, model = create_model(input_shape, anchors, len(class_names),
        load_pretrained=True, freeze_body=True)
    train_color_model(infer_model, dataset, log_dir=log_dir)
    model.save(output_path)



def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    if os.path.isfile(anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            return np.array(anchors).reshape(-1, 2)
    else:
        Warning("Could not open anchors file, using default.")
        return YOLO_ANCHORS


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=True):
    '''create the training model'''
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)//3
    y_true = [Input(shape=(h//32, w//32, num_anchors, num_classes+5)),
              Input(shape=(h//16, w//16, num_anchors, num_classes+5)),
              Input(shape=(h//8, w//8, num_anchors, num_classes+5))]

    model_body = yolo_body(image_input, num_anchors, num_classes)

    if load_pretrained:
        weights_path = os.path.join('model_data', 'yolo_weights.h5')
        if not os.path.exists(weights_path):
            print("CREATING WEIGHTS FILE" + weights_path)
            yolo_path = os.path.join('model_data', 'yolo.h5')
            orig_model = load_model(yolo_path, compile=False)
            orig_model.save_weights(weights_path)
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        if freeze_body:
            # Do not freeze 3 output layers.
            for i in range(len(model_body.layers)-3):
                model_body.layers[i].trainable = False

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)
#    pdb.set_trace()
    return model_body, model

def train_color_model(model, dataset, log_dir='logs/'):
    '''retrain/fine-tune the model'''
    data = dataset['data']/255.0
    label = dataset['label']
    num_color = label.shape[1]
#    model.compile(optimizer='adam', loss={
#        # use custom yolo_loss Lambda layer.
#        'yolo_loss': lambda y_true, y_pred: y_pred})
    pdb.set_trace()
    x = model.layers[13].output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    pred_color = Dense(num_color, activation='softmax')(x)
    model_color = Model(inputs=model.input, outputs=pred_color)
    for layer in model.layers:
        layer.trainable = False
    model_color.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])
    model_color.fit(data, label, epochs=10, batch_size=8, validation_split=0.1)
    model_color.save_weights('color_model.h5')
if __name__ == '__main__':
    _main()
