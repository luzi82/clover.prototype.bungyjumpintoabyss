import os
import json
import cv2
import numpy as np
from functools import lru_cache
import clover.image_recognition

MODEL_PATH = os.path.join('image_recognition','model','state')
WEIGHT_FILENAME = 'weight.hdf5'
DATA_FILENAME   = 'data.json'

from . import classifier_state_model

WIDTH  = classifier_state_model.WIDTH
HEIGHT = classifier_state_model.HEIGHT

load_img = clover.image_recognition.load_img

def preprocess_img(img):
    img = cv2.resize(img,dsize=(WIDTH,HEIGHT),interpolation=cv2.INTER_AREA)
    img = np.append(img,clover.image_recognition.xy_layer(WIDTH,HEIGHT),axis=2)
    return img

class StateClassifier:

    def __init__(self, model_path):
        weight_path = os.path.join(model_path, WEIGHT_FILENAME)
        data_path   = os.path.join(model_path, DATA_FILENAME)
        with open(data_path,'r') as fin:
            self.data = json.load(fin)
        self.model = classifier_state_model.create_model(len(self.data['label_name_list']))
        self.model.load_weights(weight_path)

    def get_state(self, img):
        img = preprocess_img(img)
        p = self.model.predict(np.expand_dims(img, axis=0))
        score = np.max(p)
        label_idx = np.argmax(p)
        return self.data['label_name_list'][label_idx], score

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='state classifier')
    parser.add_argument('img_file', help="img_file")
    args = parser.parse_args()
    
    img = load_img(args.img_file)

    sc = StateClassifier(MODEL_PATH)

    label, score = sc.get_state(img)
    print('{} {}'.format(label, score))
