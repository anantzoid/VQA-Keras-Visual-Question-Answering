import numpy as np
from keras.utils.np_utils import to_categorical
import json
import h5py

def read_data():
    print "Reading Data..."
    img_data = h5py.File('data/data_img.h5')
    ques_data = h5py.File('data/data_prepro.h5')
  
    img_data = np.array(img_data['images_train'])
    train_X = [img_data, ques_data['ques_train']]
    train_y = to_categorical(ques_data['answers'])

    return train_X, train_y

def get_word_index():
    meta_data = json.load(open('data/data_prepro.json', 'r'))
    word_index = meta_data['ix_to_word']
    return word_index

