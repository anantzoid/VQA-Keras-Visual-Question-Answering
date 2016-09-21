import numpy as np
from keras.utils.np_utils import to_categorical
import json
import h5py

def read_data():
    print "Reading Data..."
    img_data = h5py.File('data/data_img.h5')
    ques_data = h5py.File('data/data_prepro.h5')
  
    img_data = np.array(img_data['images_train'])
    train_img_data = np.array([img_data[_-1,:] for _ in ques_data['img_pos_train']]) 

    train_X = [img_data, ques_data['ques_train']]
    train_y = to_categorical(ques_data['answers'])

    test_img_data = np.array([img_data[_-1,:] for _ in ques_data['img_pos_test']])
    test_X = [img_data, ques_data['ques_test']]
    #test_y = to_categorical(ques_data['answers'])

    return train_X, train_y

def get_metadata():
    meta_data = json.load(open('data/data_prepro.json', 'r'))
    meta_data['ix_to_word'] = {str(word):int(i) for i,word in meta_data['ix_to_word'].items()}
    return meta_data

