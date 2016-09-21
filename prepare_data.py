import numpy as np
from keras.utils.np_utils import to_categorical
import json
import h5py
import os

embedding_matrix_filename = "data/ckpts/embeddings.h5"
glove_path = 'data/glove.6B.100d.txt'
train_questions_path = 'data/Questions_Train_mscoco/MultipleChoice_mscoco_train2014_questions.json'

def right_align(seq,lengths):
    v = np.zeros(np.shape(seq))
    N = np.shape(seq)[1]
    for i in range(np.shape(seq)[0]):
        v[i][N-lengths[i]:N]=seq[i][0:lengths[i]]
    return v

def read_data():
    DATA_LIMIT = 10000
    print "Reading Data..."
    img_data = h5py.File('data/data_img.h5')
    ques_data = h5py.File('data/data_prepro.h5')
  
    img_data = np.array(img_data['images_train'])
    img_pos_train = ques_data['img_pos_train'][:DATA_LIMIT]
    train_img_data = np.array([img_data[_-1,:] for _ in img_pos_train])
    # Normalizing images
    tem = np.sqrt(np.sum(np.multiply(train_img_data, train_img_data), axis=1))
    train_img_data = np.divide(train_img_data, np.transpose(np.tile(tem,(4096,1))))

    #shifting padding to left side
    ques_train = np.array(ques_data['ques_train'])[:DATA_LIMIT, :]
    ques_length_train = np.array(ques_data['ques_length_train'])[:DATA_LIMIT]
    ques_train = right_align(ques_train, ques_length_train)

    train_X = [train_img_data, ques_train]
    train_y = to_categorical(ques_data['answers'])[:DATA_LIMIT, :]

    #test_img_data = np.array([img_data[_-1,:] for _ in ques_data['img_pos_test']])
    #test_X = [img_data, ques_data['ques_test']]
    #test_y = to_categorical(ques_data['answers'])

    return train_X, train_y

def get_metadata():
    meta_data = json.load(open('data/data_prepro.json', 'r'))
    meta_data['ix_to_word'] = {str(word):int(i) for i,word in meta_data['ix_to_word'].items()}
    return meta_data

def prepare_embeddings(NUM_WORDS, EMBEDDING_DIM, metadata):
    if os.path.exists(embedding_matrix_filename):
        with h5py.File(embedding_matrix_filename) as f:
            return np.array(f['embedding_matrix'])

    print "Embedding Data..."
    with open(train_questions_path, 'r') as qs_file:
        questions = json.loads(qs_file.read())
        texts = [str(_['question']) for _ in questions['questions']]
    
    embeddings_index = {}
    with open(glove_path, 'r') as glove_file:
        for line in glove_file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((NUM_WORDS, EMBEDDING_DIM))
    word_index = metadata['ix_to_word']

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
   
    with h5py.File(embedding_matrix_filename, 'w') as f:
        f.create_dataset('embedding_matrix', data=embedding_matrix)

    return embedding_matrix


