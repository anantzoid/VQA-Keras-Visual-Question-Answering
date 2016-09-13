import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Merge, Flatten, Embedding
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from scipy.misc import imread, imresize
import json
import os
import pickle
from models import *

train_img_path = 'data/train2014'
train_annotations_path = 'data/mscoco_train2014_annotations.json'
train_questions_path = 'data/Questions_Train_mscoco/MultipleChoice_mscoco_train2014_questions.json'
vgg_weights_path = 'data/vgg16_weights.h5'
glove_path = 'data/glove.6B.100d.txt'
data_file = 'data/processed_data.pkl'
SEQ_LENGTH = 23
EMBEDDING_DIM = 100
WORD_LIMIT = 10000

def prepare_image(img_path):
    im = imresize(imread(img_path), (224, 224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    return im

def prepare_embeddings(embeddings_index, texts):
    print "Embedding Data..."
    tokenizer = Tokenizer(nb_words=WORD_LIMIT)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    word_index = tokenizer.word_index
    NUM_WORDS = len(word_index)+1
    
    embedding_matrix = np.zeros((NUM_WORDS, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    return tokenizer, embedding_matrix, NUM_WORDS


def prepare_text(tokenizer, text):
    tokenizer.fit_on_texts(text)
    t_sequences = tokenizer.texts_to_sequences(text)
    return pad_sequences(t_sequences, maxlen=SEQ_LENGTH)


def read_data():
    if os.path.exists(data_file):
        with open(data_file, 'rb') as d_file:
            data = pickle.load(d_file) 
    else:
        data = {}

    with open(train_annotations_path, 'r') as an_file:
        annotations = json.loads(an_file.read())

    with open(train_questions_path, 'r') as qs_file:
        questions = json.loads(qs_file.read())    

    print "Making data files..."
    classes = []
    for question in questions['questions']:
        key = str(question['question_id'])
        if key not in data:
            image_id = str(question['image_id'])
            image_file = 'COCO_train2014_'+('0'*(12-len(image_id)))+image_id+'.jpg'
            image_file = os.path.join(train_img_path, image_file)
            if not os.path.exists(image_file):
                continue
            row = {}
            row['question'] = str(question['question'])
            row['image'] = prepare_image(image_file)
            row['answer'] = [_ for _ in annotations['annotations'] if _['question_id']==question['question_id']][0]['multiple_choice_answer']
            data[key] = row
        classes.append(data[key]['answer'])

    classes = list(set(classes))

    data_pickle_file = open(data_file, 'wb')
    pickle.dump(data, data_pickle_file)
    data_pickle_file.close()

    return data.values(), classes

def prepare_data(data, tokenizer, classes):

    im = [_['image'] for _ in data]
    im = np.array([_[0] for _ in im])
    embeddings = prepare_text(tokenizer, [_['question'] for _ in data])
    targets = [_['answer'] for _ in data]

    index_labels = to_categorical(range(len(classes)))
    target_labels = np.array([index_labels[classes.index(_)] for _ in targets])
    return im, embeddings, target_labels

def create_model(embedding_matrix, NUM_WORDS, target_shape):
    vgg_model = VGG_16(vgg_weights_path)
    lstm_model = Word2VecModel(embedding_matrix, NUM_WORDS, EMBEDDING_DIM, SEQ_LENGTH)
    print "Merging Final Model..."
    fc_model = Sequential()
    fc_model.add(Merge([vgg_model, lstm_model], mode='concat'))
    fc_model.add(Activation('relu'))
    fc_model.add(Dropout(0.5))
    fc_model.add(Dense(1024, activation='relu'))
    fc_model.add(Dropout(0.5))
    fc_model.add(Dense(target_shape, activation='softmax'))
    fc_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
         metrics=['accuracy'])
    return fc_model

def main():
    [data, classes] = read_data()
    exit(0)
    print "Loading Embeddings..."
    embeddings_index = {}
    with open(glove_path, 'r') as glove_file:
        for line in glove_file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    [data, classes] = read_data()
    [tokenizer, embedding_matrix, NUM_WORDS] = prepare_embeddings(embeddings_index, [_['question'] for _ in data])
    [im, embeddings, target_labels] = prepare_data(data, tokenizer, classes)


    model = create_model(embedding_matrix, NUM_WORDS, target_labels.shape[1])
    
    model.fit([im, np.array(embeddings)], target_labels, 
        nb_epoch=100, batch_size=64)


if __name__ == "__main__":
    main()
