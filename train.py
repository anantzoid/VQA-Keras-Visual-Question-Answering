import numpy as np
from keras.models import Sequential, model_from_json#load_model
from keras.layers import Dense, Activation, Dropout, Merge
from keras.callbacks import ModelCheckpoint
from scipy.misc import imread, imresize
from collections import Counter
import json
import os
import pickle
from models import *
from read_data import read_data, get_word_index

train_questions_path = 'data/Questions_Train_mscoco/MultipleChoice_mscoco_train2014_questions.json'
glove_path = 'data/glove.6B.100d.txt'
num_classes = 1000
SEQ_LENGTH = 26
EMBEDDING_DIM = 100
model_filename = "data/ckpts/model.h5"
model_weights_filename = "data/ckpts/model_weights.h5"
NUM_WORDS = 12603

def prepare_embeddings():
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
    word_index = get_word_index()

    for i, word in word_index.items():
        embedding_vector = embeddings_index.get(str(word))
        if embedding_vector is not None:
            embedding_matrix[int(i)] = embedding_vector
    
    return embedding_matrix

def create_model():
    if os.path.exists(model_filename):
        print "Loading Model..."
        with open(model_filename, 'r') as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(model_weights_filename)
        return model
    else:
        embedding_matrix = prepare_embeddings()
        print "Creating Model..."
        vgg_model = img_model()
        lstm_model = Word2VecModel(embedding_matrix, NUM_WORDS, EMBEDDING_DIM, SEQ_LENGTH)
        print "Merging Final Model..."
        fc_model = Sequential()
        fc_model.add(Merge([vgg_model, lstm_model], mode='mul'))
        fc_model.add(Activation('tanh'))
        fc_model.add(Dropout(0.5))
        fc_model.add(Dense(1000, activation='tanh'))
        fc_model.add(Dropout(0.5))
        fc_model.add(Dense(num_classes, activation='softmax'))
        fc_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
            metrics=['accuracy'])
        return fc_model

def main():

    train_X, train_y = read_data()    
    model = create_model()
    checkpointer = ModelCheckpoint(filepath=model_weights_filename,verbose=1,save_best_only=True)
    model.fit(train_X, train_y, nb_epoch=5, batch_size=64, callbacks=[checkpointer])
    model_json = model.to_json()
    with open(model_filename, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(model_weights_filename)


if __name__ == "__main__":
    main()
