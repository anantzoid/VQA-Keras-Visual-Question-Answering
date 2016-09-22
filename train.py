import numpy as np
from keras.models import model_from_json#load_model
from keras.callbacks import ModelCheckpoint
import os
from models import *
from prepare_data import *

model_filename = "data/ckpts/model.h5"
model_weights_filename = "data/ckpts/model_weights.h5"
num_classes = 1000
SEQ_LENGTH = 26
EMBEDDING_DIM = 300
dropout_rate = 0.5
metadata = get_metadata()
NUM_WORDS = len(metadata['ix_to_word'].keys())

def get_model():
    '''
    if os.path.exists(model_filename):
        print "Loading Model..."
        with open(model_filename, 'r') as json_file:
            loaded_model_json = json_file.read()
        fc_model = model_from_json(loaded_model_json)
        fc_model.load_weights(model_weights_filename)
    else:
    '''
    print "Creating Model..."
    embedding_matrix = prepare_embeddings(NUM_WORDS, EMBEDDING_DIM, metadata)
    fc_model = vqa_model(embedding_matrix, NUM_WORDS, EMBEDDING_DIM, SEQ_LENGTH, dropout_rate, num_classes)
    if os.path.exists(model_weights_filename):
        print "Loading Weights..."
        fc_model.load_weights(model_weights_filename)

    return fc_model

def main():

    train_X, train_y = read_data()    
    model = get_model()
    checkpointer = ModelCheckpoint(filepath=model_weights_filename,verbose=1)
    model.fit(train_X, train_y, nb_epoch=10, batch_size=128, callbacks=[checkpointer], shuffle="batch")
    model_json = model.to_json()
    with open(model_filename, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(model_weights_filename, overwrite=True)


if __name__ == "__main__":
    main()
