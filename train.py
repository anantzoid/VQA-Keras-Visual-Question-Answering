import numpy as np
from keras.models import model_from_json#load_model
from keras.callbacks import ModelCheckpoint
import os
import argparse
from models import *
from prepare_data import *
from constants import *

def get_model(dropout_rate, model_weights_filename):
    print "Creating Model..."
    metadata = get_metadata()
    num_classes = len(metadata['ix_to_ans'].keys())
    num_words = len(metadata['ix_to_word'].keys())

    embedding_matrix = prepare_embeddings(num_words, embedding_dim, metadata)
    model = vqa_model(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes)
    if os.path.exists(model_weights_filename):
        print "Loading Weights..."
        model.load_weights(model_weights_filename)

    return model

def train(args):
    dropout_rate = 0.5
    train_X, train_y = read_data(args.data_limit)    
    model = get_model(dropout_rate, model_weights_filename)
    checkpointer = ModelCheckpoint(filepath=ckpt_model_weights_filename,verbose=1)
    model.fit(train_X, train_y, nb_epoch=args.epoch, batch_size=args.batch_size, callbacks=[checkpointer], shuffle="batch")
    model.save_weights(model_weights_filename, overwrite=True)

def val():
    val_X, val_y, multi_val_y = get_val_data() 
    model = get_model(0.0, model_weights_filename)
    print "Evaluating Accuracy on validation set:"
    metric_vals = model.evaluate(val_X, val_y)
    print ""
    for metric_name, metric_val in zip(model.metrics_names, metric_vals):
        print metric_name, " is ", metric_val

    # Comparing prediction against multiple choice answers
    true_positive = 0
    preds = model.predict(val_X)
    pred_classes = [np.argmax(_) for _ in preds]
    for i, _ in enumerate(pred_classes):
        if _ in multi_val_y[i]:
            true_positive += 1
    print "true positive rate: ", np.float(true_positive)/len(pred_classes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--data_limit', type=int, default=215359, help='Number of data points to fed for training')
    args = parser.parse_args()

    if args.type == 'train':
        train(args)
    elif args.type == 'val':
        val()
