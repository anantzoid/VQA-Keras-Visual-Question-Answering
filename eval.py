import numpy as np
from keras.models import model_from_json
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
import os
from read_data import get_metadata
from models import *

#TODO take from command line as arg
img_path = '/Users/jedi/Downloads/COCO_train2014_000000037062.jpg'
question = 'What color is the truck'

model_filename = "data/ckpts/10/model.h5"
model_weights_filename = "data/ckpts/10/model_weights.h5"
vgg_weights_path = 'data/vgg19_weights.h5'
SEQ_LENGTH = 26

#TODO confirm is same params are used in original vgg model
def prepare_image(img_path):
    im = imresize(imread(img_path), (224, 224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    return im

vgg_model = VGG(vgg_weights_path) 
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
img_vector = model.predict(prepare_image(img_path))[0]

question_vector = []
metadata = get_metadata()
word_index = metadata['ix_to_word']
for word in word_tokenize(question):
    if word in word_index:
        question_vector.append(word_index[word])
    else:
        question_vector.append(0)
question_vector = np.array(pad_sequences(question_vector, maxlen=SEQ_LENGTH))

if not(os.path.exists(model_filename) and os.path.exists(model_weights_filename)):
    print "Model not trained!"

print "Loading Model..."
# TODO define model, not load
with open(model_filename, 'r') as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
model.load_weights(model_weights_filename)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
    metrics=['accuracy'])
pred = model.predict([img_vector, question_vector])[0]
top_pred = pred.argsort()[:-5][::-1]

print [(metadata['ix_to_ans'][str(_)], pred[_]) for _ in top_pred]

