Visual Question Answering with Keras
===================================

Recent developments in Deep Learning has paved the way to accomplish tasks involving multimodal learning. Visual Question Answering [(VQA)](http://www.visualqa.org/) is one such challenge which requires high-level scene interpretation from images combined with language modelling of relevant Q&A. Given an image and a natural language question about the image, the task is to provide an accurate natural language answer. This is a [Keras](http://keras.io) implementation of one such end-to-end system to accomplish the task.

Checkout the demo [here](https://anantzoid.github.io/VQA-Keras-Visual-Question-Answering/): 
![Demo](http://i.imgur.com/pB3bGeo.jpg)

### Architecture
The learning architecture behind this demo is based on the model proposed in the [VQA paper](http://arxiv.org/pdf/1505.00468v6.pdf).

![Architecure](http://i.imgur.com/2zJ09mQ.png)

The problem is considered as a classification task here, wherein, 1000 top answers are chosen as classes. Images are transformed by passing it through the [VGG-19 model](https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d) that generates a 4096 dimensional vector in the second last layer. The tokens in the question are first embedded into 300 dimensional GloVe vectors and then passed through 2 layer LSTMs. Both multimodal data points are then passed through a dense layer of 1024 units and combined using point-wise multiplication. The new vector serves as input for a fully-connected model having a `tanh` and a final `softmax` layer.

### Data
Preprocessed features provided by [VT Vision Lab](https://github.com/VT-vision-lab) was used which consisted of images transformed through VGG19 model and indexed tokens.

### Installation
The following packages need to be installed before running the scripts:
-   [Keras](https://keras.io/) (and the corresponding backend: [Theano](https://pypi.python.org/pypi/Theano)/[TensorFlow](http://tensorflow.org/))
-   [h5py](http://www.h5py.org/)

Then go to the `data` folder and download the requirements given over there.

### Training
Run `python train.py` along with the following optional parameters: `--epoch`, `--batch_size`, `--data_limit`.

To evaluate the model on validation set, run `python train.py --type val`.

### Training Details
Preprocessed features have been used based on these scripts written by the  [VT vision lab](https://github.com/VT-vision-lab/VQA_LSTM_CNN) team. These features already consist of transformed image vectors, indexed tokens for text and other metadata, for both the training and validation set.

Training was done on g2.2xlarge spot instance of AWS. Mutltiple commuity AMIs can be found having all the required packages pre-installed. g2.2xlarge has a NVIDIA Grid K520 with 4GB memory and takes ~277 seconds/epoch for a batch size of 256. The model has been trained on 50 epochs and has a accuracy of 45.03% on the validation set. Also, the accuracy started decreasing after 70 epochs. Thus, there is a lot of scope for hyper-parameter tuning here.

### Running the application
For details on how to run the demo app, check the docs in `app/` folder.

### Feedback
If you have any feedback or suggestions, do ping me at anant718@gmail.com


