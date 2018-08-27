from __future__ import print_function

import numpy as np
import warnings

from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing import image
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from imagenet_utils import decode_predictions, preprocess_input

from vgg16 import VGG16
from analyze import Analyze
from food5k import load_data
from anomalydata import load_anomaly_data

if __name__ == '__main__':

  # (X_train, y_train), (X_test, y_test) = cifar10.load_data()
  # print('y_train: ' , y_train)
  #print('y_test: ' , y_test)

  (x_train,x_test,y_test) = load_anomaly_data()

  model = VGG16(include_top=True, weights='imagenet')
  preds_train = model.predict(x_train)
  preds_test = model.predict(x_test)  
  Analyze(preds_train,preds_test,y_test)



