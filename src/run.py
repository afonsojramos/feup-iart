from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from IPython.display import SVG
from collections import Counter
from keras.utils.vis_utils import model_to_dot
from imblearn.datasets import make_imbalance

import numpy as np
import pandas as pds

""" dataframe = pds.read_csv('../dataset/HTRU_2.csv', names=['Integrated Profile: Mean', 'Standard Deviation', 'Excess Kurtosis', 'Skewness', 'DM-SNR Curve: Mean', '_Standard Deviation', '_Excess Kurtosis', '_Skewness']) """

np.random.seed(42)

dataframe = np.loadtxt('../dataset/HTRU_2.csv', delimiter=',', dtype=np.float64)

def split_train_dataset(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(test_ratio * len(data))
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data[train_indices,:], data[test_indices,:]

def ratio_func(y, multiplier, minority_class):
    target_stats = Counter(y)
    return {minority_class: int(multiplier * target_stats[minority_class])}

data_dim = 8
 
model = Sequential()
 
model.add(Dense(128, input_dim=data_dim, kernel_initializer='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, kernel_initializer='uniform'))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
 
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',  
              metrics=["accuracy"])
 
# generate training data
train, test = split_train_dataset(dataframe, 0.2)
x_train, y_train = train[:,:8], train[:,8]

multipliers = [0.9, 0.75, 0.5, 0.25, 0.1]

for i, multiplier in enumerate(multipliers, start=1):
    X_train, Y_train = make_imbalance(x_train, y_train, ratio=ratio_func,
                        **{"multiplier": multiplier,
                        "minority_class": 0})
print(test.shape, train.shape, X_train.shape)

# generate test data
x_test, y_test = test[:,:8], test[:,8]
 
# model fitting
model.fit(x_train, y_train,
          epochs=5,
          batch_size=128)
 
score = model.evaluate(x_test, y_test, batch_size=128)
print(score)

model.save('my_model.h5')
#SVG(model_to_dot(model).create(prog='dot', format='svg'))