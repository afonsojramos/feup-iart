from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot
import numpy as np
import pandas as pds

dataframeX = pds.read_csv('../dataset/HTRU_2.csv', names=['Integrated Profile: Mean', 'Standard Deviation', 'Excess Kurtosis', 'Skewness', 'DM-SNR Curve: Mean', '_Standard Deviation', '_Excess Kurtosis', '_Skewness'], usecols=[0, 1, 2, 3, 4, 5, 6, 7])
dataframeY = pds.read_csv('../dataset/HTRU_2.csv', names=["Class"] , usecols=[8])

print(dataframeX.head())
print(dataframeY.head())
 
data_dim = 20
nb_classes = 4
 
model = Sequential()
 
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, input_dim=data_dim, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, init='uniform'))
model.add(Activation('softmax'))
 
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',  
              metrics=["accuracy"])
 
# generate dummy training data
x_train = np.random.random((1000, data_dim))
y_train = np.random.random((1000, nb_classes))
 
# generate dummy test data
x_test = np.random.random((100, data_dim))
y_test = np.random.random((100, nb_classes))
 
model.fit(x_train, y_train,
          nb_epoch=20,
          batch_size=16)
 
score = model.evaluate(x_test, y_test, batch_size=16)

#SVG(model_to_dot(model).create(prog='dot', format='svg'))