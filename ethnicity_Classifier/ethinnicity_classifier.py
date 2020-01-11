import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
import os
import cv2
from random import shuffle
from keras.layers import Convolution2D, MaxPooling2D, Conv2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical

import pickle

import tqdm


# model=Sequential()
# model.add(Conv2D(64, (7, 7), padding="same", activation="relu", input_shape=(128, 128, 3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.50))

# model.add(Conv2D(32, (5, 5), padding="same", strides =(1, 1), activation="relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(32, (3, 3), padding="same", strides =(2, 2), activation="relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.50))

# model.add(Flatten()) 
# model.add(Dense(526, activation ='relu')) 
# model.add(Dense(5, activation ='softmax')) 

# model.compile(loss = keras.losses.categorical_crossentropy, 
#               optimizer = keras.optimizers.SGD(lr = 0.01), 
#               metrics =['accuracy']) 

arr=os.listdir("part2")
shuffle(arr)
Y=[]
X=[]
for x in arr:
	Y.append(int(x.split('_')[2]))

Y = to_categorical(Y,num_classes=5)


if "data.pkl" not in os.listdir():
	for x in tqdm.tqdm(arr):
		z = cv2.imread("part2/" + x)
		z = cv2.resize(z, (128, 128) )
		X.append(z)
	X = np.squeeze(X)
	with open("data.pkl", 'wb') as f:
		pickle.dump(X, f)
		print("made pickle dump")

with open("data.pkl", 'rb') as f:
	X = pickle.load(f)
	print("loaded pickle")

X = X.astype('float32')
X /= 255
X_train=X[:7000]
X_validate=X[7000:8500]
X_test=X[8500:]
Y = np.squeeze(Y)
Y_train=Y[:7000]
Y_validate=Y[7000:8500]
Y_test=Y[8500:]

#print(model.summary())

#model.fit(X_train,Y_train,batch_size=32,epochs=3,validation_data=(X_validate,Y_validate))
# r=model.evaluate(X_test, Y_test, verbose=0)
# print(r[1])
file="9_0_1_20170113175830459.jpg"
from PIL import Image
im =Image.open(file)
im
X_data=[]
face = cv2.imread(file)
face = cv2.resize(face, (128, 128) )
X_data.append(face)
np.squeeze(X_data)
X = X.astype('float32')
X /= 255
loaded_model = pickle.load(open("weight.pkl", 'rb'))
y=loaded_model.predict(X_data)
print(y)







