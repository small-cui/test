import keras
import os
from keras.models import Model
from keras.datasets import mnist
from keras.layers import Input,Flatten,Dense
from keras.layers import Conv2D,MaxPooling2D,Dropout

from keras.callbacks import ModelCheckpoint
#from keras.optimizers import Adadelta
img_rows,img_cols=28,28
num_classes=10
def get_mnist_data():
  (x_train,y_train),(x_test,y_test)=mnist.load_data()

  #x_train=x_train.reshape(len(x_train),-1)
  #x_test=x_test.reshape(len(x_test),-1)
  x_train=x_train.reshape(x_train.shape[0],img_rows,img_cols,1)
  x_test=x_test.reshape(x_test.shape[0],img_rows,img_cols,1)
  x_train=x_train.astype('float32')
  x_test=x_test.astype('float32')
  x_train /=255
  x_test /=255

  y_train=keras.utils.to_categorical(y_train,num_classes)
  y_test=keras.utils.to_categorical(y_test,num_classes)
  return x_train,y_train,x_test,y_test
x_train,y_train,x_test,y_test=get_mnist_data()
def LeNet5(w_path=None):
  input_shape=(img_rows,img_cols,1)
  #img_input=Input(shape=(784,))
  img_input=Input(shape=input_shape)
  x=Conv2D(32,(3,3),activation="relu",padding="same",name="conv1")(img_input)
  x=MaxPooling2D((2,2),strides=(2,2),name='pool1')(x)
  x=Conv2D(64,(5,5),activation="relu",name="conv2")(x)
  x=MaxPooling2D((2,2),strides=(2,2),name='pool2')(x)
  x=Dropout(0.25)(x)
  
  x=Flatten(name='flatten')(x)

  x=Dense(120,activation='relu',name='fc1')(x)
  x=Dropout(0.5)(x)
  x=Dense(84,activation='relu',name='fc2')(x)
  x=Dropout(0.5)(x)
  x=Dense(10,activation='softmax',name='predictions')(x)

  model=Model(img_input,x,name='LeNet5')
  if(w_path):model.load_weights(w_path)

  return model

lenet5=LeNet5()
#lenet5.summary()

if not os.path.exists('lenet5_checkpoints'):
  os.mkdir('lenet5_checkpoints')
lenet5.compile(loss=keras.losses.categorical_crossentropy,
               optimizer=keras.optimizers.Adadelta(),
               metrics=['accuracy'])
checkpoint=ModelCheckpoint(monitor='val_acc',
                           filepath='lenet5_checkpoints/model_{epoch:02d}.h5',
                           save_best_only=True)
lenet5.fit(x_train,y_train,
           batch_size=128,
           epochs=2,
           verbose=1,
           validation_data=(x_test,y_test),
           callbacks=[checkpoint])
score=lenet5.evaluate(x_test,y_test,verbose=0)
print('Test loss:',score[0])
print('Test accurary:',score[1])