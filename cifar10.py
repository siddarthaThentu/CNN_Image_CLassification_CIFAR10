import keras
from tensorflow.keras.datasets import cifar10
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D

(x_train,y_train),(x_test,y_test) = cifar10.load_data()

y_train_one_hot = keras.utils.to_categorical(y_train,10)
y_test_one_hot = keras.utils.to_categorical(y_test,10)

x_train = x_train.astype('float32')
x_test =x_test.astype('float32')
x_train = x_train/255
x_test = x_test/255

model = Sequential()

model.add(Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(32,32,3)))
model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

hist = model.fit(x_train, y_train_one_hot, batch_size=32, epochs=20,validation_split=0.2)

model.save('my_cifar10_model.h5')

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train','Validation'],loc='upper right')
plt.show()

#plt.plot(hist.history['acc'])
#plt.plot(hist.history['val_acc'])
#plt.title('Model accuracy')
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
#plt.legend(['Train','Validation'],loc='upper right')
#plt.show()

model.evaluate(x_test,y_test_one_hot)[1]

model.save('my_cifar10_model.h5')