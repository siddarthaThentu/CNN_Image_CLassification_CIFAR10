from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from skimage.transform import resize
import numpy as np

model=load_model('my_cifar10_model.h5')

image = plt.imread("frog.jpg")

image_resized = resize(image,(32,32,3))

img = plt.imshow(image_resized)

probabilities = model.predict(np.array([image_resized,]))

print(probabilities)

number_to_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
index = np.argsort(probabilities[0,:])
print("Most likely class:", number_to_class[index[9]], "-- Probability:", probabilities[0,index[9]])
print("Second most likely class:", number_to_class[index[8]], "-- Probability:", probabilities[0,index[8]])
print("Third most likely class:", number_to_class[index[7]], "-- Probability:", probabilities[0,index[7]])
print("Fourth most likely class:", number_to_class[index[6]], "-- Probability:", probabilities[0,index[6]])
print("Fifth most likely class:", number_to_class[index[5]], "-- Probability:", probabilities[0,index[5]])