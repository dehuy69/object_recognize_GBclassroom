from tensorflow import keras
import numpy as np

size = (224,224,3)
model = keras.applications.MobileNet(input_shape=size)
im = keras.preprocessing.image.load_img('test.jpeg', target_size=size)
im = keras.preprocessing.image.img_to_array(im)
x = np.expand_dims(im, axis=0)
x = keras.applications.mobilenet.preprocess_input(x)
pred = model.predict(x)
out = keras.applications.mobilenet.decode_predictions(pred)
print out