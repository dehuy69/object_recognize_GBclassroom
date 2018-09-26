from tensorflow import keras
import numpy as np
import time

size = (112,112,3)
model = keras.applications.MobileNet(input_shape=size)
im = keras.preprocessing.image.load_img('test.jpeg', target_size=size)
im = keras.preprocessing.image.img_to_array(im)
start_time = time.time()
x = np.expand_dims(im, axis=0)
x = keras.applications.mobilenet.preprocess_input(x)
pred = model.predict(x)
out = keras.applications.mobilenet.decode_predictions(pred)
print (out)
print("--- %s seconds ---" % (time.time() - start_time))