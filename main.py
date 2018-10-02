from tensorflow import keras
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import time
import cv2
from processes import predict

camera = PiCamera()
stream = PiRGBArray(camera)
size = (224,224,3)
camera.start_preview(fullscreen=False, window=(100,20,800,600))
while True:
    time.sleep(1)
    try:
        camera.capture(stream, format='bgr')
        image = stream.array
        im = cv2.resize(image, size[:2])
        #im = keras.preprocessing.image.load_img('test.jpeg', target_size=size)
        #im = keras.preprocessing.image.img_to_array(im)
        start_time = time.time()
        x = np.expand_dims(im.astype('float32'), axis=0)
        # x = keras.applications.mobilenet.preprocess_input(x)
        # pred = model.predict(x)
        # out = keras.applications.mobilenet.decode_predictions(pred)
        out = predict(x)
        print (out)
        print("--- %s seconds ---" % (time.time() - start_time))
        stream.seek(0)
    except:
        camera.stop_preview()
        break
