from tensorflow import keras
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import time
camera = PiCamera()
camera.resolution = (800, 600)
rawCapture = PiRGBArray(camera)
privious_data = []
privious_time = 0
camera.start_preview(fullscreen=False, window=(100,20,800,600))
size = (224,224,3)
model = keras.applications.MobileNet(input_shape=size)
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    try:
        frame = rawCapture.array
        im = cv2.resize(frame, size[:2])
        # im = keras.preprocessing.image.load_img('test.jpeg', target_size=size)
        im = keras.preprocessing.image.img_to_array(im)
        start_time = time.time()
        x = np.expand_dims(im, axis=0)
        x = keras.applications.mobilenet.preprocess_input(x)
        pred = model.predict(x)
        out = keras.applications.mobilenet.decode_predictions(pred)
        print (out)
        print("--- %s seconds ---" % (time.time() - start_time))
        rawCapture.truncate(0)
    except:
        camera.stop_preview()
        break