import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import mobilenet
from keras.layers import GlobalAveragePooling2D, Dense, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.utils import to_categorical

labels = ['Ruler', 'Paper', 'No Object', 'Tape', 'Scissor', 'Eraser', 'Pencil',
          'Rectangular Builder Block', 'Whiteboard marker', 'Primary Battery', 'Paint Brush',
          'Triangular Prism Building block']

def load_data(path='dataset', shape=(224,224)):
    x_train =[]
    y_train =[]
    for label in labels:
        label_path = os.path.join(path, label)
        im_names = os.listdir(label_path)
        for im_name in im_names:
            im_path = os.path.join(label_path, im_name)
            im = load_img(im_path, target_size=shape)
            im = img_to_array(im)
            x_train.append(im)
            y_train.append(labels.index(label))
    return np.array(x_train), np.array(y_train)

def create_model():
    encode_model = mobilenet.MobileNet(include_top=False, input_shape=(224, 224, 3))
    x = encode_model.output
    x = GlobalMaxPooling2D()(x)
    out = Dense(len(labels), activation='softmax')(x)
    model = Model(inputs=encode_model.input, outputs=out)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  #optimizer=SGD(lr=0.0001, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
    return model

def retrain_process():
    x_train, y_train = load_data()
    x_encode = mobilenet.preprocess_input(x_train)
    y_encode = to_categorical(y_train, len(labels))
    model = create_model()
    model.fit(x_encode, y_encode, epochs=100, batch_size=32)
    model.save('model.h5')
    model.save_weights('weight.h5')

model = load_model('model.h5')
model.load_weights('weight.h5')
def predict(im):
    x = np.expand_dims(im.astype('float32'), axis=0)
    x = mobilenet.preprocess_input(x)
    pred = model.predict(x)[0]
    print (pred)
    max_score = pred.max()
    index = pred.tolist().index(max_score)
    label_pred = labels[index]
    print (label_pred)

if __name__ == '__main__':
    retrain_process()
    # im = load_img('dataset/Eraser/Eraser (1).JPG', target_size=(224, 224))
    # im = img_to_array(im)
    # predict(im)