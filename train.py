from architecture import *
import os
import cv2
import mtcnn
import pickle
import numpy as np
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import load_model


class Train():

    def __init__(self):
        super().__init__()
        self.face_path = ''
        # share required to fit model
        self.required_shape = (160,160)

        # detect the face in image
        self.face_detector = mtcnn.MTCNN()

        # network to encode the face into a vector to predict
        self.face_encoder = InceptionResNetV2()

        # pretrained weight
        self.path = "weight/facenet_keras_weights.h5"
        self.face_encoder.load_weights(self.path)

        self.encodes = []
        self.encoding_dict = dict()
        self.l2_normalizer = Normalizer('l2')


    def train(self):
        #path to face data
        self.face_path = 'Faces/'

        # start the training process
        # iterate through the face dataset
        for face_names in os.listdir(self.face_path):
            # subdir in faces directory
            person_dir = os.path.join(self.face_path, face_names)

            # iterate through each image
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)

                img_BGR = cv2.imread(image_path)
                img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

                x = self.face_detector.detect_faces(img_RGB)
                x1, y1, width, height = x[0]['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                face = img_RGB[y1:y2, x1:x2]

                # normalize and resize the face to the shape needed
                face = self.normalize(face)
                face = cv2.resize(face, self.required_shape)
                face_d = np.expand_dims(face, axis=0)
                encode = self.face_encoder.predict(face_d)[0]
                self.encodes.append(encode)

            if self.encodes:
                encode = np.sum(self.encodes, axis=0)
                encode = self.l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
                self.encoding_dict[face_names] = encode

        path = 'encodings/encodings.pkl'
        with open(path, 'wb') as file:
            pickle.dump(self.encoding_dict, file)


    def normalize(self, img):
        mean, std = img.mean(), img.std()
        return (img-mean) / std

