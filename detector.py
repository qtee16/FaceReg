import cv2
import numpy as np
import mtcnn
from architecture import *

from scipy.spatial.distance import cosine
from tensorflow.keras.models import load_model
import pickle
import train


class Detector:
    def __init__(self,trainer=None):
        super().__init__()
        self.confidence_t = 0.99
        self.recognition_t = 0.5
        self.required_size = (160, 160)

        self.face_encoder = InceptionResNetV2()
        self.path_m = "weight/facenet_keras_weights.h5"
        self.face_encoder.load_weights("weight/facenet_keras_weights.h5")
        self.encodings_path = 'encodings/encodings.pkl'
        self.face_detector = mtcnn.MTCNN()
        self.encoding_dict = None

        if(trainer==None):
            self.trainer = train.Train()
        else:
            self.trainer = trainer

    def get_face(self,img, box):
        x1, y1, width, height = box
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = img[y1:y2, x1:x2]
        return face, (x1, y1), (x2, y2)

    def get_encode(self, face, size):

        face = self.trainer.normalize(face)
        face = cv2.resize(face, size)
        encode = self.face_encoder.predict(np.expand_dims(face, axis=0))[0]
        return encode

    def load_pickle(self,path):
        with open(path, 'rb') as f:
            self.encoding_dict = pickle.load(f)

    def detect(self,img, face_only=False):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detector.detect_faces(img_rgb)
        name = ''
        for res in results:
            if res['confidence'] < self.confidence_t:
                continue
            face, pt_1, pt_2 = self.get_face(img_rgb, res['box'])

            encode = self.get_encode( face, self.required_size)
            encode = self.trainer.l2_normalizer.transform(encode.reshape(1, -1))[0]
            name = 'unknown'

            distance = float("inf")
            self.load_pickle(self.encodings_path)
            for db_name, db_encode in self.encoding_dict.items():
                dist = cosine(db_encode, encode)
                if dist < self.recognition_t and dist < distance:
                    name = db_name
                    distance = dist

            if name == 'unknown':
                cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
                cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            else:
                cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
                cv2.putText(img, name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 200, 200), 2)
        if face_only:
            if len(results)!=0:
                return results[-1]['box'], name
            else:
                return None, name
        return img

if __name__== "__main__":
    cap = cv2.VideoCapture(0)
    detector = Detector()
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("CAM NOT OPEN")
            break
        frame = detector.detect(frame)

        cv2.imshow('camera', frame)

        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
