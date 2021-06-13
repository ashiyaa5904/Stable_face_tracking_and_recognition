import pickle
import cv2
import numpy as np
from face_recognition.train_data import normalize


def get_face(img, box):
    # box is drawn using the values x1,y1,w,h
    x1, y1, width, height = box
    # first coordinate
    x1, y1 = abs(x1), abs(y1)
    # second coordinate
    x2, y2 = x1 + width, y1 + height
    # used to crop the face
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)


def get_encode(face_encoder, face, size):
    # Normalization also makes the training process less sensitive to the scale of the features.
    # This results in getting better coefficients after training.
    face = normalize(face)
    # resize the image to an appropriate size
    face = cv2.resize(face, size)
    # encode the image
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode


def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict
