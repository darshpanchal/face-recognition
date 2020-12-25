from PIL import Image
from pycoral.adapters import classify, common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
import platform
import base64
import numpy as np
import io
import cv2
import argparse
from faceextract import faceextract


def faceedit(face, size, flipdir):
    face = cv2.resize(face, size)
    face = cv2.flip(face, flipdir)
    return face

def faceinference(interpreter, face, labels, frame, person,i):
    common.set_input(interpreter, face)
    interpreter.invoke()
    classes = classify.get_classes(interpreter, 1, 0.0)
    pred = []
    for class1 in classes:
        pred.append(str(labels.get(class1.id, class1.id)))
    print(pred)
    if person in pred:
        cv2.imwrite('pics/'+str(i)+'.jpg', frame)


def videoread(vid, face_cascade, interpreter, labels, person):
    cap = cv2.VideoCapture(vid)
    # savedframes = []
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        else:
            faces = faceextract(frame, face_cascade)
            print(len(faces))
            if faces:
                for face in faces:
                    face = faceedit(face, (224,224), 1)
                    faceinference(interpreter, face, labels, frame, person,i)
        i += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', required=True,
                        help='File path of edgetpu .tflite file.')
    parser.add_argument('-l', '--labels', required=True,
                        help='File path of labels .txt file.')
    parser.add_argument('-c', '--haar', required=True,
                        help='File path of haarcascade .xml file.')
    parser.add_argument('-v', '--video', required=True,
                        help='File path of video file.')
    parser.add_argument('-f', '--find', required=True,
                        help='Label to look for in video.')
    args = parser.parse_args()
    print("Looking for " + args.find)
    face_cascade = cv2.CascadeClassifier(args.haar)
    labels = read_label_file(args.labels)
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    size = common.input_size(interpreter)
    videoread(args.video, face_cascade, interpreter, labels, args.find)