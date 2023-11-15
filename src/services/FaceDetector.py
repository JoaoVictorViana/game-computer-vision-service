import cv2
import numpy as np
import logging

WIDTH_DEFAULT = 300
HEIGHT_DEFAULT = 300

class FaceDetector:
    __slots__ = ("classificator", "eyesClassificator", "width", "height")

    def __init__(self) -> None:
        self.classificator = cv2.CascadeClassifier(filename=cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eyesClassificator = cv2.CascadeClassifier(filename=cv2.data.haarcascades + "haarcascade_eye.xml")
        self.width = WIDTH_DEFAULT
        self.height = HEIGHT_DEFAULT
    
    def detect(self, fileName):
        image = cv2.imread(filename=fileName)
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        facesDetect = self.classificator.detectMultiScale(imageGray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

        for (x, y, l, a) in facesDetect:
            regionEyes = image[y:y + a, x:x + l]
            regionGrayEyes = cv2.cvtColor(regionEyes, cv2.COLOR_BGR2GRAY)
            eyesDetect = self.eyesClassificator.detectMultiScale(regionGrayEyes)

            for (ex, ey, el, ea) in eyesDetect:
                return cv2.resize(imageGray[y:y + a, x:x + l], (self.width, self.height))
        
        return False
    
    def train(self, X, classes, output):
        eigenFace = cv2.face.EigenFaceRecognizer.create(num_components=50)
        fisherFace = cv2.face.FisherFaceRecognizer.create()
        lbph = cv2.face.LBPHFaceRecognizer.create(radius=2,neighbors=10, grid_x=10, grid_y=10)

        eigenFace.train(X, labels=classes)
        eigenFace.write(f'{output}/eigen.yml')

        # fisherFace.train(X, labels=classes)
        # fisherFace.write(f'{output}/fisher.yml')

        lbph.train(X, labels=classes)
        lbph.write(f'{output}/lbph.yml')
    
    def predict(self, fileName, output):
        img = self.detect(fileName=fileName)

        logging.warn(img)
        if (type(img) != np.ndarray):
            return False

        eigenFace = cv2.face.EigenFaceRecognizer.create()
        fisherFace = cv2.face.FisherFaceRecognizer.create()
        lbph = cv2.face.LBPHFaceRecognizer.create()

        eigenFace.read(f'{output}/eigen.yml')
        lbph.read(f'{output}/lbph.yml')

        idEigen, trustEigen = eigenFace.predict(img)
        idLbph, trustLbph = lbph.predict(img)

        logging.warning(idLbph)

        if (idEigen == -1):
            return False
        
        return idEigen

            