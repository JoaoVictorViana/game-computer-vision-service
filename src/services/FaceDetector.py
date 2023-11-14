import cv2
import numpy as np

WIDTH_DEFAULT = 300
HEIGHT_DEFAULT = 300

class FaceDetector:
    __slots__ = ("classificator", "width", "height")

    def __init__(self) -> None:
        self.classificator = cv2.CascadeClassifier(filename=cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.width = WIDTH_DEFAULT
        self.height = HEIGHT_DEFAULT
    
    def detect(self, fileName, output):
        # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.imread(filename=fileName)
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        facesDetect = self.classificator.detectMultiScale(imageGray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

        n = 0
        for (x, y, l, a) in facesDetect:
            img = cv2.rectangle(imageGray, (x, y), (x + l, y + a), (0, 255, 0), 4)
            cv2.imwrite(f'./src/storage/faces/detect/teste-2.jpg', img)

            imageFace = cv2.resize(imageGray[y:y + a, x:x + 1], (self.width, self.height))
            cv2.imwrite(f'./src/storage/faces/detect/teste-{++n}.jpg', imageFace)
            