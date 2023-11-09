import cv2
import numpy as np

WIDTH_DEFAULT = 300
HEIGHT_DEFAULT = 300

class FaceDetector:
    __slots__ = ("classificator", "width", "height")

    def __init__(self) -> None:
        self.classificator = cv2.CascadeClassifier("../models/haarcascade_frontalface_default.xml")
        self.width = WIDTH_DEFAULT
        self.height = HEIGHT_DEFAULT
    
    def detect(self, image, fileName):
        # image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        facesDetect = self.classificator.detectMultiScale(imageGray, scaleFactor=1.5, minSize=(100,100))

        for(x, y, l, a) in facesDetect:
            imageFace = cv2.resize(imageGray[y:y + a, x:x + 1], (self.width, self.height))
            cv2.imwrite(f"../storage/faces/{fileName}", imageFace)