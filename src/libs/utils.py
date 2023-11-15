import os
import cv2
import numpy as np

def getImagesForTrain(path: str):
    imagesPathnames = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []

    for imagePath in imagesPathnames:
        imageFace = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(imagePath)[-1].split('-')[1])
        ids.append(id)
        faces.append(imageFace)

    return np.array(ids), faces 

