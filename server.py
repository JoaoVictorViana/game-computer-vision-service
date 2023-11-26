from flask import Flask, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
from src.services.FaceDetector import FaceDetector
from src.services.ImageClassificator import ImageClassificator
from src.libs.utils import getImagesForTrain, getKeyboardForTrain
import os
import cv2
import numpy as np
import logging

app = Flask(__name__)
app.config.from_pyfile('application.cfg', silent=True)
CORS(app)

@app.route("/detect/face", methods=['POST'])
def upload_image_detector():
    f = request.files['image']

    IMAGE_DETECTOR_PATH = app.config.get('IMAGE_DETECTOR_PATH')
    MODEL_OUTPUT_PATH = app.config.get('MODEL_OUTPUT_PATH')
    RAW_IMAGE_PATH = os.path.join(app.root_path, f'{IMAGE_DETECTOR_PATH}/raw')
    OUTPUT_MODEL_PATH = os.path.join(app.root_path, f'{MODEL_OUTPUT_PATH}/output')

    os.makedirs(RAW_IMAGE_PATH, exist_ok=True)
    os.makedirs(OUTPUT_MODEL_PATH, exist_ok=True)
    fileName = os.path.join(RAW_IMAGE_PATH, secure_filename(f.filename))
    f.save(fileName)

    faceDetector = FaceDetector()
    userId = faceDetector.predict(fileName=fileName, output=OUTPUT_MODEL_PATH)

    if (not userId):
        return {
            "data": None,
            "message": "Usuário não identificado"
        }, 401

    return {
        "data": userId
    }


@app.route("/upload/face/train", methods=['POST'])
def upload_images_train():
    requestForm = request.form.to_dict()
    f = request.files['image']

    IMAGE_TRAIN_PATH = app.config.get('IMAGE_TRAIN_PATH')
    RAW_IMAGE_PATH = os.path.join(app.root_path, f'{IMAGE_TRAIN_PATH}/raw')
    OUTPUT_IMAGE_PATH = os.path.join(app.root_path, f'{IMAGE_TRAIN_PATH}/output')

    os.makedirs(RAW_IMAGE_PATH, exist_ok=True)
    os.makedirs(OUTPUT_IMAGE_PATH, exist_ok=True)
    fileName = os.path.join(RAW_IMAGE_PATH, secure_filename(f.filename))
    f.save(fileName)

    faceDetector = FaceDetector()
    img = faceDetector.detect(fileName)

    userId = requestForm.get('userId')

    if (type(img) == np.ndarray):
        cv2.imwrite(f'{OUTPUT_IMAGE_PATH}/user-{userId}-{secure_filename(f.filename)}', img)
        return {
            "message": f"Imagem do usuário ({userId}) salva com sucesso!"
        }, 201

    return {
        "message": "Nenhum usuário identificado na imagem, por favor envie novamente"
    }, 422

@app.route("/model/face/train", methods=['POST'])
def model_train():
    IMAGE_TRAIN_PATH = app.config.get('IMAGE_TRAIN_PATH')
    MODEL_OUTPUT_PATH = app.config.get('MODEL_OUTPUT_PATH')
    OUTPUT_IMAGE_PATH = os.path.join(app.root_path, f'{IMAGE_TRAIN_PATH}/output')
    OUTPUT_MODEL_PATH = os.path.join(app.root_path, f'{MODEL_OUTPUT_PATH}/output')

    os.makedirs(OUTPUT_IMAGE_PATH, exist_ok=True)
    os.makedirs(OUTPUT_MODEL_PATH, exist_ok=True)

    ids, faces = getImagesForTrain(OUTPUT_IMAGE_PATH)
    
    faceDetector = FaceDetector()
    faceDetector.train(X=faces, classes=ids, output=OUTPUT_MODEL_PATH)

    return 'success'

@app.route("/upload/keyboard/train", methods=['POST'])
def upload_keyboard_train():
    requestForm = request.form.to_dict()
    f = request.files['image']

    userId = requestForm.get('userId')
    keyboard = requestForm.get('keyboard')

    KEYBOARD_TRAIN_PATH = app.config.get('KEYBOARD_TRAIN_PATH')
    RAW_KEYBOARD_PATH = os.path.join(app.root_path, f'{KEYBOARD_TRAIN_PATH}/{userId}/raw')

    os.makedirs(RAW_KEYBOARD_PATH, exist_ok=True)
    fileName = os.path.join(RAW_KEYBOARD_PATH, f'{keyboard}-{secure_filename(f.filename)}')
    f.save(fileName)

    return {
        "message": f"Imagem do usuário ({userId}) salva com sucesso!"
    }, 201

@app.route("/model/keyboard/train", methods=['POST'])
def model_keyboard_train():
    requestForm = request.get_json(force=True)
    userId = requestForm.get('userId')
    
    MODEL_OUTPUT_PATH = app.config.get('MODEL_OUTPUT_PATH')
    KEYBOARD_TRAIN_PATH = app.config.get('KEYBOARD_TRAIN_PATH')
    RAW_KEYBOARD_PATH = os.path.join(app.root_path, f'{KEYBOARD_TRAIN_PATH}/{userId}/raw')
    OUTPUT_MODEL_PATH = os.path.join(app.root_path, f'{MODEL_OUTPUT_PATH}/keyboard/{userId}')

    os.makedirs(RAW_KEYBOARD_PATH, exist_ok=True)
    os.makedirs(OUTPUT_MODEL_PATH, exist_ok=True)

    keyboards, images = getKeyboardForTrain(RAW_KEYBOARD_PATH)
    
    imageClassificator = ImageClassificator()
    imageClassificator.train(X=images, classes=keyboards, output=OUTPUT_MODEL_PATH)

    return 'success'

@app.route("/detect/keyboard", methods=['POST'])
def upload_keyboard_detector():
    f = request.files['image']
    requestForm = request.form.to_dict()
    userId = requestForm.get('userId')

    KEYBOARD_DETECTOR_PATH = app.config.get('KEYBOARD_DETECTOR_PATH')
    MODEL_OUTPUT_PATH = app.config.get('MODEL_OUTPUT_PATH')
    RAW_KEYBOARD_PATH = os.path.join(app.root_path, f'{KEYBOARD_DETECTOR_PATH}/{userId}/raw')
    OUTPUT_MODEL_PATH = os.path.join(app.root_path, f'{MODEL_OUTPUT_PATH}/keyboard/{userId}')

    os.makedirs(RAW_KEYBOARD_PATH, exist_ok=True)
    fileName = os.path.join(RAW_KEYBOARD_PATH, secure_filename(f.filename))
    f.save(fileName)

    imageClassificator = ImageClassificator()
    keyboard = imageClassificator.predict(fileName=fileName, output=OUTPUT_MODEL_PATH)
    logging.warning(keyboard)
    if (not keyboard):
        return {
            "data": None,
            "message": "Não foi possível identificar o comando"
        }, 422

    return {
        "data": f'{keyboard}'
    }