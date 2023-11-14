from flask import Flask, request
from werkzeug.utils import secure_filename
from src.services.FaceDetector import FaceDetector
from PIL import Image
import os

app = Flask(__name__)

@app.route("/upload", methods=['POST'])
def upload_images():
    f = request.files['image']

    os.makedirs(os.path.join(app.root_path, 'src/storage/faces/raw'), exist_ok=True)
    os.makedirs(os.path.join(app.root_path, 'src/storage/faces/detect'), exist_ok=True)

    fileName = os.path.join(app.root_path, 'src/storage/faces/raw', secure_filename(f.filename))
    f.save(fileName)

    output = os.path.join(app.root_path, 'src/storage/faces/detect/')

    faceDetector = FaceDetector()
    faceDetector.detect(fileName, output)
    return 'success'