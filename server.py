from flask import Flask, request
from werkzeug.utils import secure_filename
from src.services.FaceDetector import FaceDetector
from PIL import Image

app = Flask(__name__)

@app.route("/upload", methods=['POST'])
def upload_images():
    f = request.files['image']
    image = Image.open(f)
    faceDetector = FaceDetector()
    faceDetector.detect(image, secure_filename(f.filename))
    return 'success'