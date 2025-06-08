import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from model.model_evaluate import load_model, class_dict, generate_cam_image

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

model = load_model()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        pred_class, class_name, result_path = generate_cam_image(model, filepath, class_dict)

        return render_template('classify.html',
                               filename=filename,
                               class_name=class_name,
                               result_image=result_path)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename=f'uploads/{filename}'))

if __name__ == '__main__':
    app.run(debug=True)

