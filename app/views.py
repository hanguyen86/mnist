"""
Handwritten Digit Recognition
Licence: BSD
Author : Hoang Anh Nguyen
"""

import os
from flask import redirect, render_template, flash, request, jsonify
from werkzeug.utils import secure_filename
from app import app
from classifier import SoftmaxClassifier, CNNClassifier

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods = ['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # check if the post request has the file/image part
        if 'file' in request.files:
            file = request.files['file']
        elif 'image' in request.files:
            file = request.files['image']
        else:
            return render_template('404.html'), 404
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return render_template('404.html'), 404
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return recognize_image(filepath)
    return render_template('index.html')

# wrap classification result in JSON response
def makeJSONOutput(label):
    if not label:
        return render_template('404.html'), 404
    
    result = {
        'classifier': app.config['CLASSIFIER_TYPE'],
        'label'     : label
    }
    return jsonify(**result)

def recognize_image(filename):
    with eval(app.config['CLASSIFIER_TYPE'])() as recognizer:
        label = recognizer.predict(filename)
        return makeJSONOutput(label)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']