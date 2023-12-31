from app import app
from flask import request, jsonify
from app.utils import process_image

@app.route('/predict-image', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return 'No image provided', 400

    image_file = request.files['image']
    result = process_image(image_file)
    return jsonify(result)
