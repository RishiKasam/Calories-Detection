from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
import io
from PIL import Image

app = Flask(__name__)

# Load the YOLO model
model = YOLO('./best2.pt')


food_calories = {
    "apple": 50,
    "banana": 100,
    "biriyani": 60,
    "chapali-kabab": 15,
    "chicken-karahi": 70,
    "kofta": 50,
    "meat-balls": 100,
    "orange": 120,
    "palak": 150,
    "paratha": 70,
    "samosa": 150,
    "seekh-kabab": 100
}


@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    # Check if the POST request has a file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Check if the file is an image
    if file and allowed_file(file.filename):
        # Read the image file
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Perform object detection
        results = model(image)

        # Extract the names and calories of detected objects
        detections = []
        for result in results:
            boxes = result[0].boxes.cpu().numpy()
            xyxy = boxes.xyxy[0]
            cls = boxes.cls[0]
            conf = round(float(boxes.conf[0]), 3)
            class_name = model.names[int(cls)]
            calories = food_calories.get(class_name)
            if calories is not None:
                detection_info = {'name': class_name, 'calories': calories}
                detections.append(detection_info)

        return jsonify({'detections': detections})

    return jsonify({'error': 'Unsupported file format'})


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
