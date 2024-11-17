import requests
import json


def detect_objects(image_path):
    url = 'http://localhost:5000/detect_objects'
    files = {'file': open(image_path, 'rb')}
    response = requests.post(url, files=files)
    if response.status_code == 200:
        data = json.loads(response.text)
        if 'detections' in data:
            print("Detected objects:")
            for detection in data['detections']:
                print(detection)
        else:
            print("No objects detected.")
    else:
        print("Error:", response.text)


if __name__ == "__main__":
    detect_objects('image2.jpg')
