from ultralytics import YOLO
# model = YOLO('yolov8m.pt')  # load an official model
model = YOLO('./best2.pt')  # load a custom model

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


results = model('parata.jpg')  # predict on an image

for result in results:
    # print(result)
    boxes = result[0].boxes.cpu().numpy()
    xyxy = boxes.xyxy[0]
    cls = boxes.cls[0]
    conf = round(float(boxes.conf[0]), 3)
    x_min, y_min, x_max, y_max = map(int, xyxy)
    text = f'{model.names[int(cls)]}: {conf}'
    # print(text.)
    print(model.names[int(cls)])
    print(food_calories[model.names[int(cls)]], 'Cal')
