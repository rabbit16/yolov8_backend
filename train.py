from ultralytics import YOLO

model = YOLO('/home/hljiang/PycharmProjects/yolov10/weights/path with spaces/yolov8n.pt')
# If you want to finetune the model with pretrained weights, you could load the
# pretrained weights like below
# model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
# or
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
# model = YOLOv10('yolov10{n/s/m/b/l/x}.pt')

model.train(data='/home/hljiang/PycharmProjects/yolov10/my_dataset/my_train.yaml', epochs=300, batch=32, imgsz=640)