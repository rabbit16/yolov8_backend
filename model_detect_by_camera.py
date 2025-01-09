import time
from os import times

import cv2
from ultralytics import YOLOv10, YOLO

# 加载预训练模型
# model = YOLOv10.from_pretrained('jameslahm/yolov10m')
# model = YOLOv10.from_pretrained('/home/hljiang/PycharmProjects/yolov10/weights/best.pt')
model = YOLO("/home/hljiang/PycharmProjects/yolov10/weights/last.pt")
# 打开摄像头
cap = cv2.VideoCapture(2)  # 0 为默认摄像头，若有多个摄像头可按需调整

while True:
    # 捕获摄像头的一帧
    ret, frame = cap.read()
    if not ret:
        print("无法获取视频帧，程序退出。")
        break

    # 进行预测
    results = model.predict(source=frame, save=False, conf=0.8)

    # 获取结果并在图像上绘制边界框
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 获取边界框的信息
            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]  # 边界框坐标
            conf = box.conf.cpu().numpy()[0]  # 置信度
            cls = box.cls.cpu().numpy()[0]  # 类别

            # 在图像上绘制边界框和标签
            label = f'Class: {result.names[int(cls)]}, Conf: {conf:.2f}'
            print(label)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 显示结果
    cv2.imshow('YOLOv10 Object Detection', frame)
    time.sleep(0.1)
    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头和关闭窗口
cap.release()
cv2.destroyAllWindows()
