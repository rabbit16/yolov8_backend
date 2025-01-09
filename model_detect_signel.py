import os

from ultralytics import YOLOv10, YOLO

# 加载预训练模型
# model = YOLOv10.from_pretrained('jameslahm/yolov10m')
model = YOLO("./weights/best.pt")
# 要进行预测的图片
source = '10.jpg'
# source = 'cat.png'
a = model.predict(source=source, save=True, conf=0.5)
# 输出预测结果
for result in a:
    boxes = result.boxes
    for box in boxes:
        # 获取边界框的信息
        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]  # 边界框坐标
        print(x1, y1, x2, y2)
        conf = box.conf.cpu().numpy()[0]  # 置信度
        cls = box.cls.cpu().numpy()[0]  # 类别
        # 在图像上绘制边界框和标签
        label = f'Class: {result.names[int(cls)]}, Conf: {conf:.2f}'
        print(label)