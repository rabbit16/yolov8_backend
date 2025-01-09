import base64
import json

import cv2
import numpy as np
from ultralytics import YOLO

# 加载 YOLOv8 模型
model = YOLO("yolov8n.pt")  # 确保模型文件路径正确
model2 = YOLO('./weights/best.pt')  #  ./weights/best.pt
# 特征提取方法（可以使用颜色直方图或其他方法）
def extract_features(image):
    # 这里简单地使用颜色直方图作为示例，你可以替换成其他更复杂的方法
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# 计算相似度
def cosine_similarity(feature1, feature2):
    return np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))

# 主函数
def detect_people(image_data, cache, employee_number):
    # 过往人物特征记录
    previous_features = json.loads(cache)  # 加载缓存列表
    new_people = []
    # frame = cv2.imread(image_data)
    image_message = base64.b64decode(image_data)
    nparr = np.frombuffer(image_message, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    output_file_path = 'output_image.jpg'  # 设置输出文件的路径和文件名
    cv2.imwrite(output_file_path, frame)
    results = model.predict(frame)  # 目标检测
    # 提取检测结果
    detections = results[0].boxes.xyxy  # 取得位置信息
    confidences = results[0].boxes.conf  # 取得置信度
    class_ids = results[0].boxes.cls  # 取得类别id
    for i, (x1, y1, x2, y2) in enumerate(detections):
        if confidences[i] > 0.5 and int(class_ids[i]) == 0:  # class_id 0 for person
            person_frame = frame[int(y1):int(y2), int(x1):int(x2)]  # 提取人物图像
            features = extract_features(person_frame)
            if not previous_features:
                previous_features.append(features.tolist())
                new_people.append([(int(x1)+int(x2))/2, (int(y1)+int(y2))/2])
                print("新人物记录！")
                continue
            # 检查相似度
            flag = 0
            for prev_feat in previous_features:
                sim = cosine_similarity(features, prev_feat)
                if sim > 0.7:  # 设置相似度阈值
                    flag = 1
                    break
            if flag == 0:
                previous_features.append(features.tolist())
                new_people.append([(int(x1) + int(x2)) / 2, (int(y1) + int(y2)) / 2])
                print("新人物记录！")
    return True, previous_features, new_people


def detect_rubbish(image_data, cache, employee_number):
    # 过往人物特征记录
    previous_features = json.loads(cache)  # 加载缓存列表
    new_people = []
    ner_rubbish_pic = []
    # frame = cv2.imread(image_data)
    image_message = base64.b64decode(image_data)
    nparr = np.frombuffer(image_message, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    output_file_path = 'output_image2.jpg'  # 设置输出文件的路径和文件名
    cv2.imwrite(output_file_path, frame)
    results = model2.predict(frame)  # 目标检测
    # 提取检测结果
    detections = results[0].boxes.xyxy  # 取得位置信息
    confidences = results[0].boxes.conf  # 取得置信度
    class_ids = results[0].boxes.cls  # 取得类别id
    for i, (x1, y1, x2, y2) in enumerate(detections):
        if confidences[i] > 0.7:  # class_id 0 for person
            person_frame = frame[int(y1):int(y2), int(x1):int(x2)]  # 提取垃圾图像
            features = extract_features(person_frame)
            if not previous_features:
                previous_features.append(features.tolist())
                new_people.append([(int(x1)+int(x2))/2, (int(y1)+int(y2))/2])
                ner_rubbish_pic.append(person_frame.tolist())
                print("新人物记录！")
                continue
            # 检查相似度
            flag = 0
            for prev_feat in previous_features:
                sim = cosine_similarity(features, prev_feat)
                if sim > 0.7:  # 设置相似度阈值
                    flag = 1
                    break
            if flag == 0:
                previous_features.append(features.tolist())
                ner_rubbish_pic.append(person_frame.tolist())
                new_people.append([(int(x1) + int(x2)) / 2, (int(y1) + int(y2)) / 2])
                print("有新的垃圾产生了！")
    return True, previous_features, new_people, ner_rubbish_pic
if __name__ == "__main__":
    # 输入你要处理的图片路径
    image_path = "../rubbish_person.jpeg"  # 替换为你的图片路径
    main(image_path)
