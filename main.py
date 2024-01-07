# 'http://192.168.30.18:5001/video'

from super_gradients.common.object_names import Models
from super_gradients.training import models
import random
import cv2
import numpy as np
import torch
from imutils.video import VideoStream
from tqdm import tqdm

# 모델 초기화 및 로드
model = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")
model.cpu()

# 비디오 스트림 시작
vs = VideoStream("http://192.168.30.18:5001/video").start()

# 라벨 색상 및 이름 초기화
label_colors = None
names = None
first_frame = True

while True:
    frame = vs.read()
    if frame is None:
        break

    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    outputs = model.predict(frame, conf=0.4, iou=0.4)
    output = outputs[0]
    bboxes = output.prediction.bboxes_xyxy
    confs = output.prediction.confidence
    labels = output.prediction.labels
    class_names = output.class_names

    if first_frame:
        random.seed(0)
        labels = [int(l) for l in list(labels)]
        label_colors = [tuple([int(i) for i in random.choices(range(256), k=3)]) for _ in range(len(class_names))]
        names = [class_names[int(label)] for label in labels]
        first_frame = False

    for idx, bbox in enumerate(bboxes):
        bbox_left = int(bbox[0])
        bbox_top = int(bbox[1])
        bbox_right = int(bbox[2])
        bbox_bot = int(bbox[3])
        label_idx = labels[idx]

        #print(class_names[int(label_idx)], bbox_left, bbox_top, bbox_right, bbox_bot)
        # 객체 이름과 신뢰도 점수
        text = f"{class_names[int(label_idx)]} {confs[idx]:.2f}"

        # 텍스트 크기와 색상
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
        text_w, text_h = text_size
        color = label_colors[label_idx]
        print(text_size, text_w, text_h)
        # 객체 이름 및 신뢰도 점수 배경 사각형
        cv2.rectangle(frame, (bbox_left, bbox_top - text_h - 2), (bbox_left + text_w, bbox_top), color, -1)

        # 객체 이름 및 신뢰도 점수 텍스트
        cv2.putText(frame, text, (bbox_left, bbox_top - 2), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

        # 객체 바운딩 박스
        cv2.rectangle(frame, (bbox_left, bbox_top), (bbox_right, bbox_bot), color, 2)

    # 프레임 표시
    cv2.imshow('Frame', frame)

    # 'q'를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 정리 작업
vs.stop()
cv2.destroyAllWindows()
