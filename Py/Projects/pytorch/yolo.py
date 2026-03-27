# Py\Projects\pytorch
from ultralytics import YOLO
import cv2

# 加载预训练模型（自带80类物体识别）
model = YOLO("yolov8n.pt")

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # 实时检测
    results = model(frame)
    # 把结果画在画面上
    annotatrame = results[0].plot()
    cv2.imshow("YOLO 实时检测", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()