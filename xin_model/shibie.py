import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 加载模型
model = load_model('cat_dog')

# 打开摄像头
cap = cv2.VideoCapture(0)  # 0代表默认摄像头，如果有多个摄像头，请根据需要选择摄像头编号

while True:
    ret, frame = cap.read()  # 读取摄像头图像

    # 对图像进行预处理
    preprocessed_frame = cv2.resize(frame, (224, 224))
    preprocessed_frame = img_to_array(preprocessed_frame)
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)
    preprocessed_frame = preprocess_input(preprocessed_frame)

    # 进行预测
    predictions = model.predict(preprocessed_frame)
    class_index = np.argmax(predictions[0])
    class_labels = ['cat', 'dog']  # 类别标签
    predicted_label = class_labels[class_index]

    # 在图像上绘制识别结果
    cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Real-time Object Detection', frame)

    # 按下'q'退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()