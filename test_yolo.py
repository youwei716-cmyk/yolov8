from ultralytics import YOLO

# 載入訓練好的模型
model = YOLO("runs/detect/train/weights/best.pt")

results = model(
    "29.jpg",
    conf=0.6,     # 信心值必須大於 60% 才顯示（直接過濾掉 0.33, 0.37, 0.51 的雜訊框）
    iou=0.45      # 非極大值抑制 (NMS) 門檻，數字越低，越會強制把重疊的框合併成一個
) 

results[0].show()

results[0].save(filename="my_yolo_result.jpg")