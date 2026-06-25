from ultralytics import YOLO


def main():
    # 1. 載入 YOLOv8 官方的預訓練微型模型 (yolov8n.pt 檔案小、在 CPU 上跑得動)
    model = YOLO("yolov8n.pt")

    print("🚀 YOLOv8 訓練正式開始...")
    
    # 2. 開始訓練
    model.train(
        data="datasets/dataset/data.yaml",  # 指向你的設定檔
        epochs=50,                                   # 訓練 50 輪
        imgsz=640,                                   # 圖片縮放大小
        batch=4,                                     # CPU 訓練建議設 4，記憶體比較不會爆
        device="cpu",                                # 強制指定使用 CPU，絕對不會噴 DLL 錯誤
        workers=0                                    # Windows 系統設 0  最安全，避免多線程卡死
    )
    
    print("🎉 恭喜！模型訓練完成！")


if __name__ == "__main__":
    main()