# yolov8

本專案基於 **YOLOv8-Nano** 輕量化架構，開發了一套能在純本機 CPU 環境下穩定運行的校園建築物自動辨識系統。專案內容包含從零構建自訂大樓資料集（以元智五館、三館為核心目標），並完成模型的微調訓練（Fine-tuning）與系統後處理優化。

## 1. 環境配置 (Environment Setup)

本專案運行於 Windows 10/11 的 Miniconda 虛擬環境中。為避免 PyTorch 在 Windows CPU 上常見的 `shm.dll` 依賴庫衝突與反向傳播錯誤，請使用以下步驟建立乾淨的環境：

```bash
# 建立 Python 3.10 獨立環境
conda create -n yolo_env python=3.10 -y

# 啟用環境
conda activate yolo_env

# 用 Pip 安裝官方獨立版 PyTorch (避開 Conda 底層 DLL 衝突的關鍵)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)

# 安裝 YOLOv8 官方核心套件
pip install ultralytics
```

## 2. 資料集準備 (Dataset Structure)

```text
├── train_yolo.py         # 訓練腳本
├── test_yolo.py          # 測試/推論腳本
└── datasets/
    └── dataset/ 
        ├── data.yaml     # 記錄類別名稱與 train/val 相對路徑的設定檔
        ├── train/
        │   ├── images/   # 訓練用大樓圖片 (.jpg / .png)
        │   └── labels/   # 訓練用 YOLO 標註文字檔 (.txt)
        └── val/
            ├── images/   # 驗證用大樓圖片
            └── labels/   # 驗證用 YOLO 標註文字檔
```
## 3.訓練模型 train_yolo.py
### 程式碼
```bash
from ultralytics import YOLO

def main():
    # 載入官方預訓練的輕量化微型模型
    model = YOLO("yolov8n.pt")

    print("🚀 YOLOv8 校園建築物模型訓練開始...")
    
    # 開始訓練
    model.train(
        data="datasets/building_dataset/data.yaml",  # 指定資料集設定檔
        epochs=50,                                   # 訓練輪數
        imgsz=640,                                   # 圖片輸入解析度
        batch=4,                                     # CPU 記憶體安全批次量
        device="cpu",                                # 強制指定使用 CPU 運行
        workers=0                                    # Windows 系統設 0，防止多線程卡死
    )
    
    print("🎉 模型訓練完成！")

if __name__ == "__main__":
    main()
```
### 執行程式
```bash
python test_yolo.py
```

## 4.驗證模型 test_yolo.py
### 程式碼
from ultralytics import YOLO
```bash
# 1. 載入親手訓練好的最強權重
model = YOLO("runs/detect/train/weights/best.pt")

# 2. 進行物件偵測推論
# 透過加入 conf 與 iou 參數，能有效合併重疊框並過濾低信心值的背景雜訊
results = model(
    r"C:\Users\T7USER\Desktop\test_building.jpg",  # 替換為你的本機圖片絕對路徑或相對路徑
    conf=0.6,                                      # 信心值門檻：大於 60% 才顯示
    iou=0.45                                       # NMS 門檻：抑制並合併重疊的雜訊框
) 

# 3. 彈出視窗展示偵測結果
results[0].show()

# 4. 將畫好精準框的結果圖片儲存到本地目錄
results[0].save(filename="my_yolo_result.jpg")
print("偵測完成！結果已儲存為 my_yolo_result.jpg")
```
### 執行程式
```bash
python test_yolo.py
```
## 預期結果

