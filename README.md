# yolov8
# make_readme.py
import os

readme_content = """# 元智大學校園建築物物件偵測系統 (YZU Campus Building Detection)

本專案基於 **YOLOv8-Nano** 輕量化架構，開發了一套能在純本機 CPU 環境下穩定運行的校園建築物自動辨識系統。專案內容包含從零構建自訂大樓資料集（以元智五館、三館為核心目標），並完成模型的微調訓練（Fine-tuning）與系統後處理優化。

## 🛠️ 1. 環境配置 (Environment Setup)

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
