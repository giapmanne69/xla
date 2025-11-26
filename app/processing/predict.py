from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2
import numpy as np

# 1. Cấu hình Model SAHI (Bọc lấy YOLO của bạn)
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov11', # Sahi hỗ trợ v8/v11 qua interface này
    model_path='app\clahe_with_negative_dataset\\best.pt', # File model của bạn
    confidence_threshold=0.4,
    device="cpu", # Hoặc 'cpu'
)

import os

# --- CẤU HÌNH ĐƯỜNG DẪN ---
SOURCE_DIR = "app\\test_data\images"  # Thư mục chứa ảnh gốc của bạn
OUTPUT_DIR = "app\\test-result-with-sahi\clahe-with-negative-data"         # Thư mục lưu ảnh kết quả

# Tạo thư mục kết quả nếu chưa có
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Các đuôi file ảnh chấp nhận
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')

# Lấy danh sách file ảnh
image_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(valid_extensions)]

print(f"🚀 Tìm thấy {len(image_files)} ảnh. Bắt đầu chạy SAHI...")

# --- VÒNG LẶP XỬ LÝ ---
for filename in image_files:
    image_path = os.path.join(SOURCE_DIR, filename)
    print(f"🔪 Đang xử lý: {filename}...")

    # 3. CHẠY PREDICT THEO CƠ CHẾ CẮT LÁT (SLICING)
    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=640, # Kích thước mảnh cắt (nhỏ giúp nhìn rõ vật thể xa)
        slice_width=416,
        overlap_height_ratio=0.2, # Chồng lấn 20% để không bị cắt đôi vật thể
        overlap_width_ratio=0.2,
        verbose=0 # Tắt log chi tiết cho đỡ rối mắt
    )

    # 4. Lưu kết quả
    # Tham số file_name giúp giữ nguyên tên gốc (vd: image1.jpg -> image1_vis.png)
    result.export_visuals(export_dir=OUTPUT_DIR, file_name=filename)

print(f"✅ Hoàn tất! Kiểm tra kết quả tại thư mục: {OUTPUT_DIR}")