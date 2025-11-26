import cv2
import os
import numpy as np
from tqdm import tqdm

def create_composite_image(img):
    # 1. Tạo Kênh R: Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Tạo Kênh G: CLAHE (Tăng tương phản thông minh)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 3. Tạo Kênh B: Canny Edge (Phát hiện biên)
    # Dùng threshold trung bình để lấy nét chính
    edges = cv2.Canny(gray, 100, 200)
    
    # Gộp 3 kênh lại thành 1 ảnh (Stacking)
    # Thứ tự OpenCV là B-G-R
    composite = cv2.merge([edges, enhanced, gray])
    
    return composite

def process_dataset(source_dir, dest_dir, target_size=(416, 416)):
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    print(f"🚀 Bắt đầu tạo dataset Lai ghép (Composite)...")
    
    for root, dirs, files in os.walk(source_dir):
        relative_path = os.path.relpath(root, source_dir)
        target_path = os.path.join(dest_dir, relative_path)
        
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        for file in tqdm(files, desc=f"Folder: {relative_path}", leave=False):
            source_file = os.path.join(root, file)
            target_file = os.path.join(target_path, file)
            file_ext = os.path.splitext(file)[1].lower()

            if file_ext in img_extensions:
                try:
                    img = cv2.imread(source_file)
                    if img is not None:
                        # Resize trước cho nhẹ
                        img_resized = cv2.resize(img, target_size)
                        
                        # Tạo ảnh lai ghép
                        final_img = create_composite_image(img_resized)
                        
                        # Lưu ảnh
                        cv2.imwrite(target_file, final_img)
                except Exception as e:
                    print(f"Lỗi: {e}")
            else:
                # Copy file label, yaml...
                import shutil
                shutil.copy2(source_file, target_file)

# --- CẤU HÌNH ---
INPUT_DIR_TRAIN = "datasets/helmet_original"  # Ảnh màu gốc
OUTPUT_DIR = "datasets/helmet_composite" # Folder mới
INPUT_DIR_VAL = "datasets/helmet_original"  # Ảnh màu gốc

if __name__ == "__main__":
    process_dataset(INPUT_DIR_TRAIN, OUTPUT_DIR)
    process_dataset(INPUT_DIR_VAL, OUTPUT_DIR)