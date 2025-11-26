import cv2
import os
import numpy as np
from tqdm import tqdm

def apply_clahe_lab(img):
    """
    Tăng cường tương phản mà vẫn giữ nguyên màu sắc tự nhiên.
    """
    # 1. Chuyển từ BGR (OpenCV) sang LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # 2. Tách các kênh: L (Sáng), A (Màu), B (Màu)
    l, a, b = cv2.split(lab)
    
    # 3. Áp dụng CLAHE chỉ trên kênh L (Lightness)
    # clipLimit: ngưỡng tương phản (2.0 là vừa, cao quá sẽ bị nhiễu hạt)
    # tileGridSize: kích thước ô cục bộ
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # 4. Gộp lại (Dùng kênh L đã xử lý + kênh A, B gốc)
    limg = cv2.merge((cl, a, b))
    
    # 5. Chuyển ngược lại về BGR để YOLO hiểu
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return final_img

def process_dataset_enhancement(source_dir, dest_dir, target_size=(640, 640)):
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    print(f"🚀 Bắt đầu tăng cường dữ liệu (LAB Enhancement)...")
    print(f"Output: {dest_dir}")
    
    for root, dirs, files in os.walk(source_dir):
        relative_path = os.path.relpath(root, source_dir)
        target_path = os.path.join(dest_dir, relative_path)
        
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        for file in tqdm(files, desc=f"Xử lý {relative_path}", leave=False):
            source_file = os.path.join(root, file)
            target_file = os.path.join(target_path, file)
            file_ext = os.path.splitext(file)[1].lower()

            if file_ext in img_extensions:
                try:
                    img = cv2.imread(source_file)
                    if img is not None:
                        # 1. Resize (Nên dùng 640 để mAP cao, 416 hơi thấp cho mũ nhỏ)
                        img_resized = cv2.resize(img, target_size)
                        
                        # 2. Xử lý nâng cao
                        final_img = apply_clahe_lab(img_resized)
                        
                        # 3. Lưu ảnh
                        cv2.imwrite(target_file, final_img)
                except Exception as e:
                    print(f"Lỗi: {e}")
            else:
                # Copy label giữ nguyên
                import shutil
                shutil.copy2(source_file, target_file)

# --- CẤU HÌNH ---
INPUT_DIR = "app\origin_dataset\\train"     # Folder ảnh gốc
OUTPUT_DIR = "app\clahe_with_negative_dataset\\train"    # Folder mới chứa ảnh xịn

if __name__ == "__main__":
    process_dataset_enhancement(INPUT_DIR, OUTPUT_DIR, target_size=(640, 640))