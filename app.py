import os
import cv2
import time
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from ultralytics import YOLO
from pathlib import Path

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================
app = Flask(__name__)

# Thư mục lưu trữ
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# --- CẤU HÌNH MODEL ---
# Bạn hãy đổi tên file này thành tên file thực tế của bạn (ví dụ: best.pt)
# Đảm bảo file .pt nằm cùng thư mục với file app.py này
MODEL_PATH = 'app\dataset\clahe_dataset\\best.pt' 

print(f"⏳ Đang tải mô hình PyTorch từ {MODEL_PATH}...")
try:
    model = YOLO(MODEL_PATH)
    print("✅ Mô hình đã sẵn sàng!")
except Exception as e:
    print(f"❌ LỖI: Không tìm thấy file {MODEL_PATH}. Hãy copy file best.pt vào đây!")
    exit()

# ==========================================
# 2. THUẬT TOÁN TIỀN XỬ LÝ (CLAHE LAB)
# ==========================================
def apply_clahe_lab(img):
    """
    Kỹ thuật tăng cường ảnh: CLAHE trên không gian màu LAB.
    """
    try:
        # B1: Chuyển từ BGR sang LAB
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # B2: Tách các kênh
        l, a, b = cv2.split(lab)
        
        # B3: Áp dụng CLAHE lên kênh L (Lightness)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # B4: Gộp lại
        limg = cv2.merge((cl, a, b))
        
        # B5: Chuyển ngược lại về BGR
        final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        return final_img
    except Exception as e:
        print(f"Lỗi xử lý ảnh: {e}")
        return img 

# ==========================================
# 3. CÁC HÀM XỬ LÝ
# ==========================================

def process_image(img_path, save_path):
    # 1. Đọc ảnh
    original_img = cv2.imread(img_path)
    
    # 2. Tiền xử lý (CLAHE)
    processed_img = apply_clahe_lab(original_img)

    # 3. Dự đoán
    # conf=0.5: Chỉ hiện những cái chắc chắn trên 50%
    results = model.predict(processed_img, imgsz=640, conf=0.5, verbose=False)
    
    # 4. Vẽ và Lưu
    final_img = results[0].plot()
    cv2.imwrite(save_path, final_img)


def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Codec mp4v
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Tiền xử lý
        processed_frame = apply_clahe_lab(frame)

        # 2. Tracking (Dùng cho video để mượt hơn)
        results = model.track(processed_frame, imgsz=640, conf=0.5, persist=True, verbose=False)
        
        # 3. Vẽ và Ghi
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()

# ==========================================
# 4. ROUTES & MAIN
# ==========================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'Không tìm thấy file'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn file'}), 400

    filename = file.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    is_image = filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    timestamp = int(time.time())
    
    if is_image:
        result_filename = f"res_{timestamp}_{filename}"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        process_image(file_path, result_path)
        ftype = 'image'
    else:
        result_filename = f"res_{timestamp}_{Path(filename).stem}.mp4"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        process_video(file_path, result_path)
        ftype = 'video'

    return jsonify({
        'status': 'success',
        'type': ftype,
        'result_url': url_for('static', filename=f'results/{result_filename}')
    })

if __name__ == '__main__':
    print("🌍 Web App đang chạy tại: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)