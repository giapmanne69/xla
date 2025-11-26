import os
import cv2
import time
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from pathlib import Path

# --- THƯ VIỆN YOLO & SAHI ---
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Đường dẫn model của bạn (ưu tiên .pt vì SAHI hỗ trợ .pt tốt hơn onnx)
MODEL_PATH = 'best.pt' # Hãy chắc chắn file này nằm cùng thư mục
# MODEL_PATH = r"D:\Duong\Dan\Cua\Ban\best.pt" 

print(f"⏳ Đang tải mô hình từ {MODEL_PATH}...")

# --- LOAD MODEL 1: CHO VIDEO (YOLO CHUẨN - TỐC ĐỘ CAO) ---
try:
    yolo_model = YOLO(MODEL_PATH)
    print("✅ Model YOLO chuẩn (cho Video): Sẵn sàng!")
except Exception as e:
    print(f"❌ Lỗi load YOLO: {e}")
    exit()

# --- LOAD MODEL 2: CHO ẢNH (SAHI - ĐỘ CHÍNH XÁC CỰC CAO) ---
try:
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8', # SAHI dùng chuẩn v8 tương thích v11
        model_path=MODEL_PATH,
        confidence_threshold=0.4, # Ngưỡng tự tin
        device="cuda:0" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    )
    print("✅ Model SAHI (cho Ảnh): Sẵn sàng!")
except Exception as e:
    print(f"❌ Lỗi load SAHI: {e}")
    # Nếu lỗi SAHI thì ta sẽ fallback về dùng yolo_model thường
    sahi_model = None

# ==========================================
# 2. THUẬT TOÁN TIỀN XỬ LÝ (CLAHE LAB)
# ==========================================
def apply_clahe_lab(img):
    """
    Tăng cường tương phản thông minh: Giữ màu, tăng nét vùng tối.
    """
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final_img
    except:
        return img

# ==========================================
# 3. XỬ LÝ ẢNH (DÙNG SAHI + CLAHE)
# ==========================================
def process_image_with_sahi(img_path, save_path):
    # 1. Đọc ảnh
    original_img = cv2.imread(img_path)
    
    # 2. Tiền xử lý (CLAHE)
    processed_img = apply_clahe_lab(original_img)
    
    # 3. Dự đoán bằng SAHI (Cắt lát)
    if sahi_model:
        # SAHI tự động cắt ảnh thành các mảnh nhỏ (slice) để soi vật thể nhỏ
        result = get_sliced_prediction(
            processed_img,
            sahi_model,
            slice_height=320,  # Kích thước mỗi mảnh cắt (càng nhỏ càng soi kỹ)
            slice_width=320,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            verbose=0
        )
        
        # 4. Vẽ kết quả lên ảnh
        # SAHI có hàm visualize riêng, ta xuất ra numpy array để lưu bằng cv2
        visualization_result = visualize_object_predictions(
            processed_img,
            object_prediction_list=result.object_prediction_list,
            rect_th=2,
            text_size=0.6,
            text_th=2
        )
        final_img = visualization_result["image"]
        # SAHI visualize trả về RGB, cần convert về BGR để OpenCV lưu đúng màu
        final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
        
    else:
        # Fallback: Nếu không load được SAHI thì dùng YOLO thường
        print("⚠️ Đang dùng YOLO thường cho ảnh (Do SAHI chưa load)")
        res = yolo_model.predict(processed_img, imgsz=640, conf=0.5)
        final_img = res[0].plot()

    # 5. Lưu ảnh
    cv2.imwrite(save_path, final_img)

# ==========================================
# 4. XỬ LÝ VIDEO (DÙNG YOLO TRACK + CLAHE)
# ==========================================
def process_video_tracking(video_path, output_path):
    """
    Video dùng YOLO chuẩn để đảm bảo FPS, không dùng SAHI vì sẽ rất lag.
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. CLAHE
        processed_frame = apply_clahe_lab(frame)

        # 2. Tracking (YOLOv11 Standard)
        # Tăng imgsz lên 1280 (nếu máy chịu nổi) để bù đắp việc không dùng SAHI
        results = yolo_model.track(processed_frame, imgsz=640, conf=0.5, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        # 3. Vẽ & Ghi
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()

# ==========================================
# 5. ROUTES
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
        # GỌI HÀM SAHI CHO ẢNH
        process_image_with_sahi(file_path, result_path)
        ftype = 'image'
    else:
        result_filename = f"res_{timestamp}_{Path(filename).stem}.mp4"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        # GỌI HÀM TRACKING CHO VIDEO
        process_video_tracking(file_path, result_path)
        ftype = 'video'

    return jsonify({
        'status': 'success',
        'type': ftype,
        'result_url': url_for('static', filename=f'results/{result_filename}')
    })

if __name__ == '__main__':
    print("🌍 Web App SAHI đang chạy tại: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)