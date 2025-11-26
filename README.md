# 🛵 Hệ Thống Phát Hiện Mũ Bảo Hiểm (Helmet Detection System)

> **Đề tài:** Ứng dụng Thị giác máy tính và Deep Learning (YOLOv11) để giám sát người tham gia giao thông.
> **Sinh viên thực hiện:** Nguyễn Thế Giáp - B22DCCN251

## 📖 Giới thiệu (Introduction)

Tại Việt Nam, tai nạn giao thông liên quan đến việc không đội mũ bảo hiểm vẫn là vấn đề nhức nhối. Dự án này xây dựng một hệ thống tự động phát hiện người đi xe máy không đội mũ bảo hiểm theo thời gian thực (Real-time) từ Camera giám sát.

Hệ thống được tối ưu hóa để chạy trên các thiết bị máy tính cá nhân thông thường (như Laptop dùng chip Intel Iris Xe) mà vẫn đảm bảo tốc độ cao nhờ sử dụng **YOLOv11 Nano** kết hợp với các kỹ thuật tiền xử lý ảnh nâng cao.

## 🧠 Tại sao chọn YOLOv11n? (Model Selection)

Trong dự án này, tôi quyết định lựa chọn **YOLOv11n (Nano)** làm mô hình lõi vì các lý do kỹ thuật sau:

1.  **Kiến trúc tối ưu (C3k2 Module):** YOLOv11 sử dụng kiến trúc module **C3k2**, cho phép tái sử dụng luồng thông tin hiệu quả mà không cần mạng lưới nơ-ron quá khổng lồ. Điều này giúp mô hình đạt tốc độ suy luận (inference) rất nhanh, phù hợp cho bài toán thời gian thực.
2.  **Cơ chế Augmentation mạnh mẽ:**
      * **Mosaic Augmentation:** Trong quá trình huấn luyện, mô hình tự động cắt ghép 4 bức ảnh ngẫu nhiên, thay đổi tỷ lệ và gộp thành 1. Điều này giúp mô hình học được các biến thể vật thể đa dạng hơn gấp nhiều lần.
      * **Biến đổi HSV:** Tự động thay đổi ngẫu nhiên 3 thông số Hue, Saturation và Value để tăng cường khả năng thích nghi với các điều kiện ánh sáng màu sắc khác nhau.
3.  **Tương thích với SAHI:** Mô hình này rất phù hợp với thuật toán **SAHI (Slicing Aided Hyper Inference)** – cho phép cắt 1 ảnh lớn thành nhiều mảnh nhỏ để dự đoán, giúp phát hiện các đối tượng nhỏ ở xa mà không làm tăng quá nhiều độ trễ.

## 🚀 Tính năng nổi bật (Key Features)

  * **Phát hiện thời gian thực 7 lớp đối tượng:**
      * `Driver with helmet` / `Driver without helmet` (Người lái)
      * `Passenger with helmet` / `Passenger without helmet` (Người ngồi sau)
      * `Bike` (Xe máy)
      * `Driver` / `Passenger` (Nhãn chung)
  * **Xử lý ánh sáng thông minh:** Tích hợp thuật toán **CLAHE trên không gian màu LAB**, giúp nhận diện tốt trong điều kiện lóa nắng hoặc thiếu sáng.
  * **Chống báo giả (Anti-False Positive):** Huấn luyện với kỹ thuật *Negative Mining* (200 ảnh nền rỗng như cây cối, đường vắng) để mô hình học cách không nhận diện nhầm.
  * **Giao diện Web:** Dashboard theo dõi qua Flask, kết nối trực tiếp Camera.
  * **Tối ưu phần cứng:** Hỗ trợ chạy tốt trên CPU/iGPU thông qua OpenVINO/ONNX Runtime.

## 🛠️ Phương pháp luận & Thuật toán (Methodology)

Dự án áp dụng kiến thức từ môn **Xử lý ảnh (Image Processing)** kết hợp **Deep Learning**. Tôi đã thử nghiệm 2 phương pháp tiền xử lý ảnh chính:

### 1\. Phương pháp Ảnh Lai Ghép (Hybrid Composite Image) - *Thử nghiệm*

Đây là phương pháp phá vỡ cấu trúc RGB truyền thống, ép 3 kênh đầu vào chứa 3 loại thông tin khác nhau nhằm ép mô hình học hình dạng hình học.

  * **Workflow:**
      * **Kênh 1 (Biên - Edge):** Sử dụng thuật toán **Canny** (Lọc Gaussian giảm nhiễu $\rightarrow$ Tính gradient Sobel $\rightarrow$ Làm mảnh nét $\rightarrow$ Lọc cạnh yếu) để lấy đường viền mũ.
      * **Kênh 2 (Chi tiết - Texture):** Sử dụng CLAHE trên ảnh xám để tăng độ nét bề mặt.
      * **Kênh 3 (Cường độ):** Giữ thông tin độ sáng tổng quát.
  * **Đánh giá:** Phương pháp này **không hiệu quả** với YOLOv11.
      * *Lý do:* YOLOv11n học rất tốt dựa trên màu sắc tự nhiên (ví dụ: nhận biết cây màu xanh, bầu trời màu lam). Việc dùng ảnh lai ghép làm mất hoàn toàn thông tin màu, cộng với việc thuật toán Canny thường tạo ra các đường nét đứt đoạn, khiến mô hình bị "rối".

### 2\. Phương pháp CLAHE trên không gian màu LAB - *Giải pháp đề xuất*

Đây là giải pháp được chọn để khắc phục nhược điểm ánh sáng phức tạp mà vẫn giữ nguyên màu sắc tự nhiên.

  * **Bản chất:** Kéo dãn độ tương phản cục bộ, cắt ngọn các phần nhiễu quá giới hạn và phân phối sang vùng khác.
  * **Workflow:**
    1.  Chuyển đổi không gian màu: **RGB $\rightarrow$ LAB**.
    2.  Tách kênh **L (Lightness)**. *Lý do chỉ chỉnh kênh L: Vì kênh này chứa thông tin sáng tối, không làm sai lệch màu sắc của vật thể.*
    3.  Áp dụng **CLAHE** lên kênh L: Chia lưới ảnh, tính histogram, cắt ngọn ngưỡng và nội suy song tuyến tính để xóa vết cắt.
    4.  Gộp lại kênh L (đã xử lý) với kênh A, B (giữ nguyên).
    5.  Chuyển ngược về **RGB** để đưa vào mô hình.

### 📊 So sánh hiệu quả thực nghiệm

| Tiêu chí | Ảnh Lai Ghép (Hybrid Composite) | CLAHE trên LAB (Được chọn) |
| :--- | :--- | :--- |
| **Thông tin màu sắc** | **Mất hoàn toàn** (Ảnh giả màu) | **Giữ nguyên** (Tự nhiên) |
| **Thông tin biên** | Rất mạnh (Canny Edge) | Trung bình (Tự nhiên) |
| **Tương thích YOLO** | Thấp (Mất đặc trưng màu pre-trained) | **Rất cao** |
| **Độ chính xác (mAP)** | Thấp (\< 0.5) | **Cao (\> 0.8)** |

## 📂 Cấu trúc dự án

```text
app/
├── dataset/
│   ├── clahe_dataset/    # Dataset đã áp dụng CLAHE (LAB)
│   ├── origin_dataset/   # Dataset ảnh gốc
│   └── rgb_dataset/      # Dataset ảnh lai ghép (Hybrid)
├── processing/
│   ├── predict.py        # Chạy dự đoán (Tích hợp SAHI)
│   ├── processing_clahe.py # Script xử lý ảnh CLAHE
│   ├── processing_rgb.py   # Script xử lý ảnh Hybrid
│   └── train.py          # Script huấn luyện
├── static/               # Tài nguyên Web (Kết quả, Ảnh upload)
├── templates/            # Giao diện Frontend
├── test_data/            # Dữ liệu test thủ công
├── test-result/          # Kết quả test local
├── app.py                # Main Web App
├── README.md
└── yolo11n.pt            # File trọng số mô hình (Pre-trained)
```

## ⚙️ Cài đặt & Sử dụng

### 1\. Yêu cầu hệ thống

  * Python 3.8 trở lên.
  * Khuyên dùng môi trường ảo (Virtual Environment).

### 2\. Cài đặt thư viện

```bash
pip install ultralytics flask opencv-python onnx onnxruntime
```

### 3\. Cấu hình Model

Đưa đường dẫn file `best.pt` (kết quả sau khi train theo phương pháp CLAHE) vào biến cấu hình trong file `app.py`.

### 4\. Chạy ứng dụng Web (Demo)

Kết nối Camera và chạy lệnh sau:

```bash
python app.py
```

Truy cập trình duyệt tại địa chỉ: `http://localhost:5000`

### 5\. Huấn luyện lại (Training)

Nếu muốn tự train lại (khuyên dùng GPU như Google Colab T4):

```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=960,    
    batch=16,
    device=0      # Sử dụng GPU
)
```

## 🔗 Tham khảo (References)

  * *Giáo trình Xử lý ảnh (Học viện Công nghệ Bưu chính Viễn thông)* - Lý thuyết về Lọc không gian, Biến đổi độ xám và Phát hiện biên.
  * *Ultralytics YOLOv11 Documentation*.
  * *Kaggle Helmet Detection Dataset*.

-----

© 2025 Project by **Nguyễn Thế Giáp** (B22DCCN251)