# 🛵 Hệ Thống Phát Hiện Mũ Bảo Hiểm (Helmet Detection System)

> **Đề tài:** Ứng dụng Thị giác máy tính và Deep Learning (YOLOv11) để giám sát người tham gia giao thông.

## 📖 Giới thiệu (Introduction)

Tại Việt Nam, tai nạn giao thông liên quan đến việc không đội mũ bảo hiểm vẫn là vấn đề nhức nhối. Dự án này xây dựng một hệ thống tự động phát hiện người đi xe máy không đội mũ bảo hiểm theo thời gian thực (Real-time) từ Camera giám sát.

Hệ thống được tối ưu hóa để chạy trên các thiết bị máy tính cá nhân thông thường (như Laptop dùng chip Intel Iris Xe) mà vẫn đảm bảo tốc độ cao nhờ sử dụng **YOLOv11 Nano** và định dạng **ONNX**. Ngoài ra, hệ thống còn áp dụng cả SAHI - thuật toán cắt nhỏ ảnh và đưa vào YOLO để soi thật kỹ từng ảnh.

## 🚀 Tính năng nổi bật (Key Features)

  * **Phát hiện thời gian thực:** Nhận diện 7 lớp:
      - Driver with helmet: Người lái xe có đội mũ
      - Driver without helmet: Người lái xe không đội mũ
      - Bike: Xe
      - Driver: Người trên xe nói chung
      - Passenger with helmet: Người ngồi xe có đội mũ
      - Passenger without helmet: Người ngồi xe không đội mũ
      - Passenger: Người ngồi xe nói chung
  * **Xử lý ánh sáng thông minh:** Tích hợp thuật toán **CLAHE trên không gian màu LAB** giúp nhận diện tốt trong điều kiện lóa nắng hoặc thiếu sáng.
  * **Chống báo giả (Anti-False Positive):** Mô hình được huấn luyện với kỹ thuật *Negative Mining* (ảnh nền rỗng) để tránh nhận diện nhầm cây cối/vật thể lạ thành xe.
  * **Giao diện Web:** Tích hợp Dashboard theo dõi qua Web (Flask) kết nối trực tiếp với Camera.
  * **Tối ưu phần cứng:** Hỗ trợ chạy trên CPU/iGPU thông qua OpenVINO/ONNX Runtime.

## 🛠️ Công nghệ & Thuật toán (Methodology)

Dự án áp dụng các kiến thức từ môn học **Xử lý ảnh (Image Processing)** kết hợp với **Deep Learning**:

### 1\. Mô hình lõi: YOLOv11n

  * Sử dụng kiến trúc YOLOv11 Nano (nhẹ nhất) với các khối **C3k2** giúp trích xuất đặc trưng hiệu quả.
  * Cơ chế: Mạng Nơ-ron Tích chập (CNN) tự động học các bộ lọc không gian (Spatial Filters) để phát hiện biên và hình dạng vật thể.

### 2\. Kỹ thuật Tiền xử lý ảnh (Preprocessing)

Trong quá trình nghiên cứu, tôi đã thử nghiệm các phương pháp xử lý ảnh khác nhau để tìm ra giải pháp tối ưu nhất:

#### A. Phương pháp Ảnh Lai Ghép (Hybrid Composite Image) - *Thử nghiệm*

Đây là phương pháp tận dụng 3 kênh đầu vào để chứa 3 loại thông tin khác nhau thay vì màu sắc RGB thông thường:

  * **Kênh 1 (Biên - Edge):** Sử dụng thuật toán **Canny** để lấy đường viền mũ.
  * **Kênh 2 (Chi tiết - Texture):** Sử dụng **CLAHE** để tăng độ nét bề mặt.
  * **Kênh 3 (Gốc - Grayscale):** Giữ thông tin độ sáng tổng quát.
  * *Mục đích:* Ép mô hình học hình dạng hình học của mũ bảo hiểm.

#### B. Phương pháp CLAHE trên không gian màu LAB - *Giải pháp đề xuất*

Đây là giải pháp khắc phục nhược điểm của ảnh RGB thông thường khi gặp ánh sáng phức tạp.

  * **Quy trình:**
    1.  Chuyển đổi không gian màu: **RGB $\rightarrow$ LAB**.
    2.  Tách kênh **L (Lightness)** và áp dụng **CLAHE** (Contrast Limited Adaptive Histogram Equalization).
    3.  Gộp lại kênh L (đã xử lý) với kênh A, B (giữ nguyên màu sắc).
    4.  Chuyển ngược về **RGB** để đưa vào mô hình.

### 3\. Dữ liệu (Dataset)

  * **Nguồn:** Kaggle (Andrewmvd/helmet-detection) + Ảnh tự thu thập.
  * **Tổng số lượng:** \~400 ảnh.
  * **Negative Samples:** 200 ảnh đường phố vắng/cây cối (Labels rỗng) để giảm tỷ lệ dương tính giả (False Positives).

## 📊 So sánh hiệu quả (Comparison)

Dưới đây là bảng đánh giá thực nghiệm giữa phương pháp Ảnh Lai Ghép và phương pháp CLAHE trên LAB:

| Tiêu chí | Ảnh Lai Ghép (Hybrid Composite) | CLAHE trên LAB (Proposed) |
| :--- | :--- | :--- |
| **Thông tin màu sắc** | **Mất hoàn toàn** (Ảnh giả màu) | **Giữ nguyên** (Tự nhiên) |
| **Thông tin biên** | Rất mạnh (Canny Edge) | Trung bình (Tự nhiên) |
| **Khả năng chống lóa** | Tốt | **Rất tốt** |
| **Tương thích YOLO** | Thấp (Do model pre-trained học trên ảnh màu) | **Rất cao** |
| **Độ chính xác (mAP)** | Thấp (\< 0.5) | **Cao (\> 0.8)** |
| **Kết luận** | Chỉ phù hợp bài toán hình học đơn giản | **Tối ưu cho bài toán thực tế** |

**Quyết định:** Dự án lựa chọn phương pháp **CLAHE trên LAB** kết hợp với tăng độ phân giải ảnh đầu vào để đạt hiệu quả cao nhất.

## 📂 Cấu trúc dự án

```text
app/
├── dataset/
│   ├── clahe_dataset/
│   ├── origin_dataset/
│   └── rgb_dataset/
├── processing/
│   ├── predict.py
│   ├── processing_clahe.py
│   ├── processing_rgb.py
│   └── train.py
├── static/
│   ├── results/
│   └── uploads/
├── templates/
├── test_data/
├── test-result/
│   ├── clahe/
│   ├── origin/
│   └── rgb/
├── app.py
├── README.md
└── yolo11n.pt

```
## 📂 Giải thích cấu trúc thư mục

### 1. `dataset/` (Dữ liệu)
Chứa 3 bộ dataset phục vụ cho quá trình huấn luyện và kiểm thử:
- **`clahe_dataset/`**: Bộ dữ liệu đã được áp dụng phương pháp cân bằng lược đồ xám (CLAHE) trên không gian màu LAB nhằm tăng cường độ tương phản.
- **`origin_dataset/`**: Bộ dữ liệu ảnh gốc chưa qua xử lý.
- **`rgb_dataset/`**: Bộ dữ liệu áp dụng kỹ thuật ảnh ghép lai (Hybrid composite).

### 2. `processing/` (Xử lý logic)
Chứa các mã nguồn Python chính:
- **`predict.py`**: Hàm dự đoán chạy trực tiếp (local). Tích hợp kỹ thuật **SAHI** (Slicing Aided Hyper Inference) để cắt ảnh, giúp tăng độ chính xác khi nhận diện vật thể nhỏ.
- **`processing_clahe.py`**: Script chuyển đổi từ ảnh gốc sang ảnh xử lý theo kỹ thuật CLAHE (LAB).
- **`processing_rgb.py`**: Script chuyển đổi từ ảnh gốc sang ảnh xử lý theo kỹ thuật ghép lai.
- **`train.py`**: Script thực hiện huấn luyện mô hình với các tham số đã cấu hình.

### 3. Thành phần Ứng dụng Web
- **`app.py`**: File khởi chạy ứng dụng web (Main entry point).
- **`templates/`**: Chứa mã nguồn Frontend (Giao diện người dùng).
- **`static/`**: Thư mục lưu trữ tài nguyên tĩnh:
  - Ảnh người dùng tải lên để phân tích.
  - Ảnh kết quả sau khi mô hình xử lý xong.

### 4. Kiểm thử Local
- **`test_data/`**: Thư mục chứa dữ liệu hình ảnh để chạy thử nghiệm thủ công.
- **`test-result/`**: Thư mục chứa kết quả đầu ra khi chạy trực tiếp file `predict.py`.

## ⚙️ Cài đặt & Sử dụng

### 1\. Yêu cầu hệ thống

  * Python 3.8 trở lên.
  * Khuyên dùng môi trường ảo (Virtual Environment).

### 2\. Cài đặt thư viện

```bash
pip install ultralytics flask opencv-python onnx onnxruntime
```
### 3\. Đưa đường dẫn file best.pt (nằm trong dataset) của mô hình đã được train theo phương pháp bạn muốn vào trong file app.py.

### 4\. Chạy ứng dụng Web (Demo)

Kết nối Camera và chạy lệnh sau:

```bash
python app.py
```

Truy cập trình duyệt tại địa chỉ: `http://localhost:5000`

### 4\. Huấn luyện lại (Training)

Tôi đã train với GPU T4 trên Google Colab.
Nếu bạn muốn tự train lại mô hình:

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

  * *Giáo trình Xử lý ảnh (XLAS\_1.pdf, XLAS\_4.pdf, XLA6.pdf)* - Lý thuyết về Lọc không gian, Biến đổi độ xám và Phát hiện biên.
  * *Ultralytics YOLOv11 Docs*.
  * *Kaggle Helmet Detection Dataset*.

-----

**Thực hiện bởi:** [Nguyễn Thế Giáp/B22DCCN251]