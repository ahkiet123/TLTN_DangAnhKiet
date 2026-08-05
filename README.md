# Phân loại bệnh trên lá sầu riêng

Dự án xây dựng mô hình nhận diện tình trạng lá sầu riêng từ ảnh. Mục tiêu là hỗ trợ phân loại nhanh lá khỏe và một số bệnh thường gặp, phục vụ cho việc tìm hiểu ứng dụng xử lý ảnh trong nông nghiệp.

## Chức năng chính

- Đọc và kiểm tra bộ dữ liệu ảnh theo các tập train, validation và test.
- Huấn luyện mô hình phân loại ảnh cho 6 nhóm: đốm rong, cháy lá, thán thư, lá khỏe, Phomopsis và Rhizoctonia.
- Đánh giá kết quả bằng Accuracy, F1-score và ma trận nhầm lẫn.
- Lưu mô hình và lịch sử huấn luyện để có thể xem lại kết quả.

## Kết quả

- Bộ dữ liệu gồm 2.595 ảnh lá sầu riêng, chia thành 6 nhóm.
- Kết quả thử nghiệm trên 394 ảnh test đạt **89,09% Accuracy** và **88,93% Weighted F1-score**.

## Công nghệ sử dụng

- Python
- PyTorch
- NumPy
- Matplotlib

## Cấu trúc thư mục

```text
data/       Dữ liệu ảnh
src/        Mã nguồn chính
scripts/    Các lệnh chạy dự án
outputs/    Mô hình và kết quả huấn luyện
reports/    Báo cáo, biểu đồ phân tích dữ liệu
docs/       Tài liệu dự án
```

## Cài đặt và chạy

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Đặt bộ dữ liệu tại `data/raw/Durian_Leaf_Diseases/` với các thư mục `train`, `val` và `test`. Nếu bộ dữ liệu ở vị trí khác, có thể đặt biến môi trường:

```powershell
$env:DURIAN_DATASET_ROOT = "D:\duong-dan\Durian_Leaf_Diseases"
```

Chạy kiểm tra dữ liệu, huấn luyện và đánh giá:

```powershell
python scripts/run_eda.py
python scripts/train.py --model mobilenet_v2
python scripts/evaluate.py --model mobilenet_v2
```

## Tác giả

Đặng Anh Kiệt
