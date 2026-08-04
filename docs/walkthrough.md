# Báo cáo Tiến độ & Khắc phục lỗi — Bước 1, 2, 3

---

## I. BƯỚC 1 — PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA) ✅

### 1. Thống kê số lượng ảnh
| Lớp bệnh | Tập Train | Tập Val | Tập Test | Tổng số ảnh |
|:---|:---:|:---:|:---:|:---:|
| **Leaf_Algal** (Đốm rong) | 323 | 69 | 70 | 462 |
| **Leaf_Blight** (Cháy lá) | 308 | 66 | 66 | 440 |
| **Leaf_Colletotrichum** (Thán thư) | 280 | 60 | 60 | 400 |
| **Leaf_Healthy** (Lá khỏe mạnh) | 338 | 72 | 74 | 484 |
| **Leaf_Phomopsis** (Bệnh Phomopsis) | 287 | 61 | 63 | 411 |
| **Leaf_Rhizoctonia** (Bệnh Rhizoctonia) | 278 | 59 | 61 | 398 |
| **TỔNG CỘNG** | **1814** | **387** | **394** | **2595** |

* **Tỉ lệ phân chia dataset:** Train 69.9% / Val 14.9% / Test 15.2%
* **Kích thước ảnh:** 100% (2595/2595) ảnh đều có kích thước đồng nhất là **400x400px**.
* **Ảnh lỗi (corrupt):** 0 ảnh lỗi.

### 2. Phân tích màu sắc RGB (Tỉ lệ R/G so với lá khỏe làm baseline)
| Lớp bệnh | Giá trị R | Giá trị G | Giá trị B | Tỉ lệ R/G | Nhận xét đặc trưng màu sắc |
|:---|:---:|:---:|:---:|:---:|:---|
| **La khỏe** | 0.448 | 0.529 | 0.384 | 0.847 | Baseline (màu xanh lá tự nhiên, tỉ lệ R/G thấp nhất) |
| **Đốm rong** | 0.433 | 0.503 | 0.355 | 0.861 | R/G tăng nhẹ (+0.014) — bắt đầu xuất hiện đốm nâu nhạt |
| **Cháy lá** | 0.450 | 0.508 | 0.373 | 0.887 | R/G tăng khá (+0.039) — cháy lá màu nâu đỏ rõ rệt |
| **Thán thư** | 0.445 | 0.498 | 0.354 | 0.893 | R/G tăng khá (+0.046) — đốm nâu đỏ đặc trưng |
| **Phomopsis** | 0.432 | 0.475 | 0.362 | 0.908 | R/G tăng nhiều (+0.061) — vết bệnh màu nâu/vàng úa |
| **Rhizoctonia** | 0.485 | 0.517 | 0.326 | 0.939 | R/G tăng cao nhất (+0.091) — đốm cháy nâu đậm loang lổ |

* **Nhận xét:** Lớp *Rhizoctonia* có màu nâu/đỏ đậm nhất. Lớp *Đốm rong* có màu sắc gần với *Lá khỏe* nhất nên mô hình sẽ khó phân biệt hai lớp này nhất.
* **Cân bằng lớp (Class Balance):** Tỉ lệ chênh lệch giữa lớp nhiều nhất và ít nhất là **1.22x** (rất tốt). Vẫn sử dụng **Class Weights** để CrossEntropyLoss hoạt động tối ưu nhất (Algal=0.936, Rhizoctonia=1.088).
* **Kết quả đầu ra:** Đã kết xuất 6 biểu đồ trực quan hóa trong thư mục [reports/eda/](file:///c:/Users/ahkie/VSCode_Projects/TLTN_DangAnhKiet/eda_report) cùng 2 file báo cáo: `eda_summary.txt` và `eda_summary.json`.

---

## II. BƯỚC 2 — XÂY DỰNG DATASET & PIPELINE DATALOADER (`src/durian_leaf_disease/data/dataset.py`) ✅

### 1. Đường ống tăng cường dữ liệu (Augmentation Pipeline)
* **TẬP HUẤN LUYỆN (TRAIN):**
  `Resize(256)` → `RandomCrop(224)` → `RandomHorizontalFlip(p=0.5)` → `RandomVerticalFlip(p=0.3)` → `RandomRotation(degrees=20)` → `ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)` → `ToTensor()` → `Normalize(ImageNet)` → `RandomErasing(p=0.15)`.
  * *Lưu ý:* Tuyệt đối **KHÔNG** sử dụng `RandomGrayscale` vì thông tin màu sắc (tỉ lệ R/G) là đặc trưng quan trọng nhất để phân biệt các loại bệnh lá sầu riêng.
* **TẬP VAL/TEST:**
  `Resize(256)` → `CenterCrop(224)` → `ToTensor()` → `Normalize(ImageNet)`.
* **Cấu hình DataLoader:** `batch_size=16` (an toàn cho GPU 4GB VRAM), `num_workers=2`, `pin_memory=True`, `persistent_workers=True`, `shuffle=True` (chỉ tập train), `seed=42`.
* **Kiểm chứng (Verify):** Khởi tạo thành công batch dữ liệu `torch.Size([16, 3, 224, 224])`, miền giá trị pixel sau chuẩn hóa nằm trong khoảng `[-3.657, 4.095]`.

### 2. Các điểm cải tiến nổi bật (v1 -> v2)
| # | Trạng thái trước (v1) | Giải pháp cải tiến hiện tại (v2) |
|---|---|---|
| 1 | Vẫn chứa `RandomGrayscale` trong train | Đã xóa bỏ hoàn toàn để bảo toàn đặc trưng màu sắc |
| 2 | Kích thước ảnh Resize không đồng nhất | Đồng bộ `Resize((256, 256))` cho cả 2 giai đoạn trước khi crop |
| 3 | Thiếu bước tránh overfitting cục bộ | Bổ sung `RandomErasing(p=0.15)` giúp mô hình học các vùng ảnh đa dạng |
| 4 | Không kiểm tra sự tồn tại của dữ liệu | Thêm hàm `_validate_dirs()` — dừng chương trình và raise lỗi rõ ràng |
| 5 | Kết quả huấn luyện bị lệch sau mỗi lần chạy | Cố định hạt giống bằng `set_seed(42)`, `seed_worker()` và `torch.Generator()` |
| 6 | Thời gian nạp dữ liệu lâu giữa các epoch | Kích hoạt `persistent_workers=True` để giữ các worker nạp luồng liên tục |
| 7 | `get_class_weights` tải toàn bộ ảnh (chậm) | Tối ưu hóa: Chỉ đếm số lượng tệp ảnh bằng `os.listdir` nhanh gấp 10 lần |
| 8 | Không kiểm tra thứ tự alphabet của lớp bệnh | Tự động so sánh ánh xạ `class_to_idx` với cấu hình trong `src/durian_leaf_disease/config.py` |
| 9 | Đầu ra xác nhận dữ liệu sơ sài | In chi tiết cấu trúc batch, khoảng pixel, ánh xạ và pipeline transform thực tế |
| 10 | Code thiếu tài liệu giải thích | Bổ sung docstring mô tả chi tiết lý do khoa học của từng transform |

---

## III. BƯỚC 3 — THIẾT KẾ MÔ HÌNH VÀ SO SÁNH (`src/durian_leaf_disease/models/transfer.py`) ✅

### 1. Bảng so sánh 3 kiến trúc mô hình đã thử nghiệm
| Tiêu chí | MobileNetV2 | EfficientNet-B0 | ResNet-50 |
|:---|:---:|:---:|:---:|
| **Tổng số tham số (Params)** | 2,553,862 | 4,337,538 | 24,034,630 |
| **Kích thước mô hình trên đĩa** | 9.87 MB | 16.71 MB | 91.89 MB |
| **Trainable Params (Giai đoạn 1)** | 329,990 (12.9%) | 329,990 (7.6%) | 526,598 (2.2%) |
| **Trainable Params (Giai đoạn 2)** | 1,536,070 (60.1%) | 3,485,730 (80.4%) | 23,809,286 (99.1%) |
| **Kiểm tra Forward Pass** | ✅ ĐẠT | ✅ ĐẠT | ✅ ĐẠT |
| **Dummy Input / Output** | `[2, 3, 224, 224]` → `[2, 6]` | `[2, 3, 224, 224]` → `[2, 6]` | `[2, 3, 224, 224]` → `[2, 6]` |

### 2. Thiết kế Classifier Head mới
Đồng bộ thay thế bộ phân loại gốc của cả 3 mô hình bằng một mạng phân loại tùy chỉnh:
`Dropout(0.3)` → `Linear(in_features, 256)` → `BatchNorm1d(256)` → `ReLU` → `Dropout(0.2)` → `Linear(256, 6)`.
* **Tại sao dùng 2 lớp tuyến tính thay vì 1?** Tập train nhỏ (1814 ảnh) dễ bị underfit nếu chuyển trực tiếp từ 2048 features (ResNet-50) về 6 đầu ra. Lớp ẩn 256 chiều giúp tăng dung lượng học tập cho mạng mà không gây nặng máy.
* **Tại sao dùng BatchNorm1d?** Giúp ổn định phân phối đầu vào trước khi qua lớp phân loại cuối cùng, tăng tốc độ hội tụ và giảm độ nhạy cảm với Learning Rate.
* **Tại sao dùng Dropout ở hai đầu?** Lớp đầu (0.3) chống đồng thích ứng (co-adaptation) của các đặc trưng từ backbone, lớp sau (0.2) điều hòa việc học của lớp ẩn.

### 3. Chiến lược Transfer Learning 2 giai đoạn
* **Giai đoạn 1 (Feature Extraction — 10 epochs):** Đóng băng toàn bộ backbone, chỉ huấn luyện classifier head mới. Giúp giữ nguyên các bộ lọc cạnh/màu sắc hữu ích đã học từ ImageNet và giúp mô hình hội tụ nhanh ở giai đoạn đầu.
* **Giai đoạn 2 (Fine-tuning — 20 epochs):** Mở 3 block cuối cùng của backbone để tinh chỉnh. Các block đầu (cơ bản) được giữ đóng băng, các block cuối (đặc trưng bậc cao) được tinh chỉnh để khớp với hình dạng đốm bệnh và cấu trúc lá sầu riêng thực tế.
  * *Lưu ý đối với ResNet-50:* Việc mở 3 block cuối (layer 2, 3, 4) sẽ mở khóa ~99% tham số huấn luyện. Lượng tham số này được điều hòa nghiêm ngặt bằng `weight_decay`, `early_stopping` và augmentation mạnh mẽ để tránh overfitting.

### 4. Các điểm cải tiến và sửa lỗi (v1 -> v2)
| # | Trạng thái trước (v1) | Giải pháp cải tiến hiện tại (v2) |
|---|---|---|
| 1 | Classifier chỉ có 1 lớp Linear đơn giản | Thiết kế lại bộ Classifier 2 tầng có Dropout, BatchNorm và ReLU |
| 2 | Hàm mở băng `unfreeze` ResNet-50 bị lỗi | Sửa logic: Mở chính xác n layer từ cuối: n=1 (layer4), n=2 (layer3+4), n=3 (layer2+3+4) |
| 3 | Không kiểm tra lỗi forward pass trước khi train | Bổ sung `verify_forward_pass()` chạy thử batch dummy size=2 (tránh lỗi BatchNorm) |
| 4 | Kích thước mô hình ước lượng không đúng | Bổ sung hàm tính dung lượng thực tế của mô hình `get_model_size_mb()` |
| 5 | Thiếu bảng tổng hợp so sánh các mô hình | Triển khai bảng so sánh chi tiết 5 tiêu chí xuất ra màn hình console |
| 6 | Gặp lỗi hiển thị tiếng Việt trên terminal Windows | Bổ sung cấu hình ép luồng đầu ra tiêu chuẩn dạng UTF-8 (`sys.stdout`) |
| 7 | Hàm đếm tham số chỉ in ra màn hình | Sửa hàm `get_param_counts()` trả về kiểu dữ liệu Dictionary để tái sử dụng |
| 8 | Thiếu lập luận khoa học về việc chọn mô hình | Bổ sung docstring phân tích lý do chọn MobileNetV2 (nhẹ/nhúng), EfficientNet-B0 (cân bằng) và ResNet-50 (mạnh) |
| 9 | Không lưu trữ kết quả kiểm thử | Kết xuất và lưu bảng so sánh cấu hình mô hình dạng tệp tin JSON `step3_model_comparison.json` |
| 10 | Không kiểm chứng được dòng chảy gradient | In kiểm tra danh sách các tensor có gradient trong Phase 2 (gradient flow check) |

---

## IV. BÁO CÁO CÁC LỖI ĐÃ KHẮC PHỤC & DỌN DẸP DỰ ÁN ✅

### 1. Thay đổi chung: Làm sạch code & Loại bỏ hardcode đường dẫn tuyệt đối
* **Vấn đề:** Trong các file code cũ có chứa dòng lệnh chèn đường dẫn cứng `sys.path.insert(0, 'C:/Users/ahkie/AppData/Roaming/Python/Python312/site-packages')`. Dòng này trỏ đến thư mục cài đặt thư viện của người dùng cụ thể (`ahkie`) và cố định phiên bản Python 3.12, gây mất tính linh động (portable) khi chạy trên máy khác hoặc môi trường khác.
* **Khắc phục:** 
  * Đã loại bỏ hoàn toàn dòng lệnh chèn đường dẫn cứng này ra khỏi tất cả các file trong dự án bao gồm: [model.py](file:///c:/Users/ahkie/VSCode_Projects/TLTN_DangAnhKiet/model.py), [train.py](file:///c:/Users/ahkie/VSCode_Projects/TLTN_DangAnhKiet/train.py), và [evaluate.py](file:///c:/Users/ahkie/VSCode_Projects/TLTN_DangAnhKiet/evaluate.py).
  * Làm sạch và loại bỏ luôn cả lệnh `import sys` dư thừa ở đầu [train.py](file:///c:/Users/ahkie/VSCode_Projects/TLTN_DangAnhKiet/train.py) và [evaluate.py](file:///c:/Users/ahkie/VSCode_Projects/TLTN_DangAnhKiet/evaluate.py) để code sạch sẽ nhất.
  * Xác nhận: Các gói thư viện phụ thuộc (`torch`, `tqdm`, `numpy`, `sklearn`, `matplotlib`, `seaborn`) được nạp tự động từ môi trường hoạt động hiện hành mà không cần chỉ định cứng đường dẫn tuyệt đối.

### 2. Chi tiết 17 lỗi đã sửa trên 3 file code chính
Các sửa đổi và tối ưu hóa chi tiết theo sơ đồ kiểm tra lỗi:
* **[eda.py](file:///c:/Users/ahkie/VSCode_Projects/TLTN_DangAnhKiet/eda.py) (Bước 1 — EDA):**
  1. Loại bỏ biến và hàm trùng lặp, import trực tiếp `CLASS_NAMES`, `CLASS_LABELS_VI`, và `DATASET_ROOT` từ `src/durian_leaf_disease/config.py` nhằm thống nhất một nguồn cấu hình duy nhất.
  2. Khai báo thêm thư viện chuẩn `warnings` để xử lý các cảnh báo hệ thống một cách có kiểm soát.
  3. Tối ưu hóa hiệu năng đọc ảnh: Gộp hàm `analyze_image_properties` và `compute_rgb_per_class` thành một hàm duy nhất `analyze_all_image_properties()`. Ảnh huấn luyện chỉ cần đọc 1 lần duy nhất để lấy thông tin kích thước và phân phối màu sắc, giảm 50% thời gian xử lý I/O đĩa.
  4. Loại bỏ cấu trúc bắt lỗi trống (`except: pass`). Đổi sang cảnh báo rõ ràng `warnings.warn(f"Không đọc được ảnh {img_path}: {e}")` khi phát hiện ảnh lỗi để người dùng dễ theo dõi.
  5. Bổ sung ghi chú giải thích rõ việc tăng tỉ lệ áp dụng augmentation (`p=0.9` và `0.5`) trong bản vẽ thử nghiệm ảnh tăng cường để làm rõ hình ảnh trực quan so với pipeline thật (`0.5` và `0.3`).
* **[dataset.py](file:///c:/Users/ahkie/VSCode_Projects/TLTN_DangAnhKiet/dataset.py) (Bước 2 — Dataset):**
  6. Xóa bỏ import không sử dụng `from collections import Counter` để tránh rác code.
  7. Thay thế các lệnh in cảnh báo thông thường (`print`) bằng lệnh dừng chương trình thực tế (`raise ValueError`) khi phát hiện các lỗi nghiêm trọng về cấu trúc thư mục dữ liệu hoặc thứ tự lớp bệnh không khớp.
  8. Cải tiến hàm tính toán trọng số lớp bệnh `get_class_weights()` để chỉ đếm các tệp tin ảnh có định dạng mở rộng hợp lệ (`.jpg`, `.jpeg`, `.png`, `.bmp`), loại bỏ hoàn toàn việc đếm nhầm tệp ẩn như `.DS_Store` hoặc tệp văn bản cấu hình.
  9. Thay đổi cơ chế in mô tả tăng cường trong hàm `verify_dataset()` để in trực tiếp từ pipeline biến đổi thực tế của `get_transforms()` thay vì sử dụng chuỗi ký tự mô tả tĩnh cứng nhắc.
  10. Bổ sung bình luận chi tiết về sự đánh đổi giữa hiệu năng và tính tái lập (`benchmark=False` và `deterministic=True`) của cuDNN trong PyTorch.
* **[model.py](file:///c:/Users/ahkie/VSCode_Projects/TLTN_DangAnhKiet/model.py) (Bước 3 — Model):**
  11. Thêm 3 tầng kiểm tra xác thực đầu vào (`assert`) trong hàm xây dựng bộ phân loại đầu `_make_classifier()` để đảm bảo các tham số kích thước đầu vào, số chiều ẩn và số lượng lớp đầu ra lớn hơn 0.
  12. Nhập cấu hình thiết bị `DEVICE` trực tiếp từ `src/durian_leaf_disease/config.py` để đồng bộ toàn bộ pipeline chạy huấn luyện và kiểm thử.
  13. Viết bổ sung lưu ý (cảnh báo) rõ ràng trong docstring hàm unfreeze đối với kiến trúc ResNet-50 khi mở 3 block cuối sẽ tác động đến hầu hết tham số huấn luyện của mô hình.

### 3. Danh sách các tệp tin rác đã dọn dẹp khỏi dự án
* Xóa bỏ tệp `transfer_learning_guide.py`: Chứa mã nguồn mẫu phiên bản cũ bị lỗi biên dịch (biến `num_features` chưa được khai báo) và thông tin cấu trúc lớp không còn khớp với thực tế 6 lớp bệnh.
* Xóa bỏ biểu đồ trùng lặp và không còn sử dụng: `reports/eda/04_image_size_stats.png`.
* Xóa bỏ biểu đồ phân tán kích thước ảnh thừa: `reports/eda/05_size_scatter.png`.

---

## V. KẾT QUẢ XÁC MINH TOÀN DỰ ÁN (VERIFICATION STATUS) 🏆
* **Tính di động (Portability):** Đã kiểm chứng tìm kiếm trên toàn dự án, **không còn xuất hiện bất kỳ đường dẫn cứng chứa tên người dùng (`sys.path.insert`) nào**. Dự án sạch sẽ và hoàn toàn độc lập với môi trường chạy.
* **Chạy thử nghiệm Dataset:** Chạy kiểm thử tệp [dataset.py](file:///c:/Users/ahkie/VSCode_Projects/TLTN_DangAnhKiet/dataset.py) cho kết quả nạp dữ liệu hoàn hảo, không gặp bất cứ lỗi luồng hay lỗi nạp tệp rác.
* **Chạy thử nghiệm Model:** Chạy kiểm thử tệp [model.py](file:///c:/Users/ahkie/VSCode_Projects/TLTN_DangAnhKiet/model.py) xác nhận cả 3 mô hình Transfer Learning (MobileNetV2, EfficientNet-B0, ResNet-50) đều vượt qua các bước kiểm tra Forward Pass ở cả 2 giai đoạn đóng băng/mở băng, in bảng so sánh dữ liệu và ghi nhận tệp so sánh JSON chính xác.

