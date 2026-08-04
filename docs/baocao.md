# BÁO CÁO TIẾN ĐỘ TIỂU LUẬN TỐT NGHIỆP — BƯỚC 1, 2, 3

---

## 📋 TỔNG QUAN ĐỀ TÀI
* **Tên đề tài:** Phân loại bệnh hại lá sầu riêng tại Việt Nam dựa trên các kiến trúc mạng nơ-ron học sâu (CNN).
* **Đối tượng so sánh:** MobileNetV2, EfficientNet-B0, ResNet-50.
* **Môi trường thực nghiệm:** PyTorch, chạy trên GPU NVIDIA RTX 3050 Ti Laptop (4GB VRAM).
* **Mục tiêu chính:** Đạt F1-Score ≥ 85% trên tập kiểm thử và xác định mô hình tối ưu cho triển khai thực tế.

---

## 📂 BƯỚC 1: THU THẬP VÀ TIỀN XỬ LÝ DỮ LIỆU
Nội dung này tương đương với quy trình khảo sát dữ liệu khám phá (EDA) và thiết kế pipeline xử lý dữ liệu (`src/durian_leaf_disease/data/dataset.py`).

### 1. Thu thập, làm sạch và phân chia dữ liệu
* **Tích hợp dataset:** Sử dụng bộ dữ liệu hình ảnh lá sầu riêng Việt Nam thực tế, đã qua kiểm tra định dạng và loại bỏ các tệp ảnh bị lỗi (0 ảnh lỗi).
* **Đồng nhất kích thước:** Toàn bộ 2,595 ảnh trong bộ dữ liệu được xác nhận có kích thước đồng nhất là **400x400px** trước khi đi vào đường ống xử lý.
* **Phân chia dữ liệu:** Tỷ lệ phân chia Train / Validation / Test xấp xỉ **70% / 15% / 15%** cụ thể như sau:
  | Lớp bệnh | Tập Train | Tập Val | Tập Test | Tổng số ảnh |
  |:---|:---:|:---:|:---:|:---:|
  | **Leaf_Algal** (Bệnh đốm rong) | 323 | 69 | 70 | 462 |
  | **Leaf_Blight** (Bệnh cháy lá) | 308 | 66 | 66 | 440 |
  | **Leaf_Colletotrichum** (Bệnh thán thư) | 280 | 60 | 60 | 400 |
  | **Leaf_Healthy** (Lá khỏe mạnh) | 338 | 72 | 74 | 484 |
  | **Leaf_Phomopsis** (Bệnh Phomopsis) | 287 | 61 | 63 | 411 |
  | **Leaf_Rhizoctonia** (Bệnh Rhizoctonia) | 278 | 59 | 61 | 398 |
  | **TỔNG CỘNG** | **1814** | **387** | **394** | **2595** |

### 2. Phân tích khám phá đặc trưng màu sắc RGB
* Thống kê trung bình giá trị pixel trên các kênh R, G, B cho từng lớp bệnh nhằm tìm ra đặc trưng định lượng:
  | Lớp bệnh | Giá trị R | Giá trị G | Giá trị B | Tỉ lệ R/G | Đặc trưng trực quan nhận diện |
  |:---|:---:|:---:|:---:|:---:|:---|
  | **Lá khỏe** | 0.448 | 0.529 | 0.384 | **0.847** | Baseline (Xanh lá tự nhiên, tỉ lệ R/G thấp nhất) |
  | **Đốm rong** | 0.433 | 0.503 | 0.355 | **0.861** | R/G tăng nhẹ — xuất hiện các đốm nâu nhạt |
  | **Cháy lá** | 0.450 | 0.508 | 0.373 | **0.887** | R/G tăng khá — các mảng cháy lá khô màu nâu đỏ |
  | **Thán thư** | 0.445 | 0.498 | 0.354 | **0.893** | R/G tăng khá — đốm tròn nâu đỏ đặc trưng |
  | **Phomopsis** | 0.432 | 0.475 | 0.362 | **0.908** | R/G tăng nhiều — vết bệnh màu nâu/vàng úa loang lổ |
  | **Rhizoctonia** | 0.485 | 0.517 | 0.326 | **0.939** | R/G cao nhất — đốm cháy lớn màu nâu sẫm hoại tử |
* **Kết luận rút ra:** Lớp *Rhizoctonia* có màu nâu đậm và đỏ nhất (tỉ lệ R/G cao nhất). Lớp *Đốm rong* có màu sắc gần với *Lá khỏe* nhất (tỉ lệ R/G chênh lệch rất ít), đây sẽ là cặp lớp gây khó khăn lớn nhất cho mô hình phân loại.

### 3. Xử lý mất cân bằng lớp (Class Imbalance)
* Tỉ lệ chênh lệch số lượng ảnh lớn nhất và nhỏ nhất giữa các lớp là **1.22x** (trong ngưỡng an toàn).
* Nhằm tối ưu hóa hàm lỗi `CrossEntropyLoss`, dự án áp dụng kỹ thuật **Class Weights** tính toán theo nghịch đảo tần suất lớp:
  $$\text{weight}_c = \frac{N}{C \times n_c}$$
  Với kết quả cụ thể: *Algal*=0.936, *Blight*=0.983, *Colletotrichum*=1.081, *Healthy*=0.894, *Phomopsis*=1.052, *Rhizoctonia*=1.088.

### 4. Áp dụng kỹ thuật tăng cường dữ liệu (Data Augmentation) & DataLoader
* **Đường ống tăng cường tập huấn luyện (Train Pipeline):**
  `Resize(256)` → `RandomCrop(224)` → `RandomHorizontalFlip(p=0.5)` → `RandomVerticalFlip(p=0.3)` → `RandomRotation(degrees=20)` → `ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)` → `ToTensor()` → `Normalize(ImageNet)` → `RandomErasing(p=0.15)`.
  * *Lập luận khoa học:* **Không sử dụng** phép biến đổi `RandomGrayscale` hoặc `ColorJitter` với độ lệch hue quá lớn vì tỉ lệ màu sắc (R/G) là đặc trưng quan trọng nhất để nhận diện các loại bệnh hại.
* **Đường ống tập Validation/Test:**
  `Resize(256)` → `CenterCrop(224)` → `ToTensor()` → `Normalize(ImageNet)`.
* **Cấu hình DataLoader:** Kích thước `batch_size = 16`, `num_workers = 2`, `pin_memory = True` nhằm tăng tốc độ truyền tải dữ liệu lên GPU, và kích hoạt `persistent_workers = True` để giữ các worker luồng nạp liên tục giữa các epoch. Cố định hạt giống bằng `set_seed(42)` để đảm bảo tính tái lập kết quả.

---

## 🛠️ BƯỚC 2: XÂY DỰNG VÀ HUẤN LUYỆN MÔ HÌNH
Nội dung này tương đương với thiết kế kiến trúc mạng (`src/durian_leaf_disease/models/transfer.py`) và viết pipeline huấn luyện (`src/durian_leaf_disease/training/trainer.py`).

### 1. Kiến trúc 3 mô hình huấn luyện
* **MobileNetV2:** Kiến trúc gọn nhẹ sử dụng tích chập phân tách chiều sâu (Depthwise Separable Convolution) và khối Bottleneck nghịch đảo.
* **EfficientNet-B0:** Tối ưu hóa đồng thời độ sâu, độ rộng và độ phân giải của mạng (Compound Scaling) giúp đạt hiệu năng cao với lượng tham số nhỏ.
* **ResNet-50:** Baseline phân loại mạnh mẽ tận dụng kết nối tắt (Skip Connections) chống tiêu biến gradient trên các tầng mạng sâu.

### 2. Thiết kế Classifier Head tùy chỉnh đồng bộ
Cả 3 mô hình pretrained được thay thế lớp phân loại gốc bằng một Classifier Head đồng bộ:
$$\text{Feature Map} \rightarrow \text{Dropout}(0.3) \rightarrow \text{Linear}(\text{in\_features}, 256) \rightarrow \text{BatchNorm1d}(256) \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.2) \rightarrow \text{Linear}(256, 6)$$
* **Tầng ẩn 256 chiều:** Giảm bớt số lượng chiều đặc trưng đột ngột (từ 2048 chiều của ResNet-50 xuống 6 lớp) để tránh làm mất mát thông tin hữu ích của tập dữ liệu nhỏ.
* **BatchNorm1d:** Ổn định phân phối dữ liệu đầu vào trước lớp tuyến tính cuối cùng, giúp hội tụ nhanh hơn.
* **Hai tầng Dropout (0.3 và 0.2):** Ngăn chặn hiện tượng đồng thích ứng (co-adaptation) của các đặc trưng học được từ backbone và chống overfitting cục bộ trên tập classifier mới.

### 3. Chiến lược Transfer Learning và Fine-tuning 2 giai đoạn
* **Giai đoạn 1 (Feature Extraction — 10 Epochs):** Đóng băng (freeze) toàn bộ trọng số của backbone mạng đã được pretrained trên ImageNet. Chỉ huấn luyện bộ phân loại Classifier Head mới với learning rate lớn (`lr = 1e-3`).
* **Giai đoạn 2 (Fine-tuning — 20 Epochs):** Khôi phục trọng số tốt nhất từ Giai đoạn 1. Mở băng (unfreeze) **3 block cuối cùng** của backbone để tinh chỉnh sâu hơn các đặc trưng cấp cao phù hợp với cấu trúc vết bệnh lá sầu riêng. Sử dụng kỹ thuật học thích ứng đa tốc độ:
  * Trọng số backbone cập nhật với learning rate rất nhỏ (`lr = 1e-5`) để tránh làm hỏng các bộ lọc cạnh/màu sắc cơ bản.
  * Trọng số classifier head cập nhật với learning rate vừa phải (`lr = 1e-4`).
  * *Lưu ý đặc biệt với ResNet-50:* Khi mở băng 3 block cuối (layer 2, 3, 4) sẽ mở khóa ~99% tham số. Chúng được kiểm soát chặt chẽ bằng Weight Decay mạnh và Early Stopping.

### 4. Cấu hình các siêu tham số huấn luyện (Hyperparameters)
* **Optimizer:** Sử dụng thuật toán tối ưu hóa `AdamW` tích hợp phân rã trọng số để chống overfitting với hệ số `weight_decay = 1e-4`.
* **LR Scheduler:** Áp dụng `CosineAnnealingLR` giảm dần tốc độ học theo hình Cosine ở cả hai giai đoạn giúp mô hình ổn định quanh điểm cực tiểu toàn cục.
* **Dừng sớm (Early Stopping):** Theo dõi `val_loss`, tự động dừng huấn luyện nếu không có cải tiến trong `patience = 7` epoch liên tiếp.
* **Độ chính xác hỗn hợp (Mixed Precision):** Sử dụng `torch.amp` (FP16 tự động) giúp giảm tải đáng kể dung lượng bộ nhớ VRAM của GPU và tăng tốc huấn luyện.

---

## 📊 BƯỚC 3: ĐÁNH GIÁ VÀ SO SÁNH THỰC NGHIỆM
Nội dung này mô tả các thông số vật lý ban đầu của 3 mô hình sau khi build thử nghiệm thành công và kế hoạch thực nghiệm chi tiết.

### 1. So sánh thông số vật lý ban đầu của 3 kiến trúc mô hình (Đã Verify)
* Chạy thử nghiệm thành công forward pass trên cả 3 mô hình với batch dummy dạng tensor `[2, 3, 224, 224]`, cho đầu ra định dạng `[2, 6]` tương ứng với 6 lớp bệnh.
* Thông số vật lý thực tế thu thập từ mã nguồn:
  | Chỉ số so sánh | MobileNetV2 | EfficientNet-B0 | ResNet-50 |
  |:---|:---:|:---:|:---:|
  | **Tổng số tham số (Params)** | 2,553,862 | 4,337,538 | 24,034,630 |
  | **Kích thước file trên đĩa** | 9.87 MB | 16.71 MB | 91.89 MB |
  | **Params huấn luyện GĐ 1** | 329,990 (12.9%) | 329,990 (7.6%) | 526,598 (2.2%) |
  | **Params huấn luyện GĐ 2** | 1,536,070 (60.1%) | 3,485,730 (80.4%) | 23,809,286 (99.1%) |
  | **Độ khớp kiểm tra Forward Pass** | ✅ ĐẠT | ✅ ĐẠT | ✅ ĐẠT |
  | **Số tensor có gradient (Phase 2)** | 27 tensors | 74 tensors | 132 tensors |

### 2. Kế hoạch đánh giá chất lượng phân loại trên tập Test
Sau khi quá trình huấn luyện hoàn tất, hiệu năng phân loại của 3 mô hình sẽ được đánh giá chi tiết thông qua các chỉ số:
* **Accuracy, Precision, Recall, F1-Score:** Tính toán trên từng lớp và trung bình (macro/weighted average) để đảm bảo chất lượng tổng quát hóa (đặc biệt mục tiêu F1-Score đạt trên 85%).
* **Confusion Matrix (Ma trận nhầm lẫn):** Trực quan hóa chi tiết các lớp bệnh dễ bị nhận diện nhầm lẫn (đặc biệt là cặp Đốm rong và Lá khỏe).
* **Đường cong ROC và chỉ số AUC:** Đánh giá độ phân biệt của mô hình trên từng lớp cụ thể.

### 3. Kế hoạch trực quan hóa Grad-CAM (Độ giải thích được)
* Dự án thiết lập tích hợp phương pháp **Grad-CAM (Gradient-weighted Class Activation Mapping)** ở bước đánh giá.
* **Cách thức:** Trích xuất bản đồ kích hoạt (activation map) từ lớp tích chập cuối cùng của backbone trước khi qua lớp Global Average Pooling (ví dụ: lớp `features` cuối ở MobileNetV2/EfficientNet-B0, hoặc lớp `layer4` ở ResNet-50).
* **Ý nghĩa:** Trực quan hóa bằng bản đồ nhiệt (heatmap) đè lên ảnh gốc để hiển thị rõ vùng đặc trưng trên lá sầu riêng mà mô hình tập trung vào khi phân loại (ví dụ: vết cháy lá khô, đốm bệnh thán thư tròn hay đốm rong loang lổ). Điều này tăng tính minh bạch và độ tin cậy của mô hình học sâu đối với nông nghiệp.

