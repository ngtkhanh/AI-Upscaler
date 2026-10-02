# RealESRGAN Integration

> Các kiến thức và troubleshooting khi tích hợp Real-ESRGAN (cả Vulkan, PyTorch và EDSR) cho ứng dụng AI Upscaler.

**Cập nhật lần cuối:** 2026-10-02

## Architecture

### Kiến trúc dự phòng khi GPU thiếu VRAM
Thay vì phụ thuộc hoàn toàn vào `realesrgan-ncnn-vulkan` (chỉ chạy GPU và dễ crash VRAM), kiến trúc ứng dụng Upscale cần fallback sang CPU. Tối ưu nhất là sử dụng `EDSR_x4.pb` qua module `cv2.dnn_superres` của OpenCV để chạy hoàn toàn trên RAM hệ thống, giữ được chất lượng cao cho ảnh thực tế mà không cần setup PyTorch.

## Bugs & Solutions

### vkAllocateMemory failed -2 trên iGPU
*   **Root Cause:** Các iGPU như AMD Radeon 780M bị giới hạn cứng về cấp phát bộ nhớ (allocation limit) bởi driver Vulkan, không thể nạp các model lớn như `realesrgan-x4plus` (33MB) ngay cả khi cắt nhỏ ảnh.
*   **Solution:** Giảm Tile Size (`-t 64`) và ép chạy đơn luồng (`-j 1:1:1`). Tuy nhiên, đối với `realesrgan-x4plus` trên card tích hợp, thao tác này vẫn thất bại. Giải pháp dứt điểm là loại bỏ model này khỏi UI hoặc fallback sang CPU model (ví dụ EDSR).

### Lỗi biến dạng chữ (Text Distortion) khi dùng model Anime
*   **Root Cause:** Model `realesrgan-x4plus-anime` cố gắng làm phẳng và mượt các đường nét. Khi gặp chữ hoặc chi tiết nhỏ của ảnh thực tế, nó tự động nội suy sai và bóp méo chữ thành các hình thù lạ.
*   **Solution:** Dùng EDSR thay thế để bảo toàn hình dáng gốc của chữ, sau đó áp dụng **Unsharp Mask (bù nét)** ở bước hậu xử lý (dùng `cv2.addWeighted`) để tăng cường độ sắc nét tương đương RealESRGAN.

### Lỗi invalid gpu device khi ép dùng CPU
*   **Root Cause:** Binary `realesrgan-ncnn-vulkan` được compile cứng cho Vulkan. Tham số `-g -1` không hợp lệ.
*   **Solution:** Không dùng tham số `-g -1` cho `realesrgan-ncnn-vulkan`. Nếu muốn chạy CPU, dùng bản PyTorch hoặc OpenCV.

### Lỗi KeyError: '__version__' khi setup.py trên Google Colab
*   **Root Cause:** Colab dùng Python 3.10+ khiến hàm `exec()` không cập nhật biến `locals()` trong context function của `setup.py` (mã nguồn cũ của Real-ESRGAN).
*   **Solution:** Replace dòng `return locals()['__version__']` trong `setup.py` thành version cứng (vd `return '0.2.5.0'`) bằng regex/sed/python script trước khi chạy `python setup.py develop`.

### Lỗi cài đặt basicsr (Getting requirements to build wheel failed)
*   **Root Cause:** Python 3.13 đã loại bỏ `distutils`, khiến setuptools bản mới (>=70) đánh lỗi khi build các project cũ như `basicsr`.
*   **Solution:** Ép cài setuptools bản cũ: `pip install "setuptools<70"` trước khi cài đặt `basicsr`.

## How-To

### Cách setup OpenCV EDSR Upscale 4x
1. Cài đặt `opencv-contrib-python`.
2. Tải model `EDSR_x4.pb` (~38MB).
3. Sử dụng code: 
```python
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel("EDSR_x4.pb")
sr.setModel("edsr", 4)
result_img = sr.upsample(input_cv_img)
```

## Patterns

### Model Selection Pattern cho UI
Cung cấp lựa chọn RÕ RÀNG cho user trong UI: một tùy chọn an toàn/ổn định chạy bằng RAM/CPU (như EDSR cho ảnh thực tế) và một tùy chọn GPU nhẹ (Anime). Tránh cung cấp tuỳ chọn dễ gây crash hệ thống nếu phần mềm được chạy trên các hardware phân mảnh mạnh (như Windows iGPU).

### Hybrid Upscaling Pattern (EDSR + Unsharp Mask)
Thay vì cố gắng nạp các GAN model nặng nề dễ văng lỗi để có được độ sắc nét, có thể dùng các mô hình an toàn (EDSR chạy bằng CPU) kết hợp với thuật toán bù nét cổ điển (Unsharp Mask qua `cv2`). Điều này cân bằng hoàn hảo giữa: độ tin cậy phần cứng tuyệt đối (không crash VRAM), tính nguyên bản của ảnh (không bị méo chữ) và cảm quan độ nét (sharpness) tốt.
