# RealESRGAN Integration

> Các kiến thức và troubleshooting khi tích hợp Real-ESRGAN (cả Vulkan, PyTorch và EDSR) cho ứng dụng AI Upscaler.

**Cập nhật lần cuối:** 2026-10-02

## Architecture

### Kiến trúc dự phòng khi GPU thiếu VRAM
Thay vì phụ thuộc hoàn toàn vào `realesrgan-ncnn-vulkan` (chỉ chạy GPU và dễ crash VRAM), kiến trúc ứng dụng Upscale cần fallback sang CPU. Tối ưu nhất là sử dụng `EDSR_x4.pb` qua module `cv2.dnn_superres` của OpenCV để chạy hoàn toàn trên RAM hệ thống, giữ được chất lượng cao cho ảnh thực tế mà không cần setup PyTorch.

## Bugs & Solutions

### vkAllocateMemory failed -2 trên iGPU
*   **Root Cause:** Các iGPU như AMD Radeon 780M bị giới hạn cứng về cấp phát bộ nhớ (allocation limit) bởi driver Vulkan, không thể nạp các model lớn như `realesrgan-x4plus` (33MB).
*   **Solution:** Giảm Tile Size xuống tối thiểu (`-t 32`) và ép chạy đơn luồng (`-j 1:1:1`). Tuy nhiên, nếu model vẫn quá lớn so với giới hạn buffer, giải pháp dứt điểm là fallback sang CPU model (ví dụ EDSR).

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
