import streamlit as st
import os
import subprocess
from PIL import Image
import cv2
import numpy as np
import tempfile
from downloader import ensure_realesrgan_exists
from download_edsr import ensure_edsr_exists

st.set_page_config(page_title="AI Upscaler (Chống Lỗi VRAM)", layout="wide")

st.title("Phần Mềm Phóng To Ảnh Bằng AI (Ổn định cao)")
st.write("Giải pháp Upscale 4x hỗ trợ tự động nhận diện và tương thích với mọi loại máy tính.")

# Đảm bảo cả hai mô hình đều có sẵn
ensure_realesrgan_exists()
ensure_edsr_exists()

# Chọn mô hình
model_name = st.selectbox(
    "Chọn mô hình AI:",
    (
        "EDSR-x4 (Cơ bản - Ảnh thực tế, chống lỗi VRAM)",
        "EDSR-x4-Sharp (Khuyên dùng - Nét vừa, giữ nguyên chữ)",
        "EDSR-x4-SuperSharp (Nét căng - Tăng cường sắc nét tối đa)",
        "realesrgan-x4plus-anime (Ảnh hoạt hình - Sắc nét nhưng vỡ chữ)",
        "realesr-animevideov3 (Anime Video - Nhanh, đỡ vỡ chữ hơn)",
        "realesr-animevideov3-Ultimate (Anime Tối Thượng - 4 Lớp Nâng Cấp Kép)"
    )
)

uploaded_file = st.file_uploader("Chọn một hình ảnh...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Mở ảnh gốc
    image = Image.open(uploaded_file).convert('RGB')
    
    # Chia làm 2 cột
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ảnh Gốc")
        st.image(image, use_container_width=True)
    
    if st.button("Bắt đầu Upscale 4x", type="primary"):
        with st.spinner('Đang xử lý bằng AI... Quá trình này có thể mất vài chục giây.'):
            
            # Xử lý bằng OpenCV EDSR
            if "EDSR-x4" in model_name:
                try:
                    # Chuyển PIL Image sang OpenCV format (BGR)
                    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    # Gọi OpenCV DNN Super Resolution
                    sr = cv2.dnn_superres.DnnSuperResImpl_create()
                    model_path = os.path.join("bin", "EDSR_x4.pb")
                    sr.readModel(model_path)
                    sr.setModel("edsr", 4) # Mô hình edsr scale 4x
                    
                    # Thực hiện Upscale trực tiếp bằng CPU (RAM) -> Chống tràn VRAM 100%
                    result_cv = sr.upsample(img_cv)
                    
                    # Áp dụng công nghệ Bù nét (Unsharp Mask) nếu người dùng chọn bản Sharp
                    if "SuperSharp" in model_name:
                        # Tăng cường nét tối đa
                        gaussian = cv2.GaussianBlur(result_cv, (0, 0), 3.0)
                        result_cv = cv2.addWeighted(result_cv, 2.0, gaussian, -1.0, 0)
                    elif "Sharp" in model_name:
                        # Làm mờ ảnh để tạo mặt nạ
                        gaussian = cv2.GaussianBlur(result_cv, (0, 0), 2.0)
                        # Cộng dồn chi tiết sắc nét vào ảnh gốc
                        result_cv = cv2.addWeighted(result_cv, 1.5, gaussian, -0.5, 0)
                    
                    # Chuyển lại sang PIL format
                    out_image = Image.fromarray(cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB))
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                        out_image.save(tmp.name)
                        output_path = tmp.name
                    
                    with col2:
                        st.subheader("Ảnh Sau Khi Upscale (4x)")
                        st.image(out_image, use_container_width=True)
                        
                        with open(output_path, "rb") as file:
                            st.download_button(
                                label="Tải Ảnh Kết Quả Về Máy",
                                data=file,
                                file_name="upscaled_edsr_x4.png",
                                mime="image/png",
                                type="primary"
                            )
                    st.success("Upscale thành công tuyệt đối! Bức ảnh của bạn đã được phóng to sắc nét.")
                    
                except Exception as e:
                    st.error(f"Đã xảy ra lỗi hệ thống: {str(e)}")
                    
            # Xử lý bằng realesrgan-x4plus-anime
            else:
                with tempfile.TemporaryDirectory() as tmpdirname:
                    input_path = os.path.join(tmpdirname, "input.png")
                    output_path = os.path.join(tmpdirname, "output.png")
                    
                    # Tiền xử lý ảnh gốc (Pre-sharpening) để AI nhận diện nét rõ hơn
                    image_to_process = image
                    if "Ultimate" in model_name:
                        img_cv_pre = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        gaussian_pre = cv2.GaussianBlur(img_cv_pre, (0, 0), 1.5)
                        img_cv_pre = cv2.addWeighted(img_cv_pre, 1.8, gaussian_pre, -0.8, 0)
                        image_to_process = Image.fromarray(cv2.cvtColor(img_cv_pre, cv2.COLOR_BGR2RGB))
                    
                    image_to_process.save(input_path)
                    
                    exe_path = os.path.abspath(os.path.join("bin", "realesrgan-ncnn-vulkan.exe"))
                    exe_dir = os.path.dirname(exe_path)
                    
                    if "animevideov3" in model_name:
                        real_model_name = "realesr-animevideov3"
                    else:
                        real_model_name = "realesrgan-x4plus-anime"

                    command = [
                        exe_path,
                        "-i", input_path,
                        "-o", output_path,
                        "-n", real_model_name,
                        "-t", "128" # Trả lại tiling 128 cho anime
                    ]
                    
                    try:
                        result = subprocess.run(
                            command, 
                            check=True, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE,
                            cwd=exe_dir
                        )
                        
                        if os.path.exists(output_path):
                            if "Ultimate" in model_name:
                                # 4 LỚP NÂNG CẤP KÉP
                                ai_img = cv2.imread(output_path)
                                orig_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                                h_ai, w_ai = ai_img.shape[:2]
                                
                                # 1. Smart Masking (Giữ nét chữ gốc, lấy nền AI)
                                orig_up = cv2.resize(orig_img, (w_ai, h_ai), interpolation=cv2.INTER_CUBIC)
                                orig_up_sharp = cv2.addWeighted(orig_up, 2.0, cv2.GaussianBlur(orig_up, (0,0), 3.0), -1.0, 0)
                                
                                gray_orig = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
                                edges = cv2.Canny(gray_orig, 100, 200)
                                mask = cv2.resize(edges, (w_ai, h_ai), interpolation=cv2.INTER_NEAREST)
                                mask = cv2.GaussianBlur(mask, (5, 5), 0)
                                mask_float = mask.astype(np.float32) / 255.0
                                mask_float = cv2.cvtColor(mask_float, cv2.COLOR_GRAY2BGR)
                                
                                blended = (ai_img * (1 - mask_float) + orig_up_sharp * mask_float).astype(np.uint8)
                                
                                # 2. Denoising (Làm mịn mảng màu mảng khối)
                                blended = cv2.bilateralFilter(blended, 5, 25, 25)
                                
                                # 3. Color Correction (Đẩy rực màu 15%)
                                hsv = cv2.cvtColor(blended, cv2.COLOR_BGR2HSV).astype(np.float32)
                                hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.15, 0, 255)
                                blended = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                                
                                # 4. Siêu nén điểm ảnh (Downscale mượt về x3)
                                new_w, new_h = int(w_ai * 0.75), int(h_ai * 0.75)
                                final_img = cv2.resize(blended, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                                
                                out_image = Image.fromarray(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
                            else:
                                out_image = Image.open(output_path)
                            
                            with col2:
                                st.subheader("Ảnh Sau Khi Upscale (4x)")
                                st.image(out_image, use_container_width=True)
                                
                                with open(output_path, "rb") as file:
                                    st.download_button(
                                        label="Tải Ảnh Kết Quả Về Máy",
                                        data=file,
                                        file_name="upscaled_anime.png",
                                        mime="image/png",
                                        type="primary"
                                    )
                            st.success("Upscale anime thành công!")
                        else:
                            st.error("Không tìm thấy file kết quả sau khi xử lý.")
                    except subprocess.CalledProcessError as e:
                        st.error(f"Đã xảy ra lỗi khi chạy mô hình:\n{e.stderr.decode('utf-8', errors='ignore')}")
