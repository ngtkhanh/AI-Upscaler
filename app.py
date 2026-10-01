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
        "EDSR-x4 (Khuyên dùng - Ảnh thực tế, chống lỗi VRAM, chạy siêu chuẩn)",
        "realesrgan-x4plus-anime (Ảnh hoạt hình - Chạy GPU)"
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
                    
                    image.save(input_path)
                    
                    exe_path = os.path.abspath(os.path.join("bin", "realesrgan-ncnn-vulkan.exe"))
                    exe_dir = os.path.dirname(exe_path)
                    
                    command = [
                        exe_path,
                        "-i", input_path,
                        "-o", output_path,
                        "-n", "realesrgan-x4plus-anime"
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
