import streamlit as st
import os
import subprocess
from PIL import Image
import tempfile
from downloader import ensure_realesrgan_exists

st.set_page_config(page_title="AI Upscaler (Real-ESRGAN)", layout="wide")

st.title("Phần Mềm Phóng To Ảnh Bằng AI (Real-ESRGAN 4x)")
st.write("Sử dụng mô hình RealESRGAN_x4plus để upscale ảnh sắc nét.")

# Kiểm tra & Tải Real-ESRGAN
ensure_realesrgan_exists()

# Chọn mô hình
model_name = st.selectbox(
    "Chọn mô hình AI:",
    ("realesrgan-x4plus", "realesrgan-x4plus-anime")
)

uploaded_file = st.file_uploader("Chọn một hình ảnh...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Mở ảnh gốc
    image = Image.open(uploaded_file)
    
    # Chia làm 2 cột
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ảnh Gốc")
        st.image(image, use_container_width=True)
    
    if st.button("Bắt đầu Upscale 4x", type="primary"):
        with st.spinner('Đang xử lý bằng AI... Quá trình này có thể mất vài giây đến vài phút tùy vào GPU của bạn.'):
            # Tạo thư mục tạm để lưu ảnh
            with tempfile.TemporaryDirectory() as tmpdirname:
                input_path = os.path.join(tmpdirname, "input.png")
                output_path = os.path.join(tmpdirname, "output.png")
                
                # Lưu ảnh người dùng upload
                image.save(input_path)
                
                # Đường dẫn exe (phải là đường dẫn tuyệt đối)
                exe_path = os.path.abspath(os.path.join("bin", "realesrgan-ncnn-vulkan.exe"))
                exe_dir = os.path.dirname(exe_path)
                
                # -i: input, -o: output, -n: model name, -t: tile size
                command = [
                    exe_path,
                    "-i", input_path,
                    "-o", output_path,
                    "-n", model_name,
                    "-t", "32", # Mức tile size thấp nhất
                    "-j", "1:1:1" # Ép chạy 1 luồng xử lý duy nhất để không nhân đôi bộ nhớ VRAM
                ]
                
                try:
                    # Truyền cwd = thư mục chứa exe để ncnn tải được file trọng số
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
                            
                            # Cung cấp nút tải về
                            with open(output_path, "rb") as file:
                                st.download_button(
                                    label="Tải Ảnh Kết Quả Về Máy",
                                    data=file,
                                    file_name=f"upscaled_{model_name}.png",
                                    mime="image/png",
                                    type="primary"
                                )
                        st.success("Upscale thành công!")
                    else:
                        st.error("Không tìm thấy file kết quả sau khi xử lý.")
                except subprocess.CalledProcessError as e:
                    st.error(f"Đã xảy ra lỗi khi chạy mô hình:\n{e.stderr.decode('utf-8', errors='ignore')}")
