import os
import urllib.request
import zipfile
import shutil
import streamlit as st

URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip"
BIN_DIR = "bin"
EXE_PATH = os.path.join(BIN_DIR, "realesrgan-ncnn-vulkan.exe")

def ensure_realesrgan_exists():
    if not os.path.exists(EXE_PATH):
        st.info("Đang tải realesrgan-ncnn-vulkan (chỉ tải ở lần đầu tiên)...")
        os.makedirs(BIN_DIR, exist_ok=True)
        zip_path = os.path.join(BIN_DIR, "realesrgan.zip")
        urllib.request.urlretrieve(URL, zip_path)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(BIN_DIR)
            
        os.remove(zip_path)
        
        # Xử lý lồng thư mục: nếu exe không nằm ngay trong bin mà nằm trong thư mục con
        if not os.path.exists(EXE_PATH):
            for root, dirs, files in os.walk(BIN_DIR):
                if "realesrgan-ncnn-vulkan.exe" in files:
                    # Tìm thấy exe ở thư mục con, di chuyển tất cả nội dung ra BIN_DIR
                    source_dir = root
                    for item in os.listdir(source_dir):
                        s = os.path.join(source_dir, item)
                        d = os.path.join(BIN_DIR, item)
                        if s != d:
                            shutil.move(s, d)
                    break
        
        if os.path.exists(EXE_PATH):
            st.success("Tải và cài đặt thành công!")
        else:
            st.error("Lỗi: Không tìm thấy file thực thi sau khi giải nén.")
