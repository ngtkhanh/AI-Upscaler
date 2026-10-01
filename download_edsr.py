import os
import urllib.request
import streamlit as st

URL = "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb"
MODEL_PATH = os.path.join("bin", "EDSR_x4.pb")

def ensure_edsr_exists():
    if not os.path.exists(MODEL_PATH):
        st.info("Đang tải mô hình AI ảnh thực (EDSR) tương thích với CPU của bạn (~38MB)...")
        os.makedirs("bin", exist_ok=True)
        urllib.request.urlretrieve(URL, MODEL_PATH)
        st.success("Tải mô hình thành công!")
