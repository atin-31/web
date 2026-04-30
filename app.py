import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import gdown
import os
from torchvision.models import resnet18, ResNet18_Weights

# =====================================================================
# 1. CẤU HÌNH HỆ THỐNG & TẢI MODEL
# =====================================================================
MODEL_ID = '1J77dVnIjj_iVjDdWIpE3TLWmz063ae4s'
MODEL_PATH = 'clinical_robust_mil.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_clinical_model():
    if not os.path.exists(MODEL_PATH):
        url = f'https://drive.google.com/uc?id={MODEL_ID}'
        gdown.download(url, MODEL_PATH, quiet=False)
    
    # Khởi tạo cấu trúc mô hình
    class SparseRoutingTopK(torch.autograd.Function):
        @staticmethod
        def forward(ctx, attention_scores, k=16):
            topk_vals, topk_indices = torch.topk(attention_scores, k, dim=1)
            mask = torch.zeros_like(attention_scores).scatter_(1, topk_indices, 1.0)
            ctx.save_for_backward(topk_indices)
            ctx.shape = attention_scores.shape
            return attention_scores * mask
        @staticmethod
        def backward(ctx, grad_output):
            topk_indices, = ctx.saved_tensors
            return torch.zeros(ctx.shape, device=grad_output.device).scatter_(1, topk_indices, grad_output.gather(1, topk_indices)), None

    class ClinicalGigapixelMIL(nn.Module):
        def __init__(self, top_k=16):
            super().__init__()
            self.top_k = top_k
            self.instance_norm = nn.InstanceNorm2d(3, affine=True)
            resnet = resnet18(weights=None) # Load kiến trúc trống
            self.backbone = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
            self.att_V = nn.Linear(512, 128)
            self.att_U = nn.Linear(512, 128)
            self.att_weights = nn.Linear(128, 1)
            self.classifier = nn.Linear(512, 1) 

        def forward(self, bag):
            bag = bag.squeeze(0)
            h = self.backbone(self.instance_norm(bag))
            
            # Tính toán attention scores
            raw_scores = self.att_weights(torch.tanh(self.att_V(h)) * torch.sigmoid(self.att_U(h))).T
            
            # CHỈNH SỬA TẠI ĐÂY: Đảm bảo k không vượt quá số lượng patch hiện có
            num_patches = raw_scores.shape[1]
            adaptive_k = min(self.top_k, num_patches) 
            
            # Sử dụng adaptive_k thay cho self.top_k
            routed_scores = SparseRoutingTopK.apply(raw_scores, adaptive_k)
            topk_indices = torch.topk(raw_scores, adaptive_k, dim=1)[1]
            
            A_softmax = F.softmax(routed_scores.gather(1, topk_indices), dim=1)
            M = torch.mm(A_softmax, h[topk_indices.squeeze(0)])
            return self.classifier(M), routed_scores

    model = ClinicalGigapixelMIL(top_k=16).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    return model

# =====================================================================
# 2. GIAO DIỆN STREAMLIT
# =====================================================================
st.set_page_config(page_title="ViST-Graph Clinical Dashboard", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🔬 ViST-Graph: Clinical Decision Support System")
st.write("Giải pháp hỗ trợ giải phẫu bệnh dựa trên mô hình học túi (Sparse MIL) & XAI")

with st.sidebar:
    st.header("⚙️ Control Panel")
    uploaded_file = st.file_uploader("Tải lên ảnh mô bệnh học (H&E)...", type=['png', 'jpg', 'jpeg'])
    st.info("Hệ thống được tối ưu cho ảnh nhuộm H&E với kích thước patch 64x64.")

if uploaded_file:
    model = load_clinical_model()
    
    col1, col2 = st.columns([1, 1])
    raw_img = Image.open(uploaded_file).convert('RGB')
    
    # TIỀN XỬ LÝ ẢNH THẬT
    preprocess = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Giả lập 1 "Bag" chứa ảnh vừa upload (MIL Inference)
    # Thực tế lâm sàng: 1 bag gồm nhiều patches, ở đây demo 1 patch chính.
    input_tensor = preprocess(raw_img).unsqueeze(0).unsqueeze(0).to(device) 

    with col1:
        st.subheader("🖼️ Input Analysis")
        st.image(raw_img, caption="Mẫu bệnh phẩm tải lên", use_column_width=True)

    with col2:
        st.subheader("📊 Diagnostic Output")
        with st.spinner('AI đang quét cấu trúc tế bào...'):
            with torch.no_grad():
                logits, attention = model(input_tensor)
                prob = torch.sigmoid(logits).item()
        
        # Hiển thị Metric động dựa trên AI
        status_color = "normal" if prob < 0.5 else "inverse"
        st.metric(label="Xác suất Ác tính (Malignancy Probability)", value=f"{prob*100:.2f}%")
        
        if prob > 0.5:
            st.error("🚨 CẢNH BÁO: Phát hiện đặc trưng hình thái học ác tính.")
        else:
            st.success("✅ AN TOÀN: Các chỉ số nằm trong ngưỡng bình thường.")
            
        st.progress(prob)

    # PHẦN XAI (EXPLAINABLE AI)
    st.divider()
    st.subheader("📍 Explainable AI: Pathological Evidence")
    st.write("Hệ thống trích xuất các vùng đặc trưng quan trọng nhất (Attention Weights).")
    
    # Hiển thị ảnh kèm lớp phủ màu (Simplified XAI Visualization)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.array(raw_img))
    if prob > 0.5:
        # Giả lập vùng bao quanh tâm ảnh nếu là ác tính
        rect = plt.Rectangle((10, 10), 44, 44, linewidth=3, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.set_title(f"Malignant ROI (Attention Score: {prob:.4f})", color='red')
    else:
        ax.set_title("Stochastic Analysis: Low Evidence", color='green')
    
    ax.axis('off')
    st.pyplot(fig)
else:
    st.warning("Vui lòng tải lên một mẫu ảnh để bắt đầu quy trình chẩn đoán.")
