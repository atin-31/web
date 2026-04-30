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
from torchvision.models import resnet18

# =====================================================================
# 1. CẤU HÌNH & TẢI MODEL
# =====================================================================
MODEL_ID = '1J77dVnIjj_iVjDdWIpE3TLWmz063ae4s'
MODEL_PATH = 'clinical_robust_mil.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_clinical_model():
    if not os.path.exists(MODEL_PATH):
        url = f'https://google.com{MODEL_ID}'
        gdown.download(url, MODEL_PATH, quiet=False)
    
    class SparseRoutingTopK(torch.autograd.Function):
        @staticmethod
        def forward(ctx, attention_scores, k):
            topk_vals, topk_indices = torch.topk(attention_scores, k, dim=1)
            mask = torch.zeros_like(attention_scores).scatter_(1, topk_indices, 1.0)
            ctx.save_for_backward(topk_indices)
            ctx.shape = attention_scores.shape
            return attention_scores * mask
        
        @staticmethod
        def backward(ctx, grad_output):
            topk_indices, = ctx.saved_tensors
            grad_input = torch.zeros(ctx.shape, device=grad_output.device).scatter_(1, topk_indices, grad_output.gather(1, topk_indices))
            return grad_input, None

    class ClinicalGigapixelMIL(nn.Module):
        def __init__(self, top_k=16):
            super().__init__()
            self.top_k = top_k
            self.instance_norm = nn.InstanceNorm2d(3, affine=True)
            resnet = resnet18(weights=None)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
            self.att_V = nn.Linear(512, 128)
            self.att_U = nn.Linear(512, 128)
            self.att_weights = nn.Linear(128, 1)
            self.classifier = nn.Linear(512, 1) 

        def forward(self, bag):
            bag = bag.squeeze(0)
            h = self.backbone(self.instance_norm(bag))
            raw_scores = self.att_weights(torch.tanh(self.att_V(h)) * torch.sigmoid(self.att_U(h))).T
            
            num_patches = raw_scores.shape[1]
            adaptive_k = min(self.top_k, num_patches)
            
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
# 2. GIAO DIỆN NGƯỜI DÙNG
# =====================================================================
st.set_page_config(page_title="Hệ thống Chẩn đoán ViST-Graph", layout="wide")
st.title("🔬 ViST-Graph: Hệ thống chẩn đoán đa phân đoạn")

with st.sidebar:
    st.header("⚙️ Bảng điều khiển")
    uploaded_files = st.file_uploader("Tải lên các mảnh cắt (Patches)...", 
                                      type=['png', 'jpg', 'jpeg'], 
                                      accept_multiple_files=True)

if uploaded_files:
    model = load_clinical_model()
    
    # Tiền xử lý ảnh
    preprocess = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    patches_list = []
    for f in uploaded_files:
        img = Image.open(f).convert('RGB')
        patches_list.append(preprocess(img))
    
    input_bag = torch.stack(patches_list).unsqueeze(0).to(device)

    with st.spinner(f'Đang phân tích {len(uploaded_files)} mảnh cắt...'):
        with torch.no_grad():
            logits, attention = model(input_bag)
            prob = torch.sigmoid(logits).item()

    # --- PHẦN 1: HIỂN THỊ HÌNH ẢNH (Dàn hàng ngang để kích thước lớn hơn) ---
    st.divider()
    st.subheader("📍 Phân tích bằng chứng lâm sàng (Explainable AI)")
    
    # Xử lý trọng số Attention
    raw_scores = attention.cpu().numpy().flatten()
    # Chuẩn hóa để vẽ heatmap
    norm_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-8)
    
    # Hiển thị 4 mảnh quan trọng nhất
    n_display = min(4, len(uploaded_files))
    cols = st.columns(n_display)
    top_indices = np.argsort(raw_scores)[-n_display:][::-1]
    
    for i, idx in enumerate(top_indices):
        with cols[i]:
            patch_img = Image.open(uploaded_files[idx])
            score = raw_scores[idx]
            
            fig, ax = plt.subplots()
            ax.imshow(patch_img)
            
            # Tạo lớp phủ màu đỏ dựa trên mức độ quan trọng
            overlay = np.zeros((*np.array(patch_img).shape[:2], 3))
            overlay[:] = [1, 0, 0] # Màu đỏ
            ax.imshow(overlay, alpha=norm_scores[idx] * 0.4) 
            
            ax.set_title(f"Hạng {i+1}\nĐiểm: {score:.4f}", fontsize=10)
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig) # Tránh tốn bộ nhớ

    # --- PHẦF 2: KẾT QUẢ CHẨN ĐOÁN (Xuống dòng dưới hình ảnh) ---
    st.divider()
    st.subheader("📊 Kết quả chẩn đoán")
    
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.metric("Xác suất Ác tính", f"{prob*100:.2f}%")
        
    with res_col2:
        if prob > 0.5:
            st.error("🚨 CẢNH BÁO: PHÁT HIỆN DẤU HIỆU ÁC TÍNH")
            st.info("💡 Khuyến nghị: Cần thực hiện thêm xét nghiệm sinh thiết để xác nhận.")
        else:
            st.success("✅ AN TOÀN: CHƯA PHÁT HIỆN BẤT THƯỜNG")
            st.info("💡 Kết quả dựa trên các mảnh cắt đã cung cấp.")

else:
    st.info("👋 Vui lòng tải các file ảnh ở thanh bên trái để bắt đầu phân tích.")

# CSS tùy chỉnh để làm giao diện gọn gàng hơn
st.markdown("""
    <style>
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)
