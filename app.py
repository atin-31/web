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
        url = f'https://drive.google.com/uc?id={MODEL_ID}'
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
            return torch.zeros(ctx.shape, device=grad_output.device).scatter_(1, topk_indices, grad_output.gather(1, topk_indices)), None

    class ClinicalGigapixelMIL(nn.Module):
        def __init__(self, top_k=16):
            super().__init__()
            self.top_k = top_k
            self.instance_norm = nn.InstanceNorm2d(3, affine=True)
            resnet = resnet18(weights=None)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
            self.att_V, self.att_U, self.att_weights = nn.Linear(512, 128), nn.Linear(512, 128), nn.Linear(128, 1)
            self.classifier = nn.Linear(512, 1) 

        def forward(self, bag):
            bag = bag.squeeze(0)
            h = self.backbone(self.instance_norm(bag))
            raw_scores = self.att_weights(torch.tanh(self.att_V(h)) * torch.sigmoid(self.att_U(h))).T
            
            # Cơ chế Adaptive Top-K
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
# 2. GIAO DIỆN
# =====================================================================
st.set_page_config(page_title="ViST-Graph Clinical Dashboard", layout="wide")
st.title("🔬 ViST-Graph: Multi-Patch Diagnostic System")

with st.sidebar:
    st.header("⚙️ Control Panel")
    # Bật tính năng nạp nhiều mảnh cùng lúc
    uploaded_files = st.file_uploader("Tải lên danh sách các mảnh (Patches)...", 
                                      type=['png', 'jpg', 'jpeg'], 
                                      accept_multiple_files=True)

# ... (Giữ nguyên phần import và load_clinical_model) ...

if uploaded_files:
    model = load_clinical_model()
    
    # Tiền xử lý
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

    # BẮT ĐẦU HIỂN THỊ
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("📊 Diagnostic Result")
        st.metric("Xác suất Ác tính", f"{prob*100:.2f}%")
        if prob > 0.5:
            st.error("🚨 CẢNH BÁO: PHÁT HIỆN ÁC TÍNH")
        else:
            st.success("✅ AN TOÀN: CHƯA PHÁT HIỆN BẤT THƯỜNG")

    with col1:
        st.subheader("📍 Phân tích bằng chứng lâm sàng")
        
        # Sửa lỗi Shape tại đây: Flatten để thành mảng 1D
        raw_scores = attention.cpu().numpy().flatten() 
        norm_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-8)
        
        # Hiển thị Top 4 mảnh quan trọng
        n_top = min(4, len(uploaded_files))
        cols = st.columns(n_top)
        top_indices = np.argsort(raw_scores)[-n_top:][::-1]
        
        for i, idx in enumerate(top_indices):
            with cols[i]:
                patch_img = Image.open(uploaded_files[idx])
                score = raw_scores[idx]
                
                fig, ax = plt.subplots()
                ax.imshow(patch_img)
                # Overlay màu đỏ
                overlay = np.zeros((*np.array(patch_img.resize((128,128))).shape[:2], 3)) 
                overlay[:] = [1, 0, 0] 
                
                # Resize ảnh gốc để đồng nhất khi vẽ heatmap nếu cần
                ax.imshow(patch_img)
                ax.set_title(f"Top {i+1}\nScore: {score:.2f}", fontsize=8)
                ax.axis('off')
                st.pyplot(fig)
