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
# 1. CẤU HÌNH & TẢI MODEL (Giữ nguyên logic của bạn)
# =====================================================================
MODEL_ID = '1J77dVnIjj_iVjDdWIpE3TLWmz063ae4s'
MODEL_PATH = 'clinical_robust_mil.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_clinical_model():
    if not os.path.exists(MODEL_PATH):
        # Lưu ý: Sửa lại URL drive chuẩn nếu cái cũ lỗi
        url = f'https://google.com{MODEL_ID}'
        gdown.download(url, MODEL_PATH, quiet=False)
    
    # [Giữ nguyên class SparseRoutingTopK và ClinicalGigapixelMIL như code cũ của bạn]
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
            self.att_V, self.att_U, self.att_weights = nn.Linear(512, 128), nn.Linear(512, 128), nn.Linear(128, 1)
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
# 2. GIAO DIỆN NÂNG CAO (UI/UX)
# =====================================================================
st.set_page_config(page_title="PathoInsight AI | Clinical Support", layout="wide")

# CSS để tối ưu giao diện y tế
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border-left: 5px solid #007bff; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .report-card { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }
    h1, h2, h3 { color: #1e293b; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# Tiêu đề mới chuyên nghiệp hơn
st.write("## 🧬 PathoInsight AI: Trợ lý Phân tích Giải phẫu bệnh")
st.caption("Hệ thống hỗ trợ quyết định lâm sàng dựa trên công nghệ học sâu đa phân đoạn.")

with st.sidebar:
    st.image("https://flaticon.com", width=80) # Icon y tế
    st.header("📋 Nhập dữ liệu")
    uploaded_files = st.file_uploader("Tải các mảnh cắt (Patches) .jpg, .png", 
                                      type=['png', 'jpg', 'jpeg'], 
                                      accept_multiple_files=True)
    st.info("Hệ thống chấp nhận tải lên hàng loạt ảnh từ cùng một mẫu bệnh phẩm.")

if uploaded_files:
    model = load_clinical_model()
    
    # Tiền xử lý
    preprocess = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    patches_list = [preprocess(Image.open(f).convert('RGB')) for f in uploaded_files]
    input_bag = torch.stack(patches_list).unsqueeze(0).to(device)

    with st.spinner('Đang tiến hành phân tích mô học...'):
        with torch.no_grad():
            logits, attention = model(input_bag)
            prob = torch.sigmoid(logits).item()

    # --- PHẦN KẾT QUẢ TỔNG QUAN (Đưa lên đầu cho bác sĩ xem trước) ---
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.subheader("📊 Kết quả phân tích")
        # Thay đổi màu sắc dựa trên xác suất
        color = "red" if prob > 0.5 else "green"
        st.markdown(f"<h1 style='color:{color}; text-align:center;'>{prob*100:.1f}%</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center;'>Xác suất bệnh lý (Malignancy)</p>", unsafe_allow_html=True)

    with c2:
        st.subheader("💡 Kết luận lâm sàng")
        if prob > 0.5:
            st.error("**NGUY CƠ CAO:** Hình ảnh ghi nhận các đặc điểm tương đồng với bệnh lý ác tính.")
            st.warning("Đề nghị hội chẩn thêm với hội đồng chuyên môn và làm hóa mô miễn dịch.")
        else:
            st.success("**NGUY CƠ THẤP:** Mẫu bệnh phẩm hiện tại chưa ghi nhận dấu hiệu ác tính đáng kể.")
            st.info("Theo dõi định kỳ theo phác đồ chuẩn.")
        
        # Thanh tiến trình rủi ro
        st.progress(prob)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- PHẦN MINH CHỨNG (HÌNH ẢNH) ---
    st.subheader("📍 Các vùng cần lưu ý (Attention Heatmaps)")
    st.write("Hệ thống tự động đánh dấu các mảnh cắt có trọng số quyết định cao nhất đến kết quả chẩn đoán.")
    
    raw_scores = attention.cpu().numpy().flatten()
    norm_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-8)
    
    n_display = min(4, len(uploaded_files))
    cols = st.columns(n_display)
    top_indices = np.argsort(raw_scores)[-n_display:][::-1]
    
    for i, idx in enumerate(top_indices):
        with cols[i]:
            patch_img = Image.open(uploaded_files[idx])
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(patch_img)
            
            # Overlay tinh tế hơn
            overlay = np.zeros((*np.array(patch_img).shape[:2], 3))
            overlay[:] = [0.8, 0, 0] 
            ax.imshow(overlay, alpha=norm_scores[idx] * 0.35) 
            
            ax.set_title(f"Vùng quan trọng #{i+1}\nTrọng số: {raw_scores[idx]:.3f}", fontsize=9)
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig)

else:
    # Màn hình chờ
    st.markdown("""
        <div style='text-align: center; padding: 50px; color: #6c757d;'>
            <img src='https://flaticon.com' width='100' style='opacity: 0.5;'>
            <h3>Sẵn sàng phân tích</h3>
            <p>Vui lòng tải lên các mảnh cắt bệnh phẩm để bắt đầu quy trình hỗ trợ chẩn đoán.</p>
        </div>
    """, unsafe_allow_html=True)

# Footer chuyên nghiệp
st.markdown("---")
st.caption("© 2024 PathoInsight AI System | Dành riêng cho mục đích nghiên cứu và hỗ trợ lâm sàng.")
