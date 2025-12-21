# Force Update: 2025-12-21 Stable Master Version (Fixed Rerun & Analysis Stickiness)
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
from PIL import Image, ImageOps

# ==========================================
# 0. 加入快取機制：防止結果跳掉
# ==========================================
@st.cache_data(show_spinner=False)
def process_ai_analysis(img_array, threshold):
    # 將後台複雜運算鎖定，避免 Rerun 消失
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(img_array)
        if not results.multi_face_landmarks:
            return None
        
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = img_array.shape
        
        # 1. 臉型運算
        cheek_width = math.sqrt(((landmarks[454].x - landmarks[234].x)*w)**2 + ((landmarks[454].y - landmarks[234].y)*h)**2)
        face_height = math.sqrt(((landmarks[152].x - landmarks[10].x)*w)**2 + ((landmarks[152].y - landmarks[10].y)*h)**2)
        jaw_width = math.sqrt(((landmarks[288].x - landmarks[58].x)*w)**2 + ((landmarks[288].y - landmarks[58].y)*h)**2)
        
        ratio_len_wid = face_height / cheek_width
        ratio_jaw_cheek = jaw_width / cheek_width
        
        # 2. 膚色運算
        cx, cy = int(landmarks[117].x * w), int(landmarks[117].y * h)
        cheek_crop = img_array[max(0, cy-10):min(h, cy+10), max(0, cx-10):min(w, cx+10)]
        is_warm = np.mean(cv2.cvtColor(cheek_crop, cv2.COLOR_RGB2LAB)[:,:,2]) > threshold if cheek_crop.size > 0 else True
        
        return {"ratio_len_wid": ratio_len_wid, "ratio_jaw_cheek": ratio_jaw_cheek, "is_warm": is_warm}

# ==========================================
# 1. 臉型與色彩文字邏輯
# ==========================================
def get_analysis_text(data):
    # 臉型邏輯
    if data["ratio_len_wid"] > 1.55:
        face = ("長臉 (Long)", "由於面部比例較長，建議透過橫向暈染腮紅與平直眉來截斷視覺長度，實現和諧比例。")
    elif data["ratio_len_wid"] < 1.15:
        if data["ratio_jaw_cheek"] > 0.85:
            face = ("方圓臉 (Soft Square)", "重點在於柔化下顎感。建議眉型帶點弧度，修容著重在腮幫轉折處，營造知性氛圍。")
        else:
            face = ("圓臉 (Round)", "建議加強T字部位提亮，眉毛要有明顯眉峰，腮紅由太陽穴向嘴角斜刷以拉長比例。")
    else:
        face = ("鵝蛋臉 (Oval)", "您的比例完美！幾乎能駕馭所有妝容，大膽嘗試各種風格，發揮您的優勢。")
    
    return face

# ==========================================
# 2. 網站設定與 CSS
# ==========================================
st.set_page_config(page_title="AI 個人形象顧問", page_icon="✨", layout="centered")

st.markdown("""
    <style>
        html, body, .stApp, h1, h2, h3, h4, h5, h6, p, div, span, li { color: #333333 !important; font-family: 'PingFang TC', sans-serif; }
        .stCard { background-color: rgba(255, 255, 255, 0.98); padding: 25px; border-radius: 18px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); margin-bottom: 20px; }
        .badge { display: inline-block; padding: 5px 15px; border-radius: 25px; font-size: 14px; font-weight: 600; color: white !important; margin-bottom: 10px; }
        .makeup-card { margin-top: 15px; padding: 15px; border-radius: 10px; border: 1px solid #eee; background-color: #fafafa; }
        .best-reason, .avoid-reason { padding: 15px; border-radius: 10px; font-size: 14px; line-height: 1.6; margin-top: 10px; }
        .best-reason { background-color: #f5fff5; border-left: 5px solid #28a745; }
        .avoid-reason { background-color: #fff5f5; border-left: 5px solid #ff4b4b; }
    </style>
""", unsafe_allow_html=True)

st.title("✨ AI 個人形象顧問")

with st.sidebar:
    st.header("⚙️ 設定")
    threshold = st.slider("冷暖分界值", 135, 155, 143)
    st.button("清除快取", on_click=st.cache_data.clear)

uploaded_file = st.file_uploader("📸 選擇照片", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")

if uploaded_file:
    image = np.array(ImageOps.exif_transpose(Image.open(uploaded_file)).convert('RGB'))
    
    with st.spinner("專業鑑定中..."):
        data = process_ai_analysis(image, threshold)

    if data:
        face_name, shape_advice = get_analysis_text(data)
        is_warm = data["is_warm"]
        
        if is_warm:
            theme_color, theme_bg = "#FF8C69", "#FFF8F5"
            tone_short, tone_type = "暖皮", "暖色調 (Warm Tone)"
            best_colors = [("#FF8C69", "珊瑚橘"), ("#E1AD01", "芥末黃"), ("#9F3025", "磚紅色"), ("#708238", "橄欖綠")]
            avoid_colors = [("#808080", "冷灰色"), ("#4169E1", "寶藍色"), ("#FF00FF", "芭比粉")]
            best_tips = "✨ **顯色原理**：暖皮肌膚與暖調色彩能產生同類色互襯，讓氣色顯得飽滿，散發健康光澤。"
            avoid_tips = "❌ **避雷指南**：避免冷灰色，這會讓您肌膚看起來發青、暗沉。"
            quick_makeup, makeup_tip = "日系元氣果汁妝", "建議底妝奶油肌，腮紅著重眼下暈染。"
            base_detail = "選用暖沙色底妝，避開偏灰冷色，用暖棕修容。"
            point_detail = "眼影用肉桂色，金檳色提亮臥蠶。"
        else:
            theme_color, theme_bg = "#87CEEB", "#F5F9FF"
            tone_short, tone_type = "冷皮", "冷色調 (Cool Tone)"
            best_colors = [("#F7C5D0", "玫瑰粉"), ("#87CEEB", "天空藍"), ("#B57EDC", "薰衣草"), ("#800020", "波爾多紅")]
            avoid_colors = [("#FFA500", "鮮橘色"), ("#FFDB58", "亮黃色"), ("#C19A6B", "土黃色")]
            best_tips = "✨ **顯色原理**：冷皮肌膚與冷色調相遇能產生淨化作用，襯托出肌膚的通透度。"
            avoid_tips = "❌ **避雷指南**：高飽和橘色會反襯血管青紫感，顯得氣色蠟黃。"
            quick_makeup, makeup_tip = "韓式清透冷感妝", "底妝強調透亮感，用藕粉或灰粉色系。"
            base_detail = "選用粉調底妝，紫色飾底乳校色，影灰色修容。"
            point_detail = "眼影用灰粉色，珍珠色提亮臥蠶。"

        st.markdown(f"<style>.stApp {{ background-color: {theme_bg}; }}</style>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.image(image, use_column_width=True)
        with col2:
            st.markdown(f"""
            <div class="stCard">
                <span class="badge" style="background-color: {theme_color};">{tone_short}</span> <b>{tone_type}</b><br>
                <span class="badge" style="background-color: #555;">{face_name.split(' ')[0]}</span> <b>{face_name}</b>
                <div style="margin-top:10px; padding-top:10px; border-top: 1px solid #eee;">
                    <p style="font-size: 14px; line-height: 1.6; color: #555;">{shape_advice}</p>
                </div>
                <div class="makeup-card" style="border-left: 5px solid {theme_color};">
                    <b style="color: #333; font-size: 15px;">💄 推薦妝容方向</b><br>
                    <span style="font-size: 14px; color: {theme_color}; font-weight: bold;">{quick_makeup}</span><br>
                    <p style="font-size: 13px; color: #666; margin-top: 5px;">{makeup_tip}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        t1, t2, t3, t4 = st.tabs(["🎨 色彩建議", "📐 造型思路", "💄 妝容解析", "💍 配飾指南"])
        with t1:
            st.markdown("#### ✅ 您的命定顯白色系")
            c = st.columns(4)
            for i, (h, n) in enumerate(best_colors):
                with c[i]:
                    st.markdown(f"<div style='background-color:{h};height:70px;border-radius:10px;'></div>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align:center;font-size:12px;'>{n}<br>{h}</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='best-reason'>{best_tips}</div>", unsafe_allow_html=True)
            st.markdown("#### ❌ 應避免的地雷色系")
            c2 = st.columns(3)
            for i, (h, n) in enumerate(avoid_colors):
                with c2[i]:
                    st.markdown(f"<div style='background-color:{h};height:70px;border-radius:10px;'></div>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align:center;font-size:12px;'>{n}<br>{h}</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='avoid-reason'>{avoid_tips}</div>", unsafe_allow_html=True)
        
        with t3:
            st.markdown(f"#### 💄 針對「{tone_short}」設計的妝容解析")
            ca, cb = st.columns(2)
            with ca: st.markdown(f"<div class='stCard'><b>底妝與修容</b><br>{base_detail}</div>", unsafe_allow_html=True)
            with cb: st.markdown(f"<div class='stCard'><b>眼部與唇彩</b><br>{point_detail}</div>", unsafe_allow_html=True)

        with t4:
            st.markdown("#### 💍 配飾與眼鏡")
            metal = "金色、玫瑰金" if is_warm else "銀色、白金"
            glass = "圓框類" if "方" in face_name else "方框類" if "圓" in face_name else "百搭框型"
            st.markdown(f"<div class='stCard'><b>最佳材質：</b>{metal}<br><b>眼鏡推薦：</b>{glass}</div>", unsafe_allow_html=True)

    else:
        st.error("❌ 無法偵測到臉部，請重試。")