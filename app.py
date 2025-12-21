# Force Update: 2025-12-21 Clean Version (No Detection Overlay)
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
from PIL import Image, ImageOps

# ==========================================
# 0. 初始化 MediaPipe 工具 (僅用於後台分析)
# ==========================================
mp_face_mesh = mp.solutions.face_mesh

# ==========================================
# 1. 核心邏輯區 (數學計算)
# ==========================================

def get_distance(p1, p2, width, height):
    x1, y1 = p1.x * width, p1.y * height
    x2, y2 = p2.x * width, p2.y * height
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def analyze_face_shape(landmarks, width, height):
    cheek_width = get_distance(landmarks[234], landmarks[454], width, height)
    face_height = get_distance(landmarks[10], landmarks[152], width, height)
    jaw_width = get_distance(landmarks[58], landmarks[288], width, height)
    
    ratio_len_wid = face_height / cheek_width
    ratio_jaw_cheek = jaw_width / cheek_width
    
    if ratio_len_wid > 1.55:
        return "長臉 (Long)", "利用瀏海縮短臉型，眉毛畫平直，腮紅橫向掃，能有效平衡比例。"
    elif ratio_len_wid < 1.15:
        if ratio_jaw_cheek > 0.9:
            return "方臉 (Square)", "強調下顎線修容，眉峰帶角度，適合歐美大氣妝容。"
        else:
            return "圓臉 (Round)", "加強T字部位打亮，眉毛要有眉峰拉長臉型，腮紅斜刷。"
    else:
        if ratio_jaw_cheek < 0.8:
            return "心形臉 (Heart)", "下巴偏尖，適合柔和眉型，修飾額頭兩側。"
        else:
            return "鵝蛋臉 (Oval)", "完美臉型！幾乎適合所有妝容，可大膽嘗試各種風格。"

# ==========================================
# 2. 網站基本設定與 CSS 美化
# ==========================================
st.set_page_config(page_title="AI 個人形象顧問", page_icon="✨", layout="centered")

st.markdown("""
    <style>
        html, body, .stApp, h1, h2, h3, h4, h5, h6, p, div, span, li {
            color: #333333 !important;
            font-family: 'Helvetica Neue', 'Arial', sans-serif;
        }
        .stCard {
            background-color: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            margin-bottom: 20px;
            height: 100%;
        }
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
            margin-right: 5px;
            color: white !important; 
        }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            height: 45px;
            border-radius: 8px;
            font-weight: bold;
            font-size: 15px;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            color: #333333 !important;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. 主程式介面
# ==========================================

st.title("✨ AI 個人形象顧問")
st.markdown("<p style='font-size: 16px; color: #666;'>上傳一張自拍，讓我為你打造專屬的造型攻略</p>", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ 設定")
    threshold = st.slider("冷暖分界值", 135, 155, 143)
    # 【小提示補回】
    st.info("💡 提示：請使用自然光、無濾鏡的正面照片，結果最準確。")

uploaded_file = st.file_uploader("📸 請選擇照片", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")

if uploaded_file is None:
    st.markdown("""
    <div style="text-align: center; padding: 40px; background: #f8f9fa; border-radius: 12px; border: 2px dashed #ddd;">
        <h3 style="color: #888;">👆 請點擊上方按鈕上傳照片</h3>
    </div>
    """, unsafe_allow_html=True)
    
else:
    try:
        image_source = Image.open(uploaded_file)
        image_source = ImageOps.exif_transpose(image_source) 
        pil_image = image_source.convert('RGB')
        pil_image.thumbnail((800, 800)) 
        image = np.array(pil_image)
    except Exception as e:
        st.error("圖片讀取失敗，請換一張試試看")
        st.stop()

    with mp_face_mesh.FaceMesh(
        static_image_mode=True, 
        max_num_faces=1,
        refine_landmarks=True, 
        min_detection_confidence=0.3) as face_mesh:

        with st.spinner("AI 正在掃描五官與膚色..."):
            results = face_mesh.process(image)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = image.shape
            
            # 1. 臉型分析
            face_shape, shape_advice = analyze_face_shape(face_landmarks.landmark, w, h)
            
            # 2. 膚色取樣 (純後台計算，不畫在圖上)
            idx = 117 
            cx, cy = int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)
            
            # --- 注意：此處不再對 annotated_image 畫任何圓圈或網格 ---
            # 直接使用原圖作為展示圖
            display_image = image.copy()
            
            cheek_crop = image[max(0, cy-10):min(h, cy+10), max(0, cx-10):min(w, cx+10)]
            
            if cheek_crop.size > 0:
                lab_skin = cv2.cvtColor(cheek_crop, cv2.COLOR_RGB2LAB)
                b_value = np.mean(lab_skin[:,:,2])
                is_warm = b_value > threshold
                
                # --- 設定色號 ---
                if is_warm:
                    theme_color = "#FF8C69" 
                    theme_bg = "#FFF5F2"
                    tone_type = "暖色調 (Warm)"
                    tone_short = "暖皮"
                    best_colors = ["#FF8C69", "#E1AD01", "#9F3025", "#708238"] 
                    avoid_colors = ["#808080", "#4169E1", "#FF00FF"] 
                else:
                    theme_color = "#87CEEB" 
                    theme_bg = "#F0F8FF"
                    tone_type = "冷色調 (Cool)"
                    tone_short = "冷皮"
                    best_colors = ["#F7C5D0", "#87CEEB", "#B57EDC", "#800020"]
                    avoid_colors = ["#FFA500", "#FFDB58", "#C19A6B"]

                # --- 設定風格 ---
                if "方臉" in face_shape or "長臉" in face_shape:
                    style_name = "歐美個性風 (Western Chic)"
                    style_desc = "您的骨骼感較強，非常適合強調輪廓的歐美妝容。"
                    makeup_base_text = "- <b>底妝</b>：霧面高遮瑕。<br>- <b>修容</b>：加強顴骨下方的陰影。"
                    makeup_point_text = "- <b>眉眼</b>：上揚挑眉與深邃眼影。<br>- <b>唇妝</b>：飽滿的裸色或紅棕色。"
                elif "圓臉" in face_shape:
                    if is_warm:
                        style_name = "日系元氣風 (Japanese Energetic)"
                        style_desc = "圓臉搭配暖色調，適合展現像陽光般溫暖的妝容。"
                        makeup_base_text = "- <b>底妝</b>：輕薄透亮的奶油肌。<br>- <b>腮紅</b>：珊瑚色大面積橫刷。"
                        makeup_point_text = "- <b>眉眼</b>：自然毛流感與暖色眼影。<br>- <b>唇妝</b>：水光感果凍唇。"
                    else:
                        style_name = "韓系清透風 (Korean Clean)"
                        style_desc = "適合走氣質、乾淨的「白開水妝容」路線。"
                        makeup_base_text = "- <b>底妝</b>：極致水光肌，加強打亮。<br>- <b>修容</b>：僅輕掃髮際線。"
                        makeup_point_text = "- <b>眉眼</b>：平緩柔和的眉毛。<br>- <b>唇妝</b>：玫瑰色系漸層咬唇。"
                else: 
                    style_name = "韓系女團風 (Korean Idol)" if not is_warm else "泰式輕混血風 (Thai Soft Mixed)"
                    style_desc = "適合您精緻的臉型，重點在於放大五官優點。"
                    makeup_base_text = "- <b>底妝</b>：精緻遮瑕，面中提亮。<br>- <b>修容</b>：自然的輪廓線條。"
                    makeup_point_text = "- <b>眉眼</b>：強調睫毛根根分明。<br>- <b>唇妝</b>：鮮豔且有層次感的唇彩。"

                # ================= 介面顯示 =================
                st.markdown(f"<style>.stApp {{ background-color: {theme_bg}; }}</style>", unsafe_allow_html=True)
                
                col_img, col_info = st.columns([1, 1.2])
                with col_img:
                    # 使用修正後的指令，且顯示的是原圖
                    st.image(display_image, use_column_width=True)
                
                with col_info:
                    st.markdown(f"""
                    <div style="background: white; padding: 20px; border-radius: 15px; border: 1px solid #eee;">
                        <p style="color: #888; font-size: 14px; margin-bottom: 5px;">分析結果</p>
                        <span class="badge" style="background-color: {theme_color};">{tone_short}</span>
                        <span style="font-size: 18px; font-weight: bold;">{tone_type}</span><br><br>
                        <span class="badge" style="background-color: #6c757d;">{face_shape.split(' ')[0]}</span>
                        <span style="font-size: 18px; font-weight: bold;">{face_shape}</span>
                    </div>
                    """, unsafe_allow_html=True)

                st.divider()

                tab1, tab2, tab3, tab4 = st.tabs(["🎨 色彩鑑定", "📐 臉型修飾", "💄 妝容建議", "💍 飾品配件"])

                with tab1:
                    st.subheader("✅ 你的命定顯白色")
                    cols = st.columns(4)
                    for i, color in enumerate(best_colors):
                        with cols[i]:
                            st.markdown(f"<div style='background-color: {color}; height: 60px; border-radius: 10px;'></div>", unsafe_allow_html=True)
                    st.subheader("❌ 建議避免的地雷色")
                    cols2 = st.columns(3)
                    for i, color in enumerate(avoid_colors):
                        with cols2[i]:
                            st.markdown(f"<div style='background-color: {color}; height: 60px; border-radius: 10px;'></div>", unsafe_allow_html=True)

                with tab2:
                    st.markdown(f"#### {face_shape} 修飾建議")
                    st.info(shape_advice)

                with tab3:
                    st.markdown(f"#### ✨ {style_name}")
                    st.write(style_desc)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"<div class='stCard'><b>底妝與修容</b><br>{makeup_base_text}</div>", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"<div class='stCard'><b>眉眼與唇妝</b><br>{makeup_point_text}</div>", unsafe_allow_html=True)

                with tab4:
                    st.markdown("#### ✨ 專屬飾品配件建議")
                    metal_text = "金色 (Gold)、玫瑰金" if is_warm else "銀色 (Silver)、白金"
                    metal_desc = "暖皮適合溫暖的金屬光澤。" if is_warm else "冷皮適合冷冽的銀亮質感。"
                    
                    if "方臉" in face_shape:
                        glasses = "圓框、飛行員鏡框"; g_desc = "柔和下顎線條。"
                    elif "圓臉" in face_shape:
                        glasses = "方框、貓眼鏡框"; g_desc = "增加臉部稜角感。"
                    elif "長臉" in face_shape:
                        glasses = "大框 (Oversized)"; g_desc = "視覺上縮短臉部長度。"
                    else:
                        glasses = "百搭款 (多邊形框)"; g_desc = "完美臉型幾乎不挑框。"

                    ac1, ac2 = st.columns(2)
                    with ac1:
                        st.markdown(f"<div class='stCard'><b>耳環/項鍊材質</b><br><span style='font-size: 18px; color:{theme_color};'><b>{metal_text}</b></span><br>{metal_desc}</div>", unsafe_allow_html=True)
                    with ac2:
                        st.markdown(f"<div class='stCard'><b>眼鏡款式推薦</b><br><span style='font-size: 18px;'><b>{glasses}</b></span><br>{g_desc}</div>", unsafe_allow_html=True)

        else:
            st.error("❌ 找不到臉部！請換一張正面清晰的照片。")