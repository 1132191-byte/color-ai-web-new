# Force Update: 2025-12-21 Final Fix v3
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
from PIL import Image, ImageOps

# ==========================================
# 0. 初始化 MediaPipe 繪圖工具
# ==========================================
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# ==========================================
# 1. 核心邏輯區 (數學計算)
# ==========================================

def get_distance(p1, p2, width, height):
    x1, y1 = p1.x * width, p1.y * height
    x2, y2 = p2.x * width, p2.y * height
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def analyze_face_shape(landmarks, width, height):
    # 取得關鍵點座標
    cheek_width = get_distance(landmarks[234], landmarks[454], width, height)
    face_height = get_distance(landmarks[10], landmarks[152], width, height)
    jaw_width = get_distance(landmarks[58], landmarks[288], width, height)
    
    # 計算比例
    ratio_len_wid = face_height / cheek_width
    ratio_jaw_cheek = jaw_width / cheek_width
    
    # 判斷邏輯
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

# --- CSS 注入 ---
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
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. 主程式介面
# ==========================================

st.title("✨ AI 個人形象顧問")
st.markdown("<p style='font-size: 16px; color: #666;'>上傳一張自拍，讓我為你打造專屬的造型攻略</p>", unsafe_allow_html=True)

# 側邊欄
with st.sidebar:
    st.header("⚙️ 設定")
    threshold = st.slider("冷暖分界值", 135, 155, 143)
    st.info("💡 提示：請使用自然光、無濾鏡的正面照片，結果最準確。")

# 檔案上傳
uploaded_file = st.file_uploader("📸 請選擇照片", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")

if uploaded_file is None:
    st.info("👆 請點擊上方按鈕上傳照片")
    
else:
    # --- 讀取圖片 ---
    try:
        image_source = Image.open(uploaded_file)
        image_source = ImageOps.exif_transpose(image_source) 
        pil_image = image_source.convert('RGB')
        pil_image.thumbnail((800, 800)) 
        image = np.array(pil_image)
    except Exception as e:
        st.error("圖片讀取失敗，請換一張試試看")
        st.stop()

    # --- AI 分析 ---
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
            
            # 2. 膚色取樣點
            idx = 117 
            cx, cy = int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)
            
            # --- 【關鍵修復區】繪圖 ---
            annotated_image = image.copy()
            
            # (A) 畫出膚色取樣點
            cv2.circle(annotated_image, (cx, cy), 15, (255, 255, 255), 2)
            cv2.circle(annotated_image, (cx, cy), 13, (255, 255, 255), -1) 
            
            # (B) 畫出臉部網格 (正確寫法：前面沒有變數接收回傳值)
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
            # (C) 畫出臉部輪廓 (正確寫法：前面沒有變數接收回傳值)
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            
            # 3. 膚色分析邏輯
            cheek_crop = image[max(0, cy-10):min(h, cy+10), max(0, cx-10):min(w, cx+10)]
            
            if cheek_crop.size > 0:
                lab_skin = cv2.cvtColor(cheek_crop, cv2.COLOR_RGB2LAB)
                b_value = np.mean(lab_skin[:,:,2])
                is_warm = b_value > threshold
                
                # --- 設定變數：色彩與風格 ---
                if is_warm:
                    theme_color = "#FF8C69" # 珊瑚橘
                    theme_bg = "#FFF5F2"    # 淺橘背景
                    tone_type = "暖色調 (Warm)"
                    tone_short = "暖皮"
                    best_colors = ["#FF8C69", "#E1AD01", "#9F3025", "#708238"] 
                    avoid_colors = ["#808080", "#4169E1", "#FF00FF"] 
                else:
                    theme_color = "#87CEEB" # 天空藍
                    theme_bg = "#F0F8FF"    # 淺藍背景
                    tone_type = "冷色調 (Cool)"
                    tone_short = "冷皮"
                    best_colors = ["#F7C5D0", "#87CEEB", "#B57EDC", "#800020"]
                    avoid_colors = ["#FFA500", "#FFDB58", "#C19A6B"]

                # --- 設定變數：妝容與建議 ---
                if "方臉" in face_shape or "長臉" in face_shape:
                    style_name = "歐美個性風 (Western Chic)"
                    style_desc = "您的骨骼感較強，非常適合強調輪廓的歐美妝容。"
                    makeup_base_text = "- <b>底妝</b>：霧面高遮瑕。<br>- <b>修容</b>：加強顴骨與下顎線。"
                    makeup_point_text = "- <b>眉眼</b>：小挑眉、截斷式眼妝。<br>- <b>唇妝</b>：裸土色或深紅霧面唇釉。"

                elif "圓臉" in face_shape:
                    if is_warm:
                        style_name = "日系元氣風 (Japanese Energetic)"
                        style_desc = "圓臉搭配暖色調，適合像陽光般溫暖的「果汁感」妝容。"
                        makeup_base_text = "- <b>底妝</b>：輕薄透亮奶油肌。<br>- <b>腮紅</b>：眼下大面積橫掃。"
                        makeup_point_text = "- <b>眉眼</b>：自然毛流眉、暖大地色眼影。<br>- <b>唇妝</b>：果凍感嘟嘟唇。"
                    else:
                        style_name = "韓系清透風 (Korean Clean)"
                        style_desc = "即現在流行的「白開水妝容」，強調極致乾淨的底妝。"
                        makeup_base_text = "- <b>底妝</b>：水光肌，多用打亮。<br>- <b>修容</b>：輕掃髮際線即可。"
                        makeup_point_text = "- <b>眉眼</b>：柔和彎眉、低飽和眼影。<br>- <b>唇妝</b>：玫瑰色漸層咬唇妝。"

                else: # 心形臉或鵝蛋臉
                    if is_warm:
                        style_name = "泰式輕混血風 (Thai Soft Mixed)"
                        style_desc = "結合歐美深邃與亞洲柔和，打造混血兒般的深邃感。"
                        makeup_base_text = "- <b>底妝</b>：微霧面，使用古銅粉修容。<br>- <b>腮紅</b>：土橘色斜刷。"
                        makeup_point_text = "- <b>眉眼</b>：野生眉、太陽花睫毛。<br>- <b>唇妝</b>：泰奶色或磚紅霧面。"
                    else:
                        style_name = "韓系女團風 (Korean Idol)"
                        style_desc = "適合上鏡的精緻妝容，重點在於放大雙眼與膨脹色應用。"
                        makeup_base_text = "- <b>底妝</b>：白皙透亮。<br>- <b>腮紅</b>：牛奶粉或紫色膨脹色。"
                        makeup_point_text = "- <b>眉眼</b>：亮片點綴、束狀睫毛。<br>- <b>唇妝</b>：高飽和櫻桃紅鏡面唇。"

                # ================= 介面顯示開始 =================
                
                col_img, col_info = st.columns([1, 1.5])
                
                with col_img:
                    # 顯示圖片 (現在這裡一定會有東西！)
                    st.image(annotated_image, use_container_width=True)
                
                with col_info:
                    st.markdown(f"""
                    <div style="background: white; padding: 20px; border-radius: 15px; border: 1px solid #eee; height: 100%;">
                        <div style="margin-bottom: 15px;">
                            <span class="badge" style="background-color: {theme_color};">{tone_short}</span>
                            <span style="font-size: 18px; font-weight: bold; color: #333;">{tone_type}</span>
                        </div>
                        <div style="margin-bottom: 15px;">
                            <span class="badge" style="background-color: #6c757d;">{face_shape.split(' ')[0]}</span>
                            <span style="font-size: 18px; font-weight: bold; color: #333;">{face_shape}</span>
                        </div>
                        <div style="background: {theme_bg}; padding: 10px; border-radius: 10px; border-left: 5px solid {theme_color};">
                            <span style="font-size: 18px; font-weight: bold; color: #333;">{style_name.split(' ')[0]}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.divider()

                # --- 分頁內容 ---
                tab1, tab2, tab3 = st.tabs(["🎨 色彩鑑定", "📐 臉型修飾", "💄 妝容建議"])

                with tab1:
                    st.subheader("✅ 你的命定顯白色")
                    cols = st.columns(len(best_colors))
                    for i, color in enumerate(best_colors):
                        with cols[i]:
                            st.markdown(f"<div style='background-color: {color}; height: 60px; border-radius: 8px;'></div>", unsafe_allow_html=True)
                    
                with tab2:
                    st.info(shape_advice)

                with tab3:
                    st.write(style_desc)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**底妝 & 修容**<br>{makeup_base_text}", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"**眼妝 & 唇彩**<br>{makeup_point_text}", unsafe_allow_html=True)

        else:
            st.error("❌ 找不到臉部！請換一張正面清晰的照片試試看。")