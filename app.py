# Force Fix 2025-12-21 v2

# Force Update: 2025-12-21 Fix Image Bug
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
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
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

# 側邊欄
with st.sidebar:
    st.header("⚙️ 設定")
    threshold = st.slider("冷暖分界值", 135, 155, 143)
    st.info("💡 提示：請使用自然光、無濾鏡的正面照片，結果最準確。")

# 檔案上傳
uploaded_file = st.file_uploader("📸 請選擇照片", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")

if uploaded_file is None:
    st.markdown("""
    <div style="text-align: center; padding: 40px; background: #f8f9fa; border-radius: 12px; border: 2px dashed #ddd;">
        <h3 style="color: #888;">👆 請點擊上方按鈕上傳照片</h3>
        <p style="color: #aaa;">支援 JPG, PNG 格式</p>
    </div>
    """, unsafe_allow_html=True)
    
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
            
            # (B) 畫出臉部網格 (絕對不能加 annotated_image = ...)
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
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
                    style_desc = "您的骨骼感較強，非常適合強調輪廓的歐美妝容。重點在於利用光影修飾稜角，將面部特徵轉化為高級的時尚氣場，展現自信與力量感。"
                    makeup_base_text = "- <b>底妝</b>：選擇霧面且高遮瑕的粉底，打造無瑕畫布。<br>- <b>修容</b>：使用灰棕色修容粉，加強在顴骨下方凹陷處、下顎線轉角與太陽穴，加深五官立體度。"
                    makeup_point_text = "- <b>眉眼</b>：畫出眉峰明顯的「小挑眉」，眼妝可嘗試截斷式畫法(Cut Crease)或大地色加深眼窩。<br>- <b>唇妝</b>：選用裸土色或深紅色的霧面唇釉，可稍微畫出嘴唇邊界，打造豐滿厚唇。"

                elif "圓臉" in face_shape:
                    if is_warm:
                        style_name = "日系元氣風 (Japanese Energetic)"
                        style_desc = "圓臉給人親切可愛的感覺，搭配暖色調，最適合展現像陽光般溫暖的「果汁感」妝容。重點在於通透的底妝與大面積的腮紅運用。"
                        makeup_base_text = "- <b>底妝</b>：選用輕薄透亮的奶油肌粉底，保留肌膚原有的光澤。<br>- <b>腮紅</b>：這是靈魂！選用珊瑚或杏桃色，大面積橫掃在眼下與鼻樑中段（曬傷妝），立刻減齡五歲。"
                        makeup_point_text = "- <b>眉眼</b>：眉毛保持自然毛流感，眼影選用暖大地色或橘棕色，並強調臥蠶的亮度。<br>- <b>唇妝</b>：選用水光感的唇釉或變色唇膏，打造像果凍般Q彈的嘟嘟唇。"
                    else:
                        style_name = "韓系清透風 (Korean Clean)"
                        style_desc = "即現在流行的「白開水妝容」。圓臉搭配冷色調，適合走氣質、乾淨的路線。重點在於低飽和度的色彩與極致乾淨的底妝，營造原生美女的氛圍。"
                        makeup_base_text = "- <b>底妝</b>：必須是水光肌！使用保濕型氣墊粉餅，並在額頭、鼻尖、下巴使用液態打亮。<br>- <b>修容</b>：減少臉側修容，僅需輕掃髮際線，避免妝感過髒。"
                        makeup_point_text = "- <b>眉眼</b>：畫出柔和的平眉或自然彎眉，眼線僅畫內眼線拉出眼尾。眼影選用低飽和的藕粉色或消腫色。<br>- <b>唇妝</b>：選用玫瑰色或莓果色的染唇液，由內向外暈染做出漸層咬唇妝。"

                else: # 心形臉或鵝蛋臉
                    if is_warm:
                        style_name = "泰式輕混血風 (Thai Soft Mixed)"
                        style_desc = "結合了歐美的深邃與亞洲的柔和，適合臉型精緻的暖色調肌膚。重點在於充滿生命力的「野生眉」與太陽曬過般的色彩，打造混血兒般的深邃感。"
                        makeup_base_text = "- <b>底妝</b>：微霧面妝感。比起修容，更建議使用古銅粉(Bronzer)輕掃臉周，增加溫暖的立體感。<br>- <b>腮紅</b>：選用土橘色或奶茶色，從顴骨向太陽穴斜刷，兼具修容效果。"
                        makeup_point_text = "- <b>眉眼</b>：使用眉膠將眉毛毛流向上梳理（野生眉）。睫毛要強調根根分明（太陽花睫毛）。<br>- <b>唇妝</b>：磚紅色、泰奶色或紅棕色的霧面口紅是最佳選擇。"
                    else:
                        style_name = "韓系女團風 (Korean Idol)"
                        style_desc = "適合上鏡的精緻妝容！針對標準臉型與冷皮，重點在於放大雙眼與使用「膨脹色」腮紅，讓面部更加飽滿立體，像偶像一樣閃閃發光。"
                        makeup_base_text = "- <b>底妝</b>：白皙透亮。建議使用紫色飾底乳校正膚色。<br>- <b>腮紅</b>：選用牛奶粉或薰衣草紫等「膨脹色」，打在面中與蘋果肌，讓臉型更澎潤。"
                        makeup_point_text = "- <b>眉眼</b>：眼妝重點在於「亮片」與「睫毛」。使用液體亮片點綴眼皮中央與臥蠶，睫毛刷成束狀。<br>- <b>唇妝</b>：高飽和度的櫻桃紅或草莓粉，質地要水亮鏡面，打造視覺焦點。"

                # ================= 介面顯示開始 =================
                
                # --- 動態背景色應用 ---
                st.markdown(f"""
                <style>
                .stApp {{
                    background-color: {theme_bg};
                    transition: background-color 0.5s ease;
                }}
                </style>
                """, unsafe_allow_html=True)
                
                col_img, col_info = st.columns([1, 1.5])
                
                with col_img:
                    # 顯示圖片 (現在 annotated_image 不會是 None 了)
                    st.image(annotated_image, use_container_width=True)
                
                with col_info:
                    st.markdown(f"""
                    <div style="background: white; padding: 20px; border-radius: 15px; border: 1px solid #eee; height: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                        <div style="margin-bottom: 15px;">
                            <span style="color: #888; font-size: 12px; display: block; margin-bottom: 4px;">您的膚色</span>
                            <span class="badge" style="background-color: {theme_color};">{tone_short}</span>
                            <span style="font-size: 18px; font-weight: bold; color: #333;">{tone_type}</span>
                        </div>
                        <div style="margin-bottom: 15px;">
                            <span style="color: #888; font-size: 12px; display: block; margin-bottom: 4px;">您的臉型</span>
                            <span class="badge" style="background-color: #6c757d;">{face_shape.split(' ')[0]}</span>
                            <span style="font-size: 18px; font-weight: bold; color: #333;">{face_shape}</span>
                        </div>
                        <div>
                            <span style="color: #888; font-size: 12px; display: block; margin-bottom: 4px;">推薦風格</span>
                            <div style="background: {theme_bg}; padding: 10px; border-radius: 10px; border-left: 5px solid {theme_color};">
                                <span style="font-size: 18px; font-weight: bold; color: #333;">{style_name.split(' ')[0]}</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.divider()

                # --- 分頁內容 ---
                tab1, tab2, tab3, tab4 = st.tabs(["🎨 色彩鑑定", "📐 臉型修飾", "💄 妝容教學", "💍 飾品配件"])

                # Tab 1: 色彩
                with tab1:
                    st.subheader("✅ 你的命定顯白色 (Best)")
                    cols = st.columns(len(best_colors))
                    for i, color in enumerate(best_colors):
                        with cols[i]:
                            st.markdown(f"""
                            <div style="background-color: {color}; height: 80px; border-radius: 12px; margin-bottom: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"></div>
                            <p style="text-align: center; color: #888; font-size: 13px; font-family: monospace; margin: 0;">{color}</p>
                            """, unsafe_allow_html=True)
                    st.markdown("<p style='color: #666; font-size: 15px; margin-top: 10px;'>穿上這些顏色能讓氣色更紅潤明亮。</p>", unsafe_allow_html=True)
                    
                    st.divider()
                    
                    st.subheader("❌ 建議避免的地雷色 (Avoid)")
                    cols_avoid = st.columns(len(avoid_colors))
                    for i, color in enumerate(avoid_colors):
                        with cols_avoid[i]:
                            st.markdown(f"""
                            <div style="background-color: {color}; height: 80px; border-radius: 12px; margin-bottom: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"></div>
                            <p style="text-align: center; color: #888; font-size: 13px; font-family: monospace; margin: 0;">{color}</p>
                            """, unsafe_allow_html=True)
                    st.markdown("<p style='color: #666; font-size: 15px; margin-top: 10px;'>這些顏色容易讓膚色看起來暗沉或蠟黃。</p>", unsafe_allow_html=True)

                # Tab 2: 臉型
                with tab2:
                    st.markdown(f"#### {face_shape} 的黃金修飾法則")
                    st.info(shape_advice)

                # Tab 3: 妝容
                with tab3:
                    st.markdown(f"#### ✨ {style_name} 妝容解析")
                    st.write(style_desc)
                    st.write("") 
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"""
                        <div class="stCard">
                            <h5 style="margin: 0 0 10px 0;">🧖‍♀️ 底妝 & 修容</h5>
                            <div style="font-size: 14px; line-height: 1.6; color: #444;">
                                {makeup_base_text}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with c2:
                        st.markdown(f"""
                        <div class="stCard">
                            <h5 style="margin: 0 0 10px 0;">👁️ 眼妝 & 唇彩</h5>
                            <div style="font-size: 14px; line-height: 1.6; color: #444;">
                                {makeup_point_text}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                # Tab 4: 飾品與配件
                with tab4:
                    st.markdown("#### ✨ 專屬飾品與配件指南")
                    st.markdown("<p style='color: #666; margin-bottom: 20px;'>配件是造型的靈魂！根據您的膚色與臉型，我們為您挑選了最能襯托優點的款式。</p>", unsafe_allow_html=True)
                    
                    # 配件邏輯
                    metal_text = "金色 (Gold)、玫瑰金 (Rose Gold)" if is_warm else "銀色 (Silver)、白金 (White Gold)"
                    metal_desc = "您的暖色調肌膚與金色系飾品相襯，能散發奢華溫暖的光澤，避免銀色讓皮膚顯得蒼白。" if is_warm else "您的冷色調肌膚搭配銀色系飾品，能展現清新高雅的透亮感，避免金色顯得俗氣。"
                    
                    if "方臉" in face_shape:
                        glasses_rec = "圓框、橢圓形、飛行員眼鏡"
                        glasses_desc = "選用圓潤的線條來柔和下顎角，避免方形框讓臉看起來更寬。"
                    elif "圓臉" in face_shape:
                        glasses_rec = "方框、貓眼、幾何多邊形"
                        glasses_desc = "利用稜角分明的鏡框來打破圓潤感，增加臉部線條的立體度。"
                    elif "長臉" in face_shape:
                        glasses_rec = "大鏡框 (Oversized)、寬版方框"
                        glasses_desc = "選擇鏡片高度較大的款式，能在視覺上縮短中庭，平衡臉部長度。"
                    else: 
                        glasses_rec = "幾乎適合所有框型 (百搭)"
                        glasses_desc = "您的臉型非常標準，可以大膽嘗試各種流行款式！"

                    # 配件顯示介面
                    ac1, ac2 = st.columns(2)
                    with ac1:
                        st.markdown(f"""
                        <div class="stCard">
                            <h5 style="margin: 0 0 10px 0;">💍 命定飾品材質</h5>
                            <div style="font-size: 20px; font-weight: bold; color: {theme_color}; margin: 15px 0;">{metal_text}</div>
                            <p style="color: #444; font-size: 14px; line-height: 1.5;">{metal_desc}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with ac2:
                        st.markdown(f"""
                        <div class="stCard">
                            <h5 style="margin: 0 0 10px 0;">👓 顯瘦眼鏡款式</h5>
                            <div style="font-size: 20px; font-weight: bold; color: #333; margin: 15px 0;">{glasses_rec}</div>
                            <p style="color: #444; font-size: 14px; line-height: 1.5;">{glasses_desc}</p>
                        </div>
                        """, unsafe_allow_html=True)

        else:
            st.error("❌ 找不到臉部！請換一張正面清晰的照片試試看。")