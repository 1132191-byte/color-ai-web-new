# Force Update: 2025-12-21 Pro Makeup Consultant Version (With Color Hex & Tips)
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
from PIL import Image, ImageOps

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
        return "長臉 (Long)", "為了縮短視覺長度，建議利用瀏海遮蓋額頭，眉型畫寬、畫平，避免挑高。腮紅從蘋果肌向耳側平掃，能有效拉寬中庭比例。"
    elif ratio_len_wid < 1.15:
        if ratio_jaw_cheek > 0.9:
            return "方臉 (Square)", "重點在於柔化下顎線條。建議眉峰帶點弧度，修容著重在腮幫子處，妝容色彩選用低飽和度，營造高級大氣的氛圍。"
        else:
            return "圓臉 (Round)", "要打破圓潤感，建議加強面中T字部位的提亮。眉毛要有明顯眉峰，腮紅則由太陽穴向嘴角斜刷，營造視覺上的消腫感。"
    else:
        if ratio_jaw_cheek < 0.8:
            return "心形臉 (Heart)", "下巴偏尖但額頭較寬。修容重點在太陽穴兩側，眉毛建議畫溫柔的弧形眉，唇妝適合畫出層次感的咬唇，平衡比例。"
        else:
            return "鵝蛋臉 (Oval)", "您的比例非常完美！各種妝容都能駕馭，建議根據心情嘗試不同風格，發揮您五官的先天優勢。"

# ==========================================
# 2. 網站設定與 CSS
# ==========================================
st.set_page_config(page_title="AI 個人形象顧問", page_icon="✨", layout="centered")

st.markdown("""
    <style>
        html, body, .stApp, h1, h2, h3, h4, h5, h6, p, div, span, li { color: #333333 !important; font-family: 'PingFang TC', 'Microsoft JhengHei', sans-serif; }
        .stCard { background-color: rgba(255, 255, 255, 0.98); padding: 25px; border-radius: 18px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); margin-bottom: 20px; }
        .badge { display: inline-block; padding: 5px 15px; border-radius: 25px; font-size: 14px; font-weight: 600; color: white !important; margin-bottom: 10px; }
        .color-tip { font-size: 13px; text-align: center; color: #444; margin-top: 8px; font-weight: 600; }
        .hex-code { font-size: 11px; text-align: center; color: #999; font-family: monospace; }
        .avoid-reason { background-color: #fff5f5; border-left: 5px solid #ff4b4b; padding: 10px; margin-top: 15px; font-size: 14px; color: #666; }
        .best-reason { background-color: #f5fff5; border-left: 5px solid #28a745; padding: 10px; margin-top: 15px; font-size: 14px; color: #666; }
    </style>
""", unsafe_allow_html=True)

st.title("✨ AI 個人形象顧問")
st.markdown("<p style='font-size: 16px; color: #888;'>上傳一張最真實的自拍，讓我為您分析專屬的變美思路</p>", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ 設定")
    threshold = st.slider("冷暖分界值", 135, 155, 143)
    st.info("💡 提示：請使用環境光均勻的照片，避免強烈濾鏡影響膚色判斷。")

uploaded_file = st.file_uploader("📸 選擇照片", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")

if uploaded_file:
    image = np.array(ImageOps.exif_transpose(Image.open(uploaded_file)).convert('RGB'))
    
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(image)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = image.shape
            face_shape, shape_advice = analyze_face_shape(face_landmarks.landmark, w, h)
            
            # 膚色分析
            idx = 117 
            cx, cy = int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)
            cheek_crop = image[max(0, cy-10):min(h, cy+10), max(0, cx-10):min(w, cx+10)]
            
            is_warm = np.mean(cv2.cvtColor(cheek_crop, cv2.COLOR_RGB2LAB)[:,:,2]) > threshold if cheek_crop.size > 0 else True
            
            # --- 風格與色號設定 ---
            if is_warm:
                theme_color, theme_bg = "#FF8C69", "#FFF8F5"
                tone_type, tone_short = "暖色調 (Warm)", "暖皮"
                best_colors = [("#FF8C69", "珊瑚橘"), ("#E1AD01", "芥末黃"), ("#9F3025", "磚紅色"), ("#708238", "橄欖綠")]
                avoid_colors = [("#808080", "冷灰"), ("#4169E1", "正藍"), ("#FF00FF", "芭比粉")]
                best_tips = "✨ 暖色調能與您肌膚中的黃色基調融合，讓氣色瞬間紅潤、顯現健康光澤。"
                avoid_tips = "❌ 這些高冷或帶螢光感的色調會與您的膚色產生衝突，容易顯得臉部發灰、暗沉蠟黃。"
            else:
                theme_color, theme_bg = "#87CEEB", "#F5F9FF"
                tone_type, tone_short = "冷色調 (Cool)", "冷皮"
                best_colors = [("#F7C5D0", "玫瑰粉"), ("#87CEEB", "天空藍"), ("#B57EDC", "薰衣草"), ("#800020", "波爾多紅")]
                avoid_colors = [("#FFA500", "鮮橘"), ("#FFDB58", "亮黃"), ("#C19A6B", "土黃")]
                best_tips = "✨ 冷色調能襯托出肌膚的通透度與明亮度，營造出一種優雅清冷的氛圍感。"
                avoid_tips = "❌ 帶有大量黃橘色調的飽和色會反襯出您肌膚的蠟黃感，遮蓋原本剔透的冷皮特質。"

            # --- 妝容具體內容 (專業彩妝師建議) ---
            if "方臉" in face_shape or "長臉" in face_shape:
                style_name = "氣場全開的歐美輕混血感"
                style_desc = "您擁有非常有張力的輪廓。化妝時不要試圖『掩蓋』骨骼，而是要利用它們，打造出具有高級感的層次。"
                makeup_base = "<b>底妝：</b>追求微霧面的絲絨感。比起大面積修容，更建議用深淺不同的粉底液進行『骨骼雕塑』。<br><b>修容：</b>重點加強顴骨下方的陰影，並在下顎線轉角處做暈染，讓臉部線條更有神采。"
                makeup_point = "<b>眼影：</b>大膽嘗試大地色系的截斷式畫法。眉毛建議畫出眉峰分明的挑眉。<br><b>唇膏：</b>選擇霧面質地的裸土色或紅棕色，甚至可以稍微畫出唇緣，增加視覺份量。"
            elif "圓臉" in face_shape:
                if is_warm:
                    style_name = "元氣滿滿的日系果汁感"
                    style_desc = "圓臉配暖皮是天生的親切感代名詞。妝容重點在於『清透感』與『大面積腮紅』，打造出溫柔好親近的氣質。"
                    makeup_base = "<b>底妝：</b>輕薄透亮的奶油肌。保留皮膚的原生質感，甚至一點點雀斑都會顯得自然。<br><b>腮紅：</b>這是靈魂。選擇珊瑚或杏桃色，在眼下與鼻頭處做圓向暈染，看起來像被太陽曬過的紅潤。"
                    makeup_point = "<b>眼影：</b>使用低飽和的暖大地色。重點在於強調纖長分明的睫毛。<br><b>唇膏：</b>透明感極強的水光唇釉，或是帶橘調的變色唇膏，打造飽滿豐潤感。"
                else:
                    style_name = "冷感氛圍的韓式開水妝"
                    style_desc = "圓臉配冷皮最適合走清冷、精緻的路線。減少顏色的複雜度，重點在於讓皮膚顯得極致乾淨、剔透。"
                    makeup_base = "<b>底妝：</b>乾淨的水光感底妝。建議使用紫色飾底乳校正黃氣。<br><b>腮紅：</b>選用牛奶粉或淡紫色的『膨脹色』腮紅，打在面中蘋果肌處，讓臉型瞬間立體消腫。"
                    makeup_point = "<b>眼影：</b>消腫色眼影（如藕粉色）平鋪。強調臥蠶的立體度。<br><b>唇膏：</b>帶有漿果調或玫瑰色的染唇液，由內向外暈染成自然的咬唇妝。"
            else:
                style_name = "精緻迷人的韓系女團風"
                style_desc = "您的臉型比例極佳，是典型的上鏡臉。妝容的核心在於強調『視覺中心點』，讓五官更加明豔動人。"
                makeup_base = "<b>底妝：</b>無瑕的半霧面質感。在面中三角區做重點提亮。<br><b>修容：</b>自然的鼻影過度與側影，重點修飾出下巴的線條感。"
                makeup_point = "<b>眼影：</b>加強大顆粒的偏光亮片點綴在眼窩中央。睫毛要強調『束狀感』。<br><b>唇膏：</b>飽滿的鏡面純釉。草莓粉或櫻桃色都能完美襯托您的氣場。"

            # =================介面顯示 =================
            st.markdown(f"<style>.stApp {{ background-color: {theme_bg}; }}</style>", unsafe_allow_html=True)
            col_img, col_info = st.columns([1, 1.2])
            with col_img:
                st.image(image, use_column_width=True) 
            with col_info:
                st.markdown(f"""
                <div class="stCard">
                    <p style="color: #999; font-size: 13px; margin-bottom: 5px;">AI 顧問分析報告</p>
                    <span class="badge" style="background-color: {theme_color};">{tone_short}</span> 
                    <b style="font-size: 18px;">{tone_type}</b><br>
                    <span class="badge" style="background-color: #555;">{face_shape.split(' ')[0]}</span> 
                    <b style="font-size: 18px;">{face_shape}</b>
                    <div style="margin-top:15px; padding-top:15px; border-top: 1px solid #eee;">
                        <p style="font-size: 14px; line-height: 1.6; color: #555;">{shape_advice}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.divider()
            tab1, tab2, tab3, tab4 = st.tabs(["🎨 色彩建議", "📐 造型思路", "💄 妝容解析", "💍 配飾指南"])

            with tab1:
                st.markdown(f"#### ✅ 推薦顯白色 <small style='color:#888'>Best Colors</small>", unsafe_allow_html=True)
                cols = st.columns(4)
                for i, (hex_code, name) in enumerate(best_colors):
                    with cols[i]:
                        st.markdown(f"<div style='background-color: {hex_code}; height: 70px; border-radius: 12px; box-shadow: inset 0 0 10px rgba(0,0,0,0.05);'></div>", unsafe_allow_html=True)
                        st.markdown(f"<p class='color-tip'>{name}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p class='hex-code'>{hex_code}</p>", unsafe_allow_html=True)
                st.markdown(f"<div class='best-reason'>{best_tips}</div>", unsafe_allow_html=True)

                st.write("")
                st.markdown(f"#### ❌ 避雷地雷色 <small style='color:#888'>Worst Colors</small>", unsafe_allow_html=True)
                cols2 = st.columns(3)
                for i, (hex_code, name) in enumerate(avoid_colors):
                    with cols2[i]:
                        st.markdown(f"<div style='background-color: {hex_code}; height: 70px; border-radius: 12px; opacity: 0.8;'></div>", unsafe_allow_html=True)
                        st.markdown(f"<p class='color-tip'>{name}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p class='hex-code'>{hex_code}</p>", unsafe_allow_html=True)
                st.markdown(f"<div class='avoid-reason'>{avoid_tips}</div>", unsafe_allow_html=True)

            with tab2:
                st.markdown(f"#### ✨ 專屬風格建議：{style_name}")
                st.write(style_desc)
                st.info(f"💡 給您的專屬 Tip：{shape_advice}")

            with tab3:
                st.markdown(f"#### 💄 變美思路解析")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"<div class='stCard'>{makeup_base}</div>", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"<div class='stCard'>{makeup_point}</div>", unsafe_allow_html=True)

            with tab4:
                metal = "金色、黃銅、玫瑰金" if is_warm else "銀色、白金、珍珠"