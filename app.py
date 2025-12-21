# Force Update: 2025-12-21 Pro Edition (Full Color IDs & Expert Advice)
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
from PIL import Image, ImageOps

# ==========================================
# 1. 核心邏輯區
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
        return "長臉 (Long)", "您的面部縱向比例較長，造型核心在於『橫向拉伸』。建議利用平直眉與橫向掃過的腮紅來截斷視覺長度，髮型則適合帶弧度的瀏海。"
    elif ratio_len_wid < 1.15:
        if ratio_jaw_cheek > 0.9:
            return "方臉 (Square)", "您的輪廓自帶一種高級的冷冷感。妝容上不要刻意掩蓋下顎，而是用圓潤的眉峰與低飽和色的唇妝來平衡硬朗感，打造大氣美。"
        else:
            return "圓臉 (Round)", "您的骨骼線條柔和，顯得非常親切減齡。妝容上建議強化眉峰與面中的提亮（T字區），增加立體度，打破視覺上的圓潤感。"
    else:
        return "鵝蛋臉 (Oval)" if ratio_jaw_cheek > 0.8 else "心形臉 (Heart)", "您擁有非常標準的黃金比例！這意味著您可以大膽嘗試各種實驗性的妝容，幾乎沒有地雷區，重點在於突出五官細節。"

# ==========================================
# 2. 網站美化 CSS
# ==========================================
st.set_page_config(page_title="AI 個人形象顧問", page_icon="✨", layout="centered")

st.markdown("""
    <style>
        html, body, .stApp, h1, h2, h3, h4, h5, h6, p, div, span, li { color: #333333 !important; font-family: 'PingFang TC', sans-serif; }
        .stCard { background-color: white; padding: 25px; border-radius: 18px; box-shadow: 0 10px 20px rgba(0,0,0,0.05); margin-bottom: 20px; border: 1px solid #f0f0f0; }
        .badge { display: inline-block; padding: 5px 15px; border-radius: 25px; font-size: 14px; font-weight: 600; color: white !important; margin-bottom: 10px; }
        .color-id { font-size: 12px; font-family: monospace; color: #999; text-align: center; margin-top: 3px; }
        .color-name { font-size: 14px; font-weight: 600; text-align: center; margin-top: 5px; color: #444; }
        .avoid-box { background-color: #fdf2f2; padding: 15px; border-radius: 10px; border-left: 5px solid #f87171; font-size: 14px; margin-top: 15px; line-height: 1.6; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. 主介面
# ==========================================
st.title("✨ AI 個人形象顧問")
st.markdown("<p style='color:#777;'>專業級肤色與臉型鑑定，為您定制專屬的變美思路</p>", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ 鑑定參數")
    threshold = st.slider("冷暖平衡調整", 135, 155, 143)
    st.info("💡 建議上傳光線充足且不帶濃妝的照片。")

uploaded_file = st.file_uploader("📸 選擇照片", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")

if uploaded_file:
    # 讀取原圖 (處理旋轉問題)
    image = np.array(ImageOps.exif_transpose(Image.open(uploaded_file)).convert('RGB'))
    
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(image)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = image.shape
            face_shape, shape_advice = analyze_face_shape(face_landmarks.landmark, w, h)
            
            # 膚色取樣邏輯
            idx = 117
            cx, cy = int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)
            cheek_crop = image[max(0, cy-10):min(h, cy+10), max(0, cx-10):min(w, cx+10)]
            is_warm = np.mean(cv2.cvtColor(cheek_crop, cv2.COLOR_RGB2LAB)[:,:,2]) > threshold if cheek_crop.size > 0 else True
            
            # --- 色彩庫設定 ---
            if is_warm:
                theme_color, theme_bg = "#FF8C69", "#FFF9F6"
                tone_type, tone_short = "暖色調 (Warm)", "暖皮"
                best_colors = [("#FF8C69", "珊瑚橘"), ("#E1AD01", "芥末黃"), ("#B85233", "磚紅色"), ("#708238", "橄欖綠")]
                avoid_colors = [("#808080", "冷灰色"), ("#4169E1", "寶藍色"), ("#FF00FF", "死亡芭比粉")]
                avoid_reason = "⚠️ <b>避雷提醒：</b>暖色調肌膚最怕『冷感太強』或『螢光度過高』的顏色。這類冷灰與冷藍色會讓您的肌膚看起來瞬間變得蠟黃且沒有血色，而螢光粉則會產生視覺衝突，顯得膚色不夠乾淨。"
            else:
                theme_color, theme_bg = "#87CEEB", "#F5FAFF"
                tone_type, tone_short = "冷色調 (Cool)", "冷皮"
                best_colors = [("#F7C5D0", "玫瑰粉"), ("#87CEEB", "天空藍"), ("#B57EDC", "薰衣草"), ("#800020", "波爾多紅")]
                avoid_colors = [("#FFA500", "亮橘色"), ("#FFDB58", "亮黃色"), ("#C19A6B", "土黃色")]
                avoid_reason = "⚠️ <b>避雷提醒：</b>冷皮應盡量避免帶有大量黃調的顏色（如橘、亮黃）。這些顏色會像一面黃色的反光板，讓您原本透亮的冷白皮顯得暗沉灰暗，失去清冷高雅的氣場。"

            # --- 妝容建議邏輯 ---
            if "方臉" in face_shape or "長臉" in face_shape:
                makeup_title = "大氣優雅的輪廓妝感"
                m_base = "<b>底妝思路：</b>適合半霧面妝效。重點在於用略深半階的修容色，加強太陽穴與下顎角的視覺收縮。<br><b>修容：</b>在顴骨下方打出陰影感，營造高級的骨相美。"
                m_point = "<b>眼影：</b>眉毛畫出帶眉峰的小挑眉。眼影選用低飽和大地色，拉長眼尾。<br><b>唇膏：</b>適合全唇厚塗。裸色系或復古紅最能襯托氣場。"
            elif "圓臉" in face_shape:
                makeup_title = "精緻減齡的氛圍妝感"
                m_base = "<b>底妝思路：</b>追求透亮的奶油肌。重點在於面中（眼下三角、額頭、鼻尖）的提亮，增加立體度。<br><b>腮紅：</b>腮紅從眼下向斜上方掃，這能瞬間『削肉』，讓臉型看起來變長一點。"
                m_point = "<b>眼影：</b>強調臥蠶。眼影選用清透色彩，增加眼部靈動感。<br><b>唇膏：</b>適合水光感的唇釉，打造嬌滴滴的果凍唇，增加親和力。"
            else:
                makeup_title = "精緻立體的名伶妝感"
                m_base = "<b>底妝思路：</b>無暇清透底妝。根據您的膚色冷暖選擇飾底乳調色。<br><b>修容：</b>輕度修飾山根與鼻翼，突出五官的精緻度。"
                m_point = "<b>眼影：</b>睫毛是重點。睫毛刷成束狀感，讓眼神更有聚焦力。<br><b>唇膏：</b>適合飽滿的咬唇妝，或今年流行的泥質唇彩，展現溫柔氣質。"

            # ================= 介面渲染 =================
            st.markdown(f"<style>.stApp {{ background-color: {theme_bg}; }}</style>", unsafe_allow_html=True)
            
            c_img, c_res = st.columns([1, 1.2])
            with c_img:
                st.image(image, use_column_width=True) # 顯示完全乾淨的原圖
            with c_res:
                st.markdown(f"""
                <div class="stCard">
                    <span class="badge" style="background-color: {theme_color};">{tone_short}</span> <b>{tone_type}</b><br>
                    <span class="badge" style="background-color: #555;">{face_shape.split(' ')[0]}</span> <b>{face_shape}</b>
                    <p style="margin-top:10px; font-size: 14px; line-height: 1.6; color:#555;">{shape_advice}</p>
                </div>
                """, unsafe_allow_html=True)

            st.divider()
            t1, t2, t3, t4 = st.tabs(["🎨 色彩建議", "📐 造型思路", "💄 妝容解析", "💍 配飾指南"])

            with t1:
                st.markdown(f"#### ✅ 您的命定顯白色 <small style='color:#999;'>Best Matches</small>", unsafe_allow_html=True)
                bcols = st.columns(4)
                for i, (hex, name) in enumerate(best_colors):
                    with bcols[i]:
                        st.markdown(f"<div style='background-color:{hex}; height:75px; border-radius:12px; box-shadow:inset 0 0 10px rgba(0,0,0,0.05);'></div>", unsafe_allow_html=True)
                        st.markdown(f"<p class='color-name'>{name}</p><p class='color-id'>{hex}</p>", unsafe_allow_html=True)
                
                st.write("")
                st.markdown(f"#### ❌ 避雷地雷色 <small style='color:#999;'>Avoid These</small>", unsafe_allow_html=True)
                acols = st.columns(3)
                for i, (hex, name) in enumerate(avoid_colors):
                    with acols[i]:
                        st.markdown(f"<div style='background-color:{hex}; height:75px; border-radius:12px; opacity:0.8;'></div>", unsafe_allow_html=True)
                        st.markdown(f"<p class='color-name'>{name}</p><p class='color-id'>{hex}</p>", unsafe_allow_html=True)
                
                st.markdown(f"<div class='avoid-box'>{avoid_reason}</div>", unsafe_allow_html=True)

            with t2:
                st.markdown(f"#### ✨ 專屬風格定位：{makeup_title}")
                st.write(f"這套思路旨在突顯您的{face_shape}優勢，減少視覺負擔感。")
                st.info(f"建議：{shape_advice}")

            with t3:
                st.markdown("#### 💄 變美思路解析")
                mc1, mc2 = st.columns(2)
                with mc1:
                    st.markdown(f"<div class='stCard'>{m_base}</div>", unsafe_allow_html=True)
                with mc2:
                    st.markdown(f"<div class='stCard'>{m_point}</div>", unsafe_allow_html=True)

            with t4:
                metal = "金色、玫瑰金、黃銅材質" if is_warm else "銀色、白金、冷光珍珠"
                st.markdown(f"""
                <div class="stCard">
                    <b>首飾材質建議：</b><br>
                    <span style="font-size: 20px; color:{theme_color};"><b>{metal}</b></span><br><br>
                    <b>眼鏡款式推薦：</b><br>
                    <span style="font-size: 16px;"><b>{'大圓框或飛行員鏡框' if '方' in face_shape else '窄長方框或貓眼框' if '圓' in face_shape else '多邊形細黑框'}</b></span>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.error("❌ 無法偵測到臉部，請確保照片光線充足且臉部完整無遮擋。")