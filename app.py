# Force Update: 2025-12-21 Final Master Version (Fixed Accessory Tab & Report Card)
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
        return "長臉 (Long)", "為了縮短視覺長度，建議利用瀏海遮蓋額頭，眉型畫寬、畫平。腮紅從蘋果肌向耳側平掃，能有效平衡比例。"
    elif ratio_len_wid < 1.15:
        if ratio_jaw_cheek > 0.85: # 稍微調降門檻以更容易偵測方圓臉
            return "方圓臉 (Soft Square)", "重點在於柔化下顎線條。建議眉峰帶點弧度，修容著重在腮幫子處，營造柔和且具高級感的氛圍。"
        else:
            return "圓臉 (Round)", "要打破圓潤感，建議加強面中T字部位提亮。眉毛要有明顯眉峰，腮紅由太陽穴向嘴角斜刷。"
    else:
        if ratio_jaw_cheek < 0.8:
            return "心形臉 (Heart)", "下巴偏尖但額頭較寬。建議畫溫柔的弧形眉，唇妝適合畫出層次感的咬唇，平衡臉部比例。"
        else:
            return "鵝蛋臉 (Oval)", "您的比例非常完美！各種妝容都能駕馭，發揮您五官的先天優勢。"

# ==========================================
# 2. 網站設定與 CSS
# ==========================================
st.set_page_config(page_title="AI 個人形象顧問", page_icon="✨", layout="centered")

st.markdown("""
    <style>
        html, body, .stApp, h1, h2, h3, h4, h5, h6, p, div, span, li { color: #333333 !important; font-family: 'PingFang TC', sans-serif; }
        .stCard { background-color: rgba(255, 255, 255, 0.98); padding: 25px; border-radius: 18px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); margin-bottom: 20px; }
        .badge { display: inline-block; padding: 5px 15px; border-radius: 25px; font-size: 14px; font-weight: 600; color: white !important; margin-bottom: 10px; }
        .color-tip { font-size: 13px; text-align: center; color: #444; margin-top: 8px; font-weight: 600; }
        .hex-code { font-size: 11px; text-align: center; color: #999; font-family: monospace; }
        .avoid-reason { background-color: #fff5f5; border-left: 5px solid #ff4b4b; padding: 10px; margin-top: 15px; font-size: 14px; color: #666; }
        .best-reason { background-color: #f5fff5; border-left: 5px solid #28a745; padding: 10px; margin-top: 15px; font-size: 14px; color: #666; }
        .makeup-card { margin-top: 15px; padding: 12px; border-radius: 8px; border: 1px solid #eee; background-color: #fafafa; }
    </style>
""", unsafe_allow_html=True)

st.title("✨ AI 個人形象顧問")

with st.sidebar:
    st.header("⚙️ 設定")
    threshold = st.slider("冷暖分界值", 135, 155, 143)
    st.info("💡 提示：請使用正面環境光照片，避免強烈陰影導致臉型誤判。")

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
            
            if is_warm:
                theme_color, theme_bg = "#FF8C69", "#FFF8F5"
                tone_type, tone_short = "暖色調 (Warm)", "暖皮"
                best_colors = [("#FF8C69", "珊瑚橘"), ("#E1AD01", "芥末黃"), ("#9F3025", "磚紅色"), ("#708238", "橄欖綠")]
                avoid_colors = [("#808080", "冷灰"), ("#4169E1", "正藍"), ("#FF00FF", "芭比粉")]
                best_tips = "✨ 暖色調能與您肌膚的黃色基調融合，讓氣色瞬間紅潤、顯現健康光澤。"
                avoid_tips = "❌ 冷感太強的色調會讓暖皮看起來暗沉蠟黃，應盡量避免。"
                quick_makeup, makeup_tip = "日系元氣果汁妝", "建議使用暖橘或杏色調，大面積腮紅暈染。"
            else:
                theme_color, theme_bg = "#87CEEB", "#F5F9FF"
                tone_type, tone_short = "冷色調 (Cool)", "冷皮"
                best_colors = [("#F7C5D0", "玫瑰粉"), ("#87CEEB", "天空藍"), ("#B57EDC", "薰衣草"), ("#800020", "波爾多紅")]
                avoid_colors = [("#FFA500", "鮮橘"), ("#FFDB58", "亮黃"), ("#C19A6B", "土黃")]
                best_tips = "✨ 冷色調能襯托出肌膚的通透度與明亮度，營造出優雅清冷的氛圍。"
                avoid_tips = "❌ 暖黃色調會遮蓋冷皮的剔透特質，顯得臉部發灰。"
                quick_makeup, makeup_tip = "韓式清透冷感妝", "強調水光肌，搭配低飽和玫瑰藕粉色系。"

            # --- 介面渲染 ---
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
                    <div style="margin-top:10px; padding-top:10px; border-top: 1px solid #eee;">
                        <p style="font-size: 14px; line-height: 1.6; color: #555;">{shape_advice}</p>
                    </div>
                    <div class="makeup-card" style="border-left: 5px solid {theme_color};">
                        <b style="color: #333; font-size: 15px;">💄 適合您的妝容</b><br>
                        <span style="font-size: 14px; color: {theme_color}; font-weight: bold;">{quick_makeup}</span><br>
                        <p style="font-size: 13px; color: #666; margin: 0;">{makeup_tip}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.divider()
            tab1, tab2, tab3, tab4 = st.tabs(["🎨 色彩建議", "📐 造型思路", "💄 妝容解析", "💍 配飾指南"])

            with tab1:
                st.markdown(f"#### ✅ 推薦顯白色", unsafe_allow_html=True)
                cols = st.columns(4)
                for i, (hex_code, name) in enumerate(best_colors):
                    with cols[i]:
                        st.markdown(f"<div style='background-color: {hex_code}; height: 70px; border-radius: 12px;'></div>", unsafe_allow_html=True)
                        st.markdown(f"<p class='color-tip'>{name}</p><p class='hex-code'>{hex_code}</p>", unsafe_allow_html=True)
                st.markdown(f"<div class='best-reason'>{best_tips}</div>", unsafe_allow_html=True)

            with tab2:
                st.markdown(f"#### ✨ 專屬變美思路")
                st.write(f"針對您的 **{face_shape}**，造型核心在於調整視覺重心。")
                st.info(f"💡 造型小提示：{shape_advice}")

            with tab3:
                st.markdown(f"#### 💄 詳細妝容思路解析")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"<div class='stCard'><b>底妝與修容</b><br>建議選用與脖子膚色一致的底妝，並在面中三角區做重點提亮。</div>", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"<div class='stCard'><b>眼影與唇彩</b><br>選用低飽和度色彩，唇妝建議畫出自然漸層感。</div>", unsafe_allow_html=True)

            with tab4:
                st.markdown("#### 💍 適合您的配飾與眼鏡")
                metal = "金色、玫瑰金、黃銅材質" if is_warm else "銀色、白金、冷光珍珠"
                metal_tips = "暖皮適合溫暖的金屬光澤。" if is_warm else "冷皮適合清冷的銀亮質感。"
                
                if "方" in face_shape:
                    glasses = "大圓框、水滴形框"; g_tips = "用圓潤線條來平衡下顎的硬感。"
                elif "圓" in face_shape:
                    glasses = "方框、貓眼框"; g_tips = "利用鏡框稜角來增加臉部線條感。"
                elif "長" in face_shape:
                    glasses = "寬大框、大方框"; g_tips = "增加橫向面積以截斷視覺長度。"
                else:
                    glasses = "多邊形框、各式流行框型"; g_tips = "標準臉型幾乎不挑框型，可大膽嘗試。"

                ac1, ac2 = st.columns(2)
                with ac1:
                    st.markdown(f"<div class='stCard'><b>飾品材質</b><br><b style='color:{theme_color};'>{metal}</b><br><br>{metal_tips}</div>", unsafe_allow_html=True)
                with ac2:
                    st.markdown(f"<div class='stCard'><b>眼鏡推薦</b><br><b>{glasses}</b><br><br>{g_tips}</div>", unsafe_allow_html=True)

        else:
            st.error("❌ 無法偵測到臉部，請更換光線均勻的照片。")