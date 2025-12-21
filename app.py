# Force Update: 2025-12-21 Pro Makeup Consultant Version (Multi-Style Edition)
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
                st.markdown(f"#### ✨ 專屬風格建議")
                st.write("根據您的臉型與膚色，以下是為您量身打造的變美方向：")
                st.info(f"💡 給您的專屬 Tip：{shape_advice}")

            with tab3:
                st.markdown(f"#### 💄 各國妝容風格建議")
                m1, m2, m3 = st.columns(3)
                
                with m1:
                    st.markdown(f"""
                    <div class='stCard' style='min-height: 350px;'>
                        <b style='color:#E67E22; font-size:18px;'>欧美妝 (Western)</b><br><br>
                        <b>特點：</b>強烈輪廓感與力量感。<br>
                        <b>底妝：</b>全霧面持久底妝。<br>
                        <b>眉眼：</b>高挑挑眉，深邃眼窩修容，誇張睫毛。<br>
                        <b>唇妝：</b>飽滿唇線，霧面土色或深紅。
                    </div>
                    """, unsafe_allow_html=True)
                
                with m2:
                    st.markdown(f"""
                    <div class='stCard' style='min-height: 350px;'>
                        <b style='color:#9B59B6; font-size:18px;'>韓式妝 (Korean)</b><br><br>
                        <b>特點：</b>水嫩剔透，視覺減齡。<br>
                        <b>底妝：</b>奶油水光肌，強調澎潤感。<br>
                        <b>眉眼：</b>平直眉或自然原生眉，清透臥蠶。<br>
                        <b>唇妝：</b>果凍感唇釉，咬唇或漸層畫法。
                    </div>
                    """, unsafe_allow_html=True)
                
                with m3:
                    st.markdown(f"""
                    <div class='stCard' style='min-height: 350px;'>
                        <b style='color:#FF69B4; font-size:18px;'>日式妝 (Japanese)</b><br><br>
                        <b>特點：</b>溫柔透明，無辜氛圍。<br>
                        <b>底妝：</b>清透半霧面，保留肌膚質感。<br>
                        <b>眉眼：</b>柔和淺色眉，大面積眼下腮紅。<br>
                        <b>唇妝：</b>潤澤感粉嫩色系，強調自然唇形。
                    </div>
                    """, unsafe_allow_html=True)

            with tab4:
                metal = "金色、黃銅、玫瑰金" if is_warm else "銀色、白金、珍珠"
                metal_tips = "您的膚色在暖色調金屬的襯托下會顯得更有神采。" if is_warm else "冷色調的金屬能讓您的膚色看起來更加通透、有透明感。"
                
                if "方臉" in face_shape:
                    glasses = "大圓框、水滴形框"; g_tips = "用圓潤的鏡框線條來平衡下顎的硬朗感。"
                elif "圓臉" in face_shape:
                    glasses = "方框、貓眼框"; g_tips = "利用鏡框的幾何線條，在視覺上拉長臉型。"
                elif "長臉" in face_shape:
                    glasses = "寬大的粗框、大方框"; g_tips = "選擇有存在感的鏡框，能有效截斷長臉的視覺感。"
                else:
                    glasses = "百搭款 (多邊形框)"; g_tips = "您的臉型不需要刻意修飾，各種流行鏡框皆可嘗試。"

                ac1, ac2 = st.columns(2)
                with ac1:
                    st.markdown(f"<div class='stCard'><b>飾品選色指南</b><br><br><span style='font-size: 18px; color:{theme_color};'><b>{metal}</b></span><br><br>{metal_tips}</div>", unsafe_allow_html=True)
                with ac2:
                    st.markdown(f"<div class='stCard'><b>顯瘦鏡框推薦</b><br><br><span style='font-size: 18px;'><b>{glasses}</b></span><br><br>{g_tips}</div>", unsafe_allow_html=True)

        else:
            st.error("❌ 無法偵測到臉部，請使用正面清晰且無遮擋的照片重試。")