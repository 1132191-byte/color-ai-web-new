# Force Update: 2025-12-21 Final Master Version (Professional Makeup Details)
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
        return "長臉 (Long)", "由於面部縱向比例較長，建議透過髮型建立橫向視覺寬度。利用平直眉及橫向暈染的腮紅來截斷面部長度，瀏海能有效縮短中庭，實現和諧平衡的比例。"
    elif ratio_len_wid < 1.15:
        if ratio_jaw_cheek > 0.85:
            return "方圓臉 (Soft Square)", "重點在於柔化下顎骨骼感。建議眉型帶點溫柔的弧度來中和稜角，修容著重在腮幫轉折處，營造出大氣且具高級感的知性氛圍。"
        else:
            return "圓臉 (Round)", "要打破面部圓潤感，造型重點在於『拉長比例』。建議加強T字部位及下巴的提亮，眉毛要有明顯眉峰以增加稜角感，腮紅則建議由太陽穴向嘴角斜刷。"
    else:
        if ratio_jaw_cheek < 0.8:
            return "心形臉 (Heart)", "典型下巴尖但額頭較寬的臉型。建議畫溫柔的弧形眉以收窄上額寬度，唇妝適合外深內淺的漸層畫法，使面部視覺重心自然向下移。"
        else:
            return "鵝蛋臉 (Oval)", "您的面部比例非常完美！幾乎能駕馭所有風格。建議可以大膽嘗試強調眼部或唇部的重點妝容，充分發揮您均衡且優雅的先天優勢。"

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
        .avoid-reason { background-color: #fff5f5; border-left: 5px solid #ff4b4b; padding: 15px; border-radius: 10px; font-size: 14px; color: #666; line-height: 1.6; }
        .best-reason { background-color: #f5fff5; border-left: 5px solid #28a745; padding: 15px; border-radius: 10px; font-size: 14px; color: #666; line-height: 1.6; }
        .makeup-card { margin-top: 15px; padding: 15px; border-radius: 10px; border: 1px solid #eee; background-color: #fafafa; }
    </style>
""", unsafe_allow_html=True)

st.title("✨ AI 個人形象顧問")

with st.sidebar:
    st.header("⚙️ 設定")
    threshold = st.slider("冷暖分界值", 135, 155, 143)
    st.info("💡 專業提示：請確保照片在自然環境光下拍攝。強烈的濾鏡或側面陰影會導致 AI 對臉型及膚色的誤判。")

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
                tone_type, tone_short = "暖色調 (Warm Tone)", "暖皮"
                best_colors = [("#FF8C69", "珊瑚橘"), ("#E1AD01", "芥末黃"), ("#9F3025", "磚紅色"), ("#708238", "橄欖綠")]
                avoid_colors = [("#808080", "冷灰色"), ("#4169E1", "寶藍色"), ("#FF00FF", "芭比粉")]
                best_tips = "✨ **顯色原理**：暖皮肌膚含有較多黃色素，與珊瑚、暖金等帶黃調的色彩能達成和諧共振，透過「同類色互襯」讓氣色顯得紅潤飽滿，散發如暖陽般的健康光澤感。"
                avoid_tips = "❌ **避雷指南**：請避免帶有大量藍色基調的『冷灰』或『寶藍』，這些色彩會讓您的肌膚看起來瞬間發青、暗沉。過於螢光的『芭比粉』更會產生強烈的色偏，顯得妝感髒亂不乾淨。"
                quick_makeup, makeup_tip = "日系元氣果汁妝感", "建議底妝以輕薄的奶油肌為主，腮紅著重於眼下大面積暈染，營造出溫暖、充滿活力且好親近的親和感。"
            else:
                theme_color, theme_bg = "#87CEEB", "#F5F9FF"
                tone_type, tone_short = "冷色調 (Cool Tone)", "冷皮"
                best_colors = [("#F7C5D0", "玫瑰粉"), ("#87CEEB", "天空藍"), ("#B57EDC", "薰衣草"), ("#800020", "波爾多紅")]
                avoid_colors = [("#FFA500", "鮮橘色"), ("#FFDB58", "亮黃色"), ("#C19A6B", "土黃色")]
                best_tips = "✨ **顯色原理**：冷皮肌膚擁有藍色/粉色底色，與玫瑰粉、清透藍相遇時能產生『淨化作用』，壓制面部的暗黃感，襯托出肌膚的通透度與明亮度，打造高冷的氛圍美。"
                avoid_tips = "❌ **避雷指南**：極度飽和的『鮮橘』或『亮黃』是冷皮的禁忌，這些高飽和的黃暖調會反襯出您肌膚底層的青紫色血管，讓整個人顯得非常疲憊、氣色蠟黃且無神。"
                quick_makeup, makeup_tip = "韓式清透冷感妝容", "底妝強調極致的透亮與水光感，眼影建議使用低飽和的藕粉或灰粉色系，打造出如白開水般乾淨、高級且具備清冷氣質的妝效。"

            # --- 介面渲染 ---
            st.markdown(f"<style>.stApp {{ background-color: {theme_bg}; }}</style>", unsafe_allow_html=True)
            col_img, col_info = st.columns([1, 1.2])
            
            with col_img:
                st.image(image, use_column_width=True) 
            
            with col_info:
                st.markdown(f"""
                <div class="stCard">
                    <p style="color: #999; font-size: 13px; margin-bottom: 5px;">AI 個人形象鑑定報告</p>
                    <span class="badge" style="background-color: {theme_color};">{tone_short}</span> 
                    <b style="font-size: 18px;">{tone_type}</b><br>
                    <span class="badge" style="background-color: #555;">{face_shape.split(' ')[0]}</span> 
                    <b style="font-size: 18px;">{face_shape}</b>
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

            st.divider()
            tab1, tab2, tab3, tab4 = st.tabs(["🎨 色彩建議", "📐 造型思路", "💄 妝容解析", "💍 配飾指南"])

            with tab1:
                st.markdown(f"#### ✅ 您的命定顯白色系", unsafe_allow_html=True)
                cols = st.columns(4)
                for i, (hex_code, name) in enumerate(best_colors):
                    with cols[i]:
                        st.markdown(f"<div style='background-color: {hex_code}; height: 75px; border-radius: 12px; box-shadow: inset 0 0 8px rgba(0,0,0,0.1);'></div>", unsafe_allow_html=True)
                        st.markdown(f"<p class='color-tip'>{name}</p><p class='hex-code'>{hex_code}</p>", unsafe_allow_html=True)
                st.markdown(f"<div class='best-reason'>{best_tips}</div>", unsafe_allow_html=True)
                
                st.write("")
                st.markdown(f"#### ❌ 應避免的地雷色系", unsafe_allow_html=True)
                cols2 = st.columns(3)
                for i, (hex_code, name) in enumerate(avoid_colors):
                    with cols2[i]:
                        st.markdown(f"<div style='background-color: {hex_code}; height: 75px; border-radius: 12px; opacity: 0.85;'></div>", unsafe_allow_html=True)
                        st.markdown(f"<p class='color-tip'>{name}</p><p class='hex-code'>{hex_code}</p>", unsafe_allow_html=True)
                st.markdown(f"<div class='avoid-reason'>{avoid_tips}</div>", unsafe_allow_html=True)

            with tab2:
                st.markdown(f"#### ✨ 專屬變美思路指南")
                st.write(f"針對您的 **{face_shape}** 骨骼基礎，整體的造型核心應該圍繞『視覺焦點平衡』來展開。")
                st.info(f"💡 專業建議：{shape_advice}")

            with tab3:
                st.markdown(f"#### 💄 精緻妝容技術解析")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""
                    <div class='stCard'>
                        <b style='font-size:16px; color:#333;'>面部底妝與修容方案</b><br><br>
                        <span style='font-size:14px; color:#555; line-height:1.7;'>
                            <b>1. 質地選擇：</b>建議選用質地細膩的<b>半霧面底妝</b>，能有效平衡臉部油光，避免因光影亂反射導致臉部視覺膨脹。<br>
                            <b>2. 修容邏輯：</b>修容應著重於『深邃感』的建立。在咬肌轉折處與太陽穴外側輕掃灰棕色修容，並利用<b>提亮色</b>強調蘋果肌頂點與下巴，使臉型立體化。<br>
                            <b>3. 腮紅位法：</b>腮紅是調整臉型關鍵。由蘋果肌高點往斜後方暈染，能拉長線條並提升整體氣色。
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class='stCard'>
                        <b style='font-size:16px; color:#333;'>色彩應用與五官平衡</b><br><br>
                        <span style='font-size:14px; color:#555; line-height:1.7;'>
                            <b>1. 眼部神采：</b>眼影建議選用<b>命定顯白色系</b>的低飽和延伸色。適度加強睫毛根部的層次感，搭配細膩的<b>臥蠶提亮</b>，能視覺放大雙眼並增加立體深邃度。<br>
                            <b>2. 眉型雕塑：</b>眉毛應根據分析報告建議的弧度進行勾勒，保持自然的毛流感，能平衡上庭比例。<br>
                            <b>3. 唇色美學：</b>選用與腮紅同色系的唇色，維持妝面色彩的一致性與高級感。邊緣微暈染的畫法更能修飾唇形，讓五官看起來更和諧。
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

            with tab4:
                st.markdown("#### 💍 適合您的首飾與眼鏡框架")
                metal = "金色系 (Gold)、玫瑰金、霧面黃銅材質" if is_warm else "銀色系 (Silver)、白金、極光珍珠材質"
                metal_tips = "暖色肌膚在溫潤金屬光的照映下能增加高級感，讓皮膚看起來更有彈性。" if is_warm else "冷調肌膚搭配清冷金屬色能突顯膚質的純淨與高貴感。"
                
                if "方" in face_shape:
                    glasses = "大尺寸圓框、貓眼框或飛行員鏡框"; g_tips = "利用鏡框的曲率來中和面部骨感的銳利感，使臉型呈現完美的流線美。"
                elif "圓" in face_shape:
                    glasses = "稜角分明的方框、幾何型框架或貓眼框"; g_tips = "利用眼鏡的直線條來人為製造稜角，打破圓臉的沉重感，增添俐落度。"
                elif "長" in face_shape:
                    glasses = "寬度顯著的大方框、寬版粗框眼鏡"; g_tips = "選擇鏡框高度較大的款式可以截斷長臉的縱向延伸，在視覺上縮短臉部比例。"
                else:
                    glasses = "多邊形框架、各式前衛流行款式皆可"; g_tips = "您的臉型屬於百搭款式，可以大膽嘗試各類實驗性設計框架，根據穿搭風格隨意變換。"

                ac1, ac2 = st.columns(2)
                with ac1:
                    st.markdown(f"<div class='stCard'><b>最佳金屬材質建議</b><br><b style='color:{theme_color}; font-size:16px;'>{metal}</b><br><br>{metal_tips}</div>", unsafe_allow_html=True)
                with ac2:
                    st.markdown(f"<div class='stCard'><b>顯瘦眼鏡款式指南</b><br><b style='font-size:16px;'>{glasses}</b><br><br>{g_tips}</div>", unsafe_allow_html=True)

        else:
            st.error("❌ 無法偵測到臉部特徵。請上傳一張正面、清晰、無遮擋且光源均勻的人像照片。")