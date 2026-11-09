
import streamlit as st
import torch
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd
import plotly.express as px
from datetime import datetime

from pathlib import Path
from PIL import Image
from torchvision import transforms

from src.model import ParkingCNN


# ==========================================
# 1. CẤU HÌNH GIAO DIỆN
# ==========================================

st.set_page_config(
    page_title="Parking Detection System",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 PARKING SPACE DETECTION")
st.write(
    "Hệ thống nhận diện và thống kê trạng thái "
    "các vị trí đỗ xe từ hình ảnh."
)

st.divider()


# ==========================================
# 2. ĐƯỜNG DẪN VÀ THIẾT BỊ
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "parking_cnn.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 128


# ==========================================
# 3. TẢI MODEL
# ==========================================

@st.cache_resource
def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    class_to_idx = checkpoint["class_to_idx"]

    idx_to_class = {
        value: key
        for key, value in class_to_idx.items()
    }

    model = ParkingCNN(num_classes=2).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, idx_to_class


if not MODEL_PATH.exists():
    st.error("Không tìm thấy model parking_cnn.pth")
    st.stop()

model, idx_to_class = load_model()


# ==========================================
# 4. TIỀN XỬ LÝ ẢNH
# ==========================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# 5. CẮT VÙNG ĐỖ XE
# ==========================================

def crop_parking_space(image, contour):

    points = np.array(
        contour,
        dtype=np.int32
    )

    x, y, w, h = cv2.boundingRect(points)

    if w <= 0 or h <= 0:
        return None

    crop = image[y:y+h, x:x+w]

    if crop.size == 0:
        return None

    mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    local_points = points - np.array([x, y])

    cv2.fillPoly(
        mask,
        [local_points],
        255
    )

    crop = cv2.bitwise_and(
        crop,
        crop,
        mask=mask
    )

    crop = cv2.resize(
        crop,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    return crop


# ==========================================
# 6. DỰ ĐOÁN
# ==========================================

def predict(crop):

    rgb = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(rgb)

    tensor = transform(pil_image)

    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

    class_name = idx_to_class[predicted.item()]

    return class_name, confidence.item()


# ==========================================
# 7. ĐỌC XML VÀ NHẬN DIỆN
# ==========================================

def detect_parking(image, xml_content):

    root = ET.fromstring(xml_content)

    spaces = root.findall("space")

    empty_count = 0
    occupied_count = 0
    skipped_count = 0

    result = image.copy()

    for space in spaces:

        contour_element = space.find("contour")

        if contour_element is None:
            skipped_count += 1
            continue

        contour = []

        for point in contour_element.findall("point"):

            x = int(float(point.get("x")))
            y = int(float(point.get("y")))

            contour.append([x, y])

        if len(contour) < 3:
            skipped_count += 1
            continue

        points = np.array(
            contour,
            dtype=np.int32
        )

        crop = crop_parking_space(
            image,
            contour
        )

        if crop is None:
            skipped_count += 1
            continue

        class_name, confidence = predict(crop)

        if class_name == "Empty":

            color = (0, 255, 0)
            empty_count += 1

        else:

            color = (0, 0, 255)
            occupied_count += 1

        cv2.polylines(
            result,
            [points],
            True,
            color,
            2
        )

    total = empty_count + occupied_count

    occupancy = (
        occupied_count / total * 100
        if total > 0 else 0
    )

    # Bảng thống kê
    cv2.rectangle(
        result,
        (10, 10),
        (390, 165),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        result,
        f"Total: {total}",
        (25, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        result,
        f"Empty: {empty_count}",
        (25, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        result,
        f"Occupied: {occupied_count}",
        (25, 114),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.putText(
        result,
        f"Occupancy: {occupancy:.1f}%",
        (25, 148),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    return result, empty_count, occupied_count, skipped_count


# ==========================================
# 8. GIAO DIỆN TẢI ẢNH
# ==========================================

st.subheader("📂 Upload parking lot data")

col1, col2 = st.columns(2)

with col1:

    uploaded_image = st.file_uploader(
        "Chọn ảnh bãi đỗ xe (JPG)",
        type=["jpg", "jpeg", "png"]
    )

with col2:

    uploaded_xml = st.file_uploader(
        "Chọn file XML tương ứng",
        type=["xml"]
    )


# ==========================================
# 9. XỬ LÝ VÀ HIỂN THỊ
# ==========================================

if uploaded_image is not None:

    image_pil = Image.open(uploaded_image).convert("RGB")

    st.subheader("Ảnh đầu vào")

    st.image(
        image_pil,
        use_container_width=True
    )

    if uploaded_xml is not None:

        if st.button(
            "🔍 START DETECTION",
            type="primary",
            use_container_width=True
        ):

            image = cv2.cvtColor(
                np.array(image_pil),
                cv2.COLOR_RGB2BGR
            )

            try:

                with st.spinner("Đang nhận diện chỗ đỗ xe..."):

                    result, empty, occupied, skipped = detect_parking(
                        image,
                        uploaded_xml.getvalue()
                    )

                result_rgb = cv2.cvtColor(
                    result,
                    cv2.COLOR_BGR2RGB
                )

                st.success("Nhận diện hoàn tất!")
                st.subheader("📊 Detection Result")

                st.image(
                    result_rgb,
                    use_container_width=True
                )

                total = empty + occupied

                occupancy = (
                    occupied / total * 100
                    if total > 0 else 0
                )

                # Lưu lịch sử nhận diện
                history_dir = BASE_DIR / "outputs"
                history_dir.mkdir(exist_ok=True)

                history_path = history_dir / "history.csv"

                history_record = pd.DataFrame([{
                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Image": uploaded_image.name,
                    "Total": total,
                    "Empty": empty,
                    "Occupied": occupied,
                    "Occupancy": round(occupancy, 2)
                }])

                if history_path.exists():
                    old_history = pd.read_csv(history_path)
                    history_record = pd.concat(
                        [old_history, history_record],
                        ignore_index=True
                    )

                history_record.to_csv(
                    history_path,
                    index=False,
                    encoding="utf-8-sig"
                )

                st.subheader("📈 Parking Statistics")

                c1, c2, c3, c4 = st.columns(4)

                c1.metric("Total spaces", total)

                c2.metric("Empty", empty)

                c3.metric("Occupied", occupied)

                c4.metric(
                    "Occupancy rate",
                    f"{occupancy:.1f}%"
                )

                st.progress(
                    occupancy / 100
                )

                st.caption(
                    f"Số vùng không xử lý được: {skipped}"
                )

                # Tải ảnh kết quả
                success, encoded_image = cv2.imencode(
                    ".jpg",
                    result
                )

                if success:

                    st.download_button(
                        label="⬇️ Download result image",
                        data=encoded_image.tobytes(),
                        file_name="parking_result.jpg",
                        mime="image/jpeg"
                    )

            except Exception as e:

                st.error(f"Lỗi khi nhận diện: {e}")

    else:

        st.info(
            "Vui lòng tải thêm file XML tương ứng với ảnh."
        )

else:

    st.info(
        "Hãy tải ảnh bãi đỗ xe và file XML để bắt đầu."
    )
    
# ==========================================
# PARKING STATISTICS DASHBOARD
# ==========================================

st.divider()

st.header("📊 Parking Statistics Dashboard")

# Chỉ hiển thị biểu đồ khi đã có kết quả nhận diện
if "empty" in locals() and "occupied" in locals():

    chart_data = pd.DataFrame({
        "Status": ["Empty", "Occupied"],
        "Spaces": [empty, occupied]
    })

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Parking Space Distribution")

        fig_pie = px.pie(
            chart_data,
            names="Status",
            values="Spaces",
            color="Status",
            color_discrete_map={
                "Empty": "#22c55e",
                "Occupied": "#ef4444"
            },
            hole=0.4
        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True
        )

    with col2:

        st.subheader("Number of Parking Spaces")

        fig_bar = px.bar(
            chart_data,
            x="Status",
            y="Spaces",
            color="Status",
            color_discrete_map={
                "Empty": "#22c55e",
                "Occupied": "#ef4444"
            },
            text="Spaces"
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True
        )


# ==========================================
# DETECTION HISTORY
# ==========================================

st.divider()

st.header("🕒 Detection History")

history_path = BASE_DIR / "outputs" / "history.csv"

if history_path.exists():

    history_df = pd.read_csv(history_path)

    st.dataframe(
        history_df.sort_values(
            by="Time",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    # Biểu đồ tỷ lệ lấp đầy qua các lần nhận diện
    if len(history_df) > 0:

        st.subheader("Occupancy Rate History")

        fig_history = px.line(
            history_df,
            x="Time",
            y="Occupancy",
            markers=True,
            title="Parking Occupancy Over Time"
        )

        fig_history.update_layout(
            yaxis_title="Occupancy (%)",
            xaxis_title="Detection Time"
        )

        st.plotly_chart(
            fig_history,
            use_container_width=True
        )

    # Tải lịch sử
    st.download_button(
        "⬇️ Download Detection History",
        data=history_df.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="parking_history.csv",
        mime="text/csv"
    )

else:

    st.info(
        "Chưa có lịch sử. Hãy thực hiện nhận diện ảnh trước."
    )



# ==========================================
# 10. THÔNG TIN HỆ THỐNG
# ==========================================




st.divider()

st.header("📊 Model Evaluation")

st.write(
    "Đánh giá hiệu suất mô hình CNN trên tập dữ liệu Test."
)

# Các chỉ số đánh giá
c1, c2, c3, c4 = st.columns(4)

c1.metric("Accuracy", "98.96%")
c2.metric("Precision", "97.76%")
c3.metric("Recall", "99.90%")
c4.metric("F1-score", "98.81%")

# Confusion Matrix
st.subheader("Confusion Matrix")

confusion_path = (
    BASE_DIR / "outputs" / "charts" / "confusion_matrix.png"
)

if confusion_path.exists():

    st.image(
        str(confusion_path),
        caption="Confusion Matrix - Parking CNN",
        use_container_width=True
    )

else:

    st.warning("Không tìm thấy Confusion Matrix.")

# Metrics chi tiết
metrics_path = (
    BASE_DIR / "outputs" / "charts" / "metrics.txt"
)

st.subheader("Detailed Evaluation")

if metrics_path.exists():

    with open(metrics_path, "r", encoding="utf-8") as file:

        metrics_content = file.read()

    st.code(metrics_content, language="text")


# ==========================================
# 11. THÔNG TIN HỆ THỐNG
# ==========================================

st.divider()

st.subheader("System Information")

st.write("Model: Parking CNN")
st.write("Image size: 128 x 128")
st.write("Classes: Empty / Occupied")
st.write(f"Device: {DEVICE}")


