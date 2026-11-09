
from pathlib import Path
import cv2
import torch
import numpy as np
import xml.etree.ElementTree as ET

from PIL import Image
from torchvision import transforms

from model import ParkingCNN


# ==========================================
# 1. KHAI BÁO ĐƯỜNG DẪN
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "parking_cnn.pth"
DATA_DIR = BASE_DIR / "PKLot"

OUTPUT_DIR = BASE_DIR / "outputs" / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "parking_result.jpg"


# ==========================================
# 2. CẤU HÌNH THIẾT BỊ
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 128

CLASS_NAMES = ["Empty", "Occupied"]

print("Thiết bị:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ==========================================
# 3. KIỂM TRA MODEL
# ==========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy model: {MODEL_PATH}"
    )

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

print("Đã tải model thành công!")


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
# 5. CẮT ẢNH TỪ POLYGON
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

    # Tạo mask cho vùng polygon
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

    # Loại bỏ phần nằm ngoài polygon
    masked_crop = cv2.bitwise_and(
        crop,
        crop,
        mask=mask
    )

    # Resize ảnh
    resized = cv2.resize(
        masked_crop,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    return resized


# ==========================================
# 6. DỰ ĐOÁN TRẠNG THÁI Ô ĐỖ XE
# ==========================================

def predict_parking_space(crop):

    # OpenCV BGR -> RGB
    rgb = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(rgb)

    tensor = transform(pil_image)

    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

    predicted_index = predicted.item()

    predicted_class = idx_to_class[predicted_index]

    confidence_value = confidence.item()

    return predicted_class, confidence_value


# ==========================================
# 7. TÌM ẢNH VÀ FILE XML PHÙ HỢP
# ==========================================

image_xml_pairs = []

for xml_path in DATA_DIR.rglob("*.xml"):

    image_path = xml_path.with_suffix(".jpg")

    if not image_path.exists():
        continue

    try:

        tree = ET.parse(xml_path)

        root = tree.getroot()

        spaces = root.findall("space")

        labels = {
            space.get("occupied")
            for space in spaces
        }

        # Chọn ảnh có cả Empty và Occupied
        if "0" in labels and "1" in labels:
            image_xml_pairs.append(
                (image_path, xml_path)
            )

    except ET.ParseError:
        continue


if not image_xml_pairs:
    raise RuntimeError(
        "Không tìm thấy cặp ảnh JPG/XML phù hợp!"
    )


# Chọn ảnh đầu tiên tìm được
image_path, xml_path = image_xml_pairs[0]

print("\nẢnh đang xử lý:", image_path.name)
print("XML đang xử lý:", xml_path.name)


# ==========================================
# 8. ĐỌC ẢNH
# ==========================================

image = cv2.imread(str(image_path))

if image is None:
    raise RuntimeError("Không đọc được ảnh!")


# ==========================================
# 9. ĐỌC XML VÀ NHẬN DIỆN
# ==========================================

tree = ET.parse(xml_path)

root = tree.getroot()

spaces = root.findall("space")

empty_count = 0
occupied_count = 0
skipped_count = 0

for space in spaces:

    contour_element = space.find("contour")

    if contour_element is None:
        skipped_count += 1
        continue

    contour = []

    for point in contour_element.findall("point"):

        px = int(float(point.get("x")))
        py = int(float(point.get("y")))

        contour.append([px, py])

    if len(contour) < 3:
        skipped_count += 1
        continue

    points = np.array(
        contour,
        dtype=np.int32
    )

    # Cắt vùng ô đỗ xe
    crop = crop_parking_space(
        image,
        contour
    )

    if crop is None:
        skipped_count += 1
        continue

    # Dự đoán bằng CNN
    predicted_class, confidence = predict_parking_space(crop)

    # Chọn màu
    if predicted_class == "Empty":

        color = (0, 255, 0)
        empty_count += 1

    else:

        color = (0, 0, 255)
        occupied_count += 1

    # Vẽ polygon
    cv2.polylines(
        image,
        [points],
        isClosed=True,
        color=color,
        thickness=2
    )


# ==========================================
# 10. TÍNH TOÁN THỐNG KÊ
# ==========================================

total_count = empty_count + occupied_count

occupancy_rate = (
    occupied_count / total_count * 100
    if total_count > 0
    else 0
)

print("\n========== KẾT QUẢ NHẬN DIỆN ==========")

print("Tổng số chỗ đỗ:", total_count)
print("Số chỗ trống:", empty_count)
print("Số chỗ có xe:", occupied_count)

print(
    f"Tỷ lệ lấp đầy: {occupancy_rate:.2f}%"
)

print("Số vùng bị bỏ qua:", skipped_count)


# ==========================================
# 11. VẼ BẢNG THỐNG KÊ
# ==========================================

# Nền bảng thống kê
cv2.rectangle(
    image,
    (10, 10),
    (390, 165),
    (30, 30, 30),
    -1
)

# Tổng số chỗ
cv2.putText(
    image,
    f"Total spaces: {total_count}",
    (25, 42),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2
)

# Số chỗ trống
cv2.putText(
    image,
    f"Empty: {empty_count}",
    (25, 78),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2
)

# Số chỗ có xe
cv2.putText(
    image,
    f"Occupied: {occupied_count}",
    (25, 114),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 0, 255),
    2
)

# Tỷ lệ lấp đầy
cv2.putText(
    image,
    f"Occupancy: {occupancy_rate:.1f}%",
    (25, 148),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (255, 255, 255),
    2
)


# ==========================================
# 12. LƯU ẢNH KẾT QUẢ
# ==========================================

success = cv2.imwrite(
    str(OUTPUT_PATH),
    image
)

if success:

    print("\nĐã lưu ảnh kết quả tại:")
    print(OUTPUT_PATH)

else:

    print("\nKhông thể lưu ảnh kết quả!")
