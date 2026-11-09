
from pathlib import Path
import cv2
import numpy as np
import random
import shutil
import xml.etree.ElementTree as ET

from sklearn.model_selection import GroupShuffleSplit

# =========================
# CẤU HÌNH
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "PKLot"
OUTPUT_DIR = BASE_DIR / "dataset" / "processed"

MAX_IMAGES = 600
IMAGE_SIZE = (128, 128)
RANDOM_STATE = 42

random.seed(RANDOM_STATE)

# =========================
# TÌM ẢNH VÀ XML
# =========================

image_files = list(DATA_DIR.rglob("*.jpg"))

valid_pairs = []

for image_path in image_files:
    xml_path = image_path.with_suffix(".xml")

    if xml_path.exists():
        valid_pairs.append((image_path, xml_path))

print("Tổng số cặp ảnh/XML:", len(valid_pairs))

if not valid_pairs:
    raise RuntimeError("Không tìm thấy cặp ảnh và XML!")

# Lấy ngẫu nhiên tối đa 600 ảnh
if len(valid_pairs) > MAX_IMAGES:
    valid_pairs = random.sample(valid_pairs, MAX_IMAGES)

# =========================
# CHIA TRAIN / VAL / TEST
# =========================

groups = []

for image_path, xml_path in valid_pairs:
    relative = image_path.relative_to(DATA_DIR)

    # Ví dụ: UFPR05/Sunny/2013-02-22
    group = "/".join(relative.parts[:3])
    groups.append(group)

indices = np.arange(len(valid_pairs))

split1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=RANDOM_STATE
)

train_idx, temp_idx = next(
    split1.split(indices, groups=groups)
)

temp_groups = [groups[i] for i in temp_idx]

split2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=RANDOM_STATE
)

val_relative_idx, test_relative_idx = next(
    split2.split(temp_idx, groups=temp_groups)
)

val_idx = temp_idx[val_relative_idx]
test_idx = temp_idx[test_relative_idx]

splits = {
    "train": train_idx,
    "val": val_idx,
    "test": test_idx
}

# =========================
# TẠO THƯ MỤC ĐẦU RA
# =========================

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

for split in splits:
    for label in ["Empty", "Occupied"]:
        (OUTPUT_DIR / split / label).mkdir(
            parents=True,
            exist_ok=True
        )

# =========================
# CẮT TỪNG Ô ĐỖ XE
# =========================

def crop_parking_space(image, space):
    contour = space.find("contour")

    if contour is None:
        return None

    points = []

    for point in contour.findall("point"):
        x = int(float(point.get("x")))
        y = int(float(point.get("y")))
        points.append([x, y])

    if len(points) < 3:
        return None

    points = np.array(points, dtype=np.int32)

    height, width = image.shape[:2]

    # Giới hạn tọa độ trong ảnh
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)

    x, y, w, h = cv2.boundingRect(points)

    if w <= 0 or h <= 0:
        return None

    # Cắt vùng chứa ô đỗ xe
    crop = image[y:y+h, x:x+w].copy()

    # Tạo mask theo đa giác
    local_points = points - np.array([x, y])

    mask = np.zeros((h, w), dtype=np.uint8)

    cv2.fillPoly(mask, [local_points], 255)

    # Loại bỏ vùng nằm ngoài ô đỗ xe
    crop = cv2.bitwise_and(crop, crop, mask=mask)

    # Resize ảnh
    crop = cv2.resize(crop, IMAGE_SIZE)

    return crop


# =========================
# XỬ LÝ DỮ LIỆU
# =========================

counts = {
    split: {"Empty": 0, "Occupied": 0}
    for split in splits
}

for split, selected_indices in splits.items():

    print(f"\nĐang xử lý tập {split}...")

    for index in selected_indices:

        image_path, xml_path = valid_pairs[index]

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            print("XML lỗi:", xml_path)
            continue

        for space in root.findall("space"):

            occupied = space.get("occupied")

            if occupied == "0":
                label = "Empty"
            elif occupied == "1":
                label = "Occupied"
            else:
                continue

            crop = crop_parking_space(image, space)

            if crop is None:
                continue

            space_id = space.get("id", "unknown")

            filename = (
                f"{index}_{space_id}_{label}.jpg"
            )

            save_path = (
                OUTPUT_DIR / split / label / filename
            )

            cv2.imwrite(str(save_path), crop)

            counts[split][label] += 1

# =========================
# THỐNG KÊ
# =========================

print("\n========== KẾT QUẢ ==========")

for split in counts:
    empty = counts[split]["Empty"]
    occupied = counts[split]["Occupied"]

    print(f"\n{split.upper()}")
    print("Empty:", empty)
    print("Occupied:", occupied)
    print("Tổng:", empty + occupied)

print("\nĐã lưu dữ liệu tại:", OUTPUT_DIR)
