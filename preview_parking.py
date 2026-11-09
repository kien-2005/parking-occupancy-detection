
from pathlib import Path
import cv2
import xml.etree.ElementTree as ET

# Thư mục chứa dataset, nằm cạnh file Python này
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "PKLot"

# Tìm một ảnh JPG
image_files = sorted(DATA_DIR.rglob("*.jpg"))

if not image_files:
    raise FileNotFoundError(f"Không tìm thấy ảnh trong {DATA_DIR}")

image_path = image_files[0]
xml_path = image_path.with_suffix(".xml")

if not xml_path.exists():
    raise FileNotFoundError(f"Không tìm thấy XML tương ứng: {xml_path}")

# Đọc ảnh
image = cv2.imread(str(image_path))

if image is None:
    raise ValueError("Không đọc được ảnh!")

# Đọc XML
tree = ET.parse(xml_path)
root = tree.getroot()

empty_count = 0
occupied_count = 0

for space in root.findall("space"):
    space_id = space.get("id")
    occupied = space.get("occupied")

    contour = space.find("contour")

    if contour is None:
        continue

    points = []

    for point in contour.findall("point"):
        x = int(float(point.get("x")))
        y = int(float(point.get("y")))
        points.append([x, y])

    if len(points) < 3:
        continue

    points = __import__("numpy").array(points, dtype="int32")

    if occupied == "0":
        color = (0, 255, 0)  # Xanh: Empty
        label = "Empty"
        empty_count += 1
    else:
        color = (0, 0, 255)  # Đỏ: Occupied
        label = "Occupied"
        occupied_count += 1

    cv2.polylines(
        image,
        [points],
        True,
        color,
        2
    )

    center = points.mean(axis=0).astype(int)

    cv2.putText(
        image,
        f"{space_id}: {label}",
        tuple(center),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        color,
        1
    )

# Thống kê
print("Ảnh:", image_path)
print("Empty:", empty_count)
print("Occupied:", occupied_count)
print("Tổng số chỗ:", empty_count + occupied_count)

# Lưu ảnh kết quả
output_dir = BASE_DIR / "outputs"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "preview.jpg"

cv2.imwrite(str(output_path), image)

print("Đã lưu ảnh:", output_path)
