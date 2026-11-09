from pathlib import Path

DATA_DIR = Path("PKLot")

images = list(DATA_DIR.rglob("*.jpg"))
xml_files = list(DATA_DIR.rglob("*.xml"))

print("========== DATASET PKLot ==========")
print("Tổng số ảnh:", len(images))
print("Tổng số file XML:", len(xml_files))

print("\n========== DANH SÁCH ẢNH ==========")
for image in images[:5]:
    print(image)

print("\n========== DANH SÁCH XML ==========")
for xml in xml_files[:5]:
    print(xml)