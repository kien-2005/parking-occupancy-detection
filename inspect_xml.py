from pathlib import Path
import xml.etree.ElementTree as ET

DATA_DIR = Path("PKLot")

xml_files = list(DATA_DIR.rglob("*.xml"))

if len(xml_files) == 0:
    print("Không tìm thấy file XML!")
else:
    xml_path = xml_files[0]

    print("File XML:", xml_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    print("\nTên thẻ gốc:", root.tag)
    print("\nNội dung XML:")

    print(
        ET.tostring(
            root,
            encoding="unicode"
        )[:5000]
    )