import os
import xml.etree.ElementTree as ET

folder_path = "./data/"

for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if not filename.endswith(".xml"):
            continue
        file_path = os.path.join(root, filename)

        tree = ET.parse(file_path)
        r = tree.getroot()
        for obj in r.findall("object"):
            if obj.find("name").text != "mast":
                print(f"Found non 'mast' object in {file_path}")
                break
