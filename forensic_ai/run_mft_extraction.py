import json
from pipeline.tsk_parser import parse_mft

records = parse_mft("data/raw/disk_image.dd")

with open("data/processed/mft.json", "w") as f:
    json.dump(records, f)

print("MFT extracted")