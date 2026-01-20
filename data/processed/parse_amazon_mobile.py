import csv
import json
import os

INPUT_FILE = "data/raw/amazon_mobile.csv"
OUTPUT_FILE = "data/processed/products.jsonl"

os.makedirs("data/processed", exist_ok=True)

count = 0

with open(INPUT_FILE, newline="", encoding="latin-1") as csvfile, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as out:

    reader = csv.DictReader(csvfile)

    for row in reader:
        text = row.get("Product Description ")
        if not text:
            continue

        record = {
            "id": count,
            "text": text.strip(),
            "price": row.get("Price(Dollar)"),
            "num_reviews": row.get("Number of reviews"),
            "shipment": row.get("Shipment"),
            "delivery_date": row.get("Delivery Date")
        }

        out.write(json.dumps(record) + "\n")
        count += 1

print(f"Parsed {count} products")
