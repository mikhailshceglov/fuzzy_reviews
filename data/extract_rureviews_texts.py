#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
from pathlib import Path

# Путь к CSV относительно скрипта
BASE_DIR = Path(__file__).parent
INPUT_CSV = BASE_DIR / "rureviews" / "women-clothing-accessories.3-class.balanced.csv"
OUTPUT_TXT = BASE_DIR / "rureviews_reviews.txt"

def main():
    with INPUT_CSV.open("r", encoding="utf-8") as src, \
         OUTPUT_TXT.open("w", encoding="utf-8") as dst:
        reader = csv.reader(src, delimiter="\t")
        for row in reader:
            if not row:
                continue
            text = row[0].replace("\n", " ").strip()
            if text:
                dst.write(text + "\n")

if __name__ == "__main__":
    main()
