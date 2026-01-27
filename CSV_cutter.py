import csv

INPUT = "IMDB Dataset.csv"
OUTPUT = "IMDB_Dataset_5000.csv"
MAX_ROWS = 5000

with open(INPUT, newline='', encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)

    rows = []
    for i, row in enumerate(reader):
        if i >= MAX_ROWS:
            break
        rows.append(row)

with open(OUTPUT, "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)
