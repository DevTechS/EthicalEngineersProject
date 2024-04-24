import csv

def load_csv_as_2d_array(csv_file_path):
    data = []
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)
    return data


csv_file_path = "output_custom.csv"
csv_data = load_csv_as_2d_array(csv_file_path)

total_count = 0
counts = {
    'White': 0,
    'East Asian': 0,
    'Middle Eastern': 0,
    'Black': 0,
    'Indian': 0,
    'Latino_Hispanic': 0,
    'Southeast Asian': 0
}

total_passes = 0
passes = {
    'White': 0,
    'East Asian': 0,
    'Middle Eastern': 0,
    'Black': 0,
    'Indian': 0,
    'Latino_Hispanic': 0,
    'Southeast Asian': 0
}

for row in csv_data[1:]:
    # print(row)
    counts[row[3]] += 1
    total_count += 1
    if (row[4] != '0'):
        passes[row[3]] += 1
        total_passes += 1

print(counts)
print(passes)

for key, value in counts.items():
    print(f"{key}: {round(100*passes[key]/value,1)}%")
print(f"Overall: {round(100*total_passes/total_count,1)}%")