import csv
import os
import glob

# for filename in glob.glob(os.path.join("sivan", "*.tsv")):
#     print(f"Processing file: {filename}")
#     i=0
#     arr =[]
#     male_count = 0
#     female_count = 0
#     with open(filename, 'r', encoding='utf-8') as f:
#         reader = csv.DictReader(f, delimiter='\t')
#         count = 0
#         for row in reader:
#             if row['gender'] == 'male':
#                 print(row)
#                 arr.append(row)
#                 count += 1
#             if count == 30000:
#                 break
#
#     with open("name_audio", "w", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow(["path"])
#         for row in arr:
#             writer.writerow([row["path"]])

def gender(str):
    results = []
    for filename in glob.glob(os.path.join("stav", "*.tsv")):
        print(f"Processing file: {filename}")
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            count = 0
            for row in reader:
                if row['gender'] == str:
                    results.append(row)
                    count += 1
                if count == 24000:
                    break
    return results

if __name__ == '__main__':
    male_results = gender("female")
    with open("name_audio", "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path"])
        for row in male_results:
            writer.writerow([row["path"]])


    # print results
    print(len(male_results))
    # print("Results for female:")
    # for result in female_results:
    #
    #     print(result)