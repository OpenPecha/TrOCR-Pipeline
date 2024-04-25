from pathlib import Path
import pyewts
import csv
from trocr.split_data import write_csv

converter = pyewts.pyewts()

def read_csv(csv_path):
    with open(csv_path, 'r') as csvfile:
        csv_data = list(csv.reader(csvfile))
    return csv_data


def convert_to_unicode(csv_data, output_path):
    data = []
    for row in csv_data:
        id = row[0]
        transcript = row[1]
        text = converter.toUnicode(transcript)
        data.append([id, text])
    write_csv(data, output_path)


def main():
    train_csv = read_csv(Path(f"./trocr/tibetan-dataset/train.csv"))
    eval_csv = read_csv(Path(f"./trocr/tibetan-dataset/eval.csv"))
    test_csv = read_csv(Path(f"./trocr/tibetan-dataset/test.csv"))
    convert_to_unicode(train_csv, Path(f"./trocr/tibetan-dataset/train_uni.csv"))
    convert_to_unicode(eval_csv, Path(f"./trocr/tibetan-dataset/eval_uni.csv"))
    convert_to_unicode(test_csv,Path(f"./trocr/tibetan-dataset/test_uni.csv"))

if __name__ == "__main__":
    main()