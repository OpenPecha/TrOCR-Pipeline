from pathlib import Path
import subprocess
import csv


def write_csv(csv_data, csv_path):
    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerows(csv_data)
 

def create_csv_and_folder(image_dir, transcription_dir, train_list, eval_list, test_list):
    train_data = []
    for image_name in train_list:
        source_path = image_dir /f"{image_name}.jpg"
        target_path = Path(f"./trocr/tibetan-dataset/train/{image_name}.jpg")
        subprocess.run(["cp", str(source_path), str(target_path)])
        transcript = (transcription_dir / f"{image_name}.txt").read_text(encoding='utf-8')
        train_data.append([f"{image_name}.jpg", transcript])
    write_csv(train_data, Path("./trocr/tibetan-dataset/train.csv"))

    eval_data = []
    for image_name in eval_list:
        source_path = image_dir /f"{image_name}.jpg"
        target_path = Path(f"./trocr/tibetan-dataset/train/{image_name}.jpg")
        subprocess.run(["cp", str(source_path), str(target_path)])
        transcript = (transcription_dir / f"{image_name}.txt").read_text(encoding='utf-8')
        eval_data.append([f"{image_name}.jpg", transcript])
    write_csv(eval_data, Path("./trocr/tibetan-dataset/eval.csv"))

    test_data = []
    for image_name in test_list:
        source_path = image_dir /f"{image_name}.jpg"
        target_path = Path(f"./trocr/tibetan-dataset/test/{image_name}.jpg")
        subprocess.run(["cp", str(source_path), str(target_path)])
        transcript = (transcription_dir / f"{image_name}.txt").read_text(encoding='utf-8')
        test_data.append([f"{image_name}.jpg", transcript])
    write_csv(test_data, Path("./trocr/tibetan-dataset/test.csv"))
    

def main():
    lhasa_kanjur_dir = Path(f"./trocr/tibetan-dataset/LhasaKanjur")
    image_dir = lhasa_kanjur_dir / "lines"
    transcription_dir = lhasa_kanjur_dir /"transcriptions"
    test_list = list(Path(f"./trocr/tibetan-dataset/test_imgs.txt").read_text(encoding='utf-8').splitlines())
    train_list = list(Path(f"./trocr/tibetan-dataset/train_imgs.txt").read_text(encoding='utf-8').splitlines())
    eval_list = list(Path(f"./trocr/tibetan-dataset/val_imgs.txt").read_text(encoding='utf-8').splitlines())
    create_csv_and_folder(image_dir, transcription_dir, train_list, eval_list, test_list)


if __name__ == "__main__":
    main()
