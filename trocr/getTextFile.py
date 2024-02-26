import os
from pathlib import Path

path = ".\\tibetan-dataset\\train\\"

files = os.listdir(path)

jpg_files = [f for f in files if f.endswith('.jpg')]
x_file_content = ''
count = 0
with open("tibetan-dataset/labels.csv", "w") as file:

        for jpg_file in jpg_files:
            try:
                for i in range(1, 11):
                    file_name = f"C:\\Users\\301212298\\Downloads\\annotated or proofread text files\\batch{i}\\batch{i}\\{jpg_file[:-4]}.txt"
                    print(file_name)
                    try:
                        content = Path(file_name).read_text(encoding='utf-8')
                        break
                    except FileNotFoundError:
                        continue

                content = Path(file_name).read_text(encoding='utf-8')
                x_file_content += jpg_file + ',"' + content + "\"\n"

                count += 1

                print('finished {}'.format(jpg_file))
            except FileNotFoundError:
                # red color print error
                print("\033[91m {}\033[00m" .format('failed {}'.format(jpg_file)))

                continue

x_file = Path('tibetan-dataset/labels.csv').write_text(x_file_content, encoding='utf-8')

