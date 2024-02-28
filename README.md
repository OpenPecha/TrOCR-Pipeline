This is the TrOCR model training and inference on handwritten tibetan text

## Installation

1. install Poetry for python package manager
2. install Python 3.10.0
3. After adding poetry to your PATH (in windows) or ~/.bashrc (in ubuntu), use Python 3.10.0 as your environment
by using `poetry env use path_to_python`. in Windows it would be `poetry env use C:\Users\<username>\AppData\Programs\Python\Python310\python.exe`
4. Open your terminal in the root of this repository
5. install the packages by `poetry install`

Use the Fine_Tune.ipynb as the training notebook
Use the Inference.ipynb as the inference notebook

inside the tibetan-dataset/ folder have two necessary things
labels.csv => which is a two column csv file that maps image name to it's label (text)
train/ folder which contains all images for the training

Make sure that in Fine_Tune.ipynb, you are using the correct names for the labels.csv file

