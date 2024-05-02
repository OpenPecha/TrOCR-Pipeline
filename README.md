This is the TrOCR model training and inference on handwritten tibetan text

## Versions
There are multiple versions of the TrOCR pipeline. 
One that might be the most important one is the accelerate version which can be found in the huggingface-trainer branch.
Using accelerate, you can run the training in multiple scenarios. Whether you are training on a single GPU system or a multi-GPU system, the accelerate branch can optimize the training based on the system. https://youtu.be/t8Krzu-nSeY ( a good video on accelerate )  

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

## Inference

1. If you have a model, make sure it's unzipped and place it in the trocr/ folder.
2. Then go to Inference.ipynb and make sure the folder name for the model matches your model's folder
3. run the inference, use an image path to get the text

## Known issues
When using poetry to download pytorch, there is a massive download.
One way to get around it is to remove pytorch from poetry management and run pip from poetry instead of using `poetry add`
So something like removing anything related to pytorch from pyproject.toml and then running `poetry run pip install torch`
Although, maybe by the time you see this, the issue is resolved: https://github.com/python-poetry/poetry/issues/6409
