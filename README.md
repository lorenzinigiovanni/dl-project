# Acknowledgment

This is a project for the course: **Deep Learning** of the University of Trento.

Professors:
- Elisa Ricci
- Willi Menapace

Students:
- Giovanni Lorenzini    223715
- Simone Luchetta       223716 
- Diego Planchenstainer 223728

# Necessary packages

Using pip install the following packages.

```sh
$ pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

```sh
$ pip3 install pandas
```

```sh
$ pip3 install tensorboardX
```

```sh
pip3 install numpy
```

# Dataset

The dataset needs to be extracted in a folder called dataset, mantaining the internal structure unchanged.

# Code execution

Before execution is necessary to set to true or false the variable `is_training` in the `main.py`.
If setted to true the network will train, this takes some time.
If setted to false, and providing a network model (`network.pt`) is possible to go directly to the test mode.
It will compute the mAP and the two required files (`classification_test.csv` and `reid_test.txt`).
Then it's possible to execute the code simply by calling python and the path of the main file.

```sh
$ python3 main.py
```
