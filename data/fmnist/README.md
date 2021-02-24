The dataset used in our experiments:

* first dowanload train and test sets [here](https://drive.google.com/file/d/1WyU9IYHUeLPElFuzPc8--qB7dRPuNZBR/view?usp=sharing) 
* unzip it, then put the `train` and `test` folders under `data/fmnist/data/`


-----
How the data were generated:


First download raw data:

```
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
```
Then we partition the dataset across 500 devices (need to double-check the data dir path in the script):

```
python create_dataset.py
```
