# Sort-of-CLEVR
Tensorflow implemetation for Sort of CLEVR dataset.

## Description
This project includes a [Tensorflow](https://www.tensorflow.org/) implementation of **Relation Networks** and a dataset generator which generates a synthetic VQA dataset named **Sort-of-CLEVR** proposed in the paper [A Simple Neural Network Module for Relational Reasoning](https://arxiv.org/abs/1706.01427).

### Relation Networks

Relational reasoning is an essential component of intelligent systems. To this end, Relation Networks (RNs) are proposed to solve problems hinging on inherently relational concepts. To be more specific, RN is a composite function:

![](figure/rn_eq.png)

<p align="center">
    <img src="figure/rn_eq.png" height="72" />
</p>

where *o* represents inidividual object while *f* and *g* are functions dealing with relational reasoning which are implemented as MLPs. Note that objects mentioned here are not necessary to be real objects; instead, they could consist of the background, particular physical objects, textures, conjunctions of physical objects, etc. In the implementation, objects are defined by convoluted features. The model architecture proposed to solve Visual Question Answering (VQA) problems is as follows.

<p align="center">
    <img src="figure/RN.png" height="350" />
</p>

In addition to the RN model, **a baseline model** which consists of convolutional layers followed by MLPs is also provided in this implementation.

We have a list of certain questions for images stored in the questions.json file. The questions are a mix of relational and non-relational questions. For each image we can randomly choose a numner of questions. In this repository, there were three approaches followed as follows:
* **Approach 1**: Generate images and per image choose a specific number of questions which would be randomly chosen. This generates the file *_questions.txt in the data directory for both train and test dataset. The questions are then tokenized and converted into vectors. For any word occuring for the first time it is written to the vocab.json with its index number. This way we give continous integer values to the word tokens in the questions.
* **Approach 2**: In this approach, the questions are converted to vectors using [Spacy](https://spacy.io/).
* **Approach 3**: Rather than writing the whole questions in the *_questions.txt file, we just write the indexes of the questions from the questions.json file. Later, these questions are encoded using one hot encoding. So rather than vector of integers, the input is one hot encoded vector.

Each image consists of 6 objects of 2 different shapes and unique colors. All the images are generated and saved as JPEG files within data directory.

## Prerequisites

- Python 2.7 or Python 3.3+
- [Tensorflow 1.0.0](https://github.com/tensorflow/tensorflow/tree/r1.0)
- [NumPy](http://www.numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [h5py](http://docs.h5py.org/en/latest/)
- [progressbar](http://progressbar-2.readthedocs.io/en/latest/index.html)
- [colorlog](https://github.com/borntyping/python-colorlog)
- [spacy](https://spacy.io/)
- [sklearn](https://scikit-learn.org/stable/)

## Usage

### Datasets

Generate a default Sort-of-CLEVR dataset:

```bash
$ python generator.py
```

Or generate your own Sort-of-CLEVR dataset by specifying args:

```bash
$ python generator.py --train_size 12345 --img_size 256
```

\*This code is still being developed and subject to change. The README.md will further be edited to include few more details.
