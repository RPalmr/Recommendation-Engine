# TensorFlow Recommendation Models

This repository is dedicated to the implementation of recommendation engine models in TensorFlow. The models are built using the Estimator API with feature columns.

## Models

Explore the following recommendation models:

- Linear classifier: [linear.py](trainers/linear.py)
- DNN classifier: [deep.py](trainers/deep.py)
- Linear & DNN classifier: [linear_deep.py](trainers/linear_deep.py)
- DeepFM: [deep_fm.py](trainers/deep_fm.py)

### DeepFM

#### Model Parameters

Configure the DeepFM model with the following parameters:

- `categorical_columns`: Input for categorical feature columns
- `numeric_columns`: Input for numeric feature columns
- `use_linear`: Flag to include the linear structure of the model (default: `True`)
- `use_mf`: Flag to include the factorization machine structure of the model (default: `True`)
- `use_dnn`: Flag to include the deep structure of the model (default: `True`)
- `embedding_size`: Embedding size of latent factors (default: `4`)
- `hidden_units`: Layer sizes of hidden units of the deep structure (default: `[16, 16]`)
- `activation_fn`: Activation function of the deep structure (default: `tf.nn.relu`)
- `dropout`: Dropout rate of the deep structure (default: `0`)
- `optimizer`: Learning optimizer (default: `"Adam"`)
- `learning_rate`: Learning rate (default: `0.001`)

## Setup

Get started with the following setup:

```bash
# Clone the repo
git clone git@github.com:yxtay/recommender-tensorflow.git && cd recommender-tensorflow
```

# Create a conda environment
conda env create -f=environment.yml

# Activate the environment
source activate dl

## Download & Process Data

For demonstration purposes, the MovieLens 100K Dataset is used. Download, process, and enrich the data using the provided script:

```bash
# Download and process MovieLens 100K dataset
python -m src.data.ml_100k local
```

Usage

```bash
python -m src.data.ml_100k local -h
```

## Train & Evaluate DeepFM

Usage

```bash
python -m trainers.deep_fm -h
```

## Tensorboard

Inspect model training metrics using Tensorboard:

```bash
tensorboard --logdir checkpoints/
```

## Other Models Available

Explore other models:

```bash
# Linear model
python -m trainers.linear

# Deep model
python -m trainers.deep

# Wide & deep model
python -m trainers.linear_deep
```

## Distributed

For distributed model training and evaluation, refer to distributed.

## References

- Harper F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems, 5(4), Article 19, 19 pages. DOI=http://dx.doi.org/10.1145/2827872.
- Cheng, H. T., Koc, L., Harmsen, J., Shaked, T., Chandra, T., Aradhye, H., ... Shah, H. (2016). Wide & Deep Learning for Recommender Systems. arXiv:1606.07792 [cs.LG].
- Guo, H., Tang, R., Ye, Y., Li, Z., & He, X. (2017). DeepFM: A Factorization-Machine based Neural Network for CTR Prediction. arXiv:1703.04247 [cs.IR].