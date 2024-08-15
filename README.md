# Ensemble learning on EHR dataset

Explore the effectiveness of using ensemble learning prediction models on the Electronic Health Records (EHR) dataset.

On two real-world COVID-19 EHR datasets (TJH and CDSL datasets), ensemble learning experiments were conducted based on death risk prediction (binary classification) tasks to determine whether the performance of deep learning models reached the upper limit on this dataset. This experiment was completed based on the [PyEHR](https://github.com/yhzhu99/pyehr) framework proposed in the paper [A Comprehensive Benchmark for COVID-19 Predictive Modeling Using Electronic Health Records in Intensive Care](https://doi.org/10.48550/arxiv.2209.07805). 

## 🎯 Prediction Task

- [x] outcome prediction

## 🚀 Model Zoo

The repository contains a variety of models from traditional machine learning, basic deep learning, and advanced deep learning models tailored for EHR data:

### Machine Learning Models

- [x] Random forest (RF)
- [x] Decision tree (DT)
- [x] Gradient Boosting Decision Tree (GBDT)
- [x] XGBoost
- [x] CatBoost

### Deep Learning Models

- [x] Multi-layer perceptron (MLP)
- [x] Recurrent neural network (RNN)
- [x] Long-short term memory network (LSTM)
- [x] Gated recurrent units (GRU)
- [x] Temporal convolutional networks
- [x] Transformer

### EHR Predictive Models

- [x] RETAIN
- [x] StageNet
- [x] Dr. Agent
- [x] AdaCare
- [x] ConCare
- [x] GRASP

The best searched hyperparameters for each model are meticulously preserved in the configs/ folder (`dl.py` and `ml.py`).

## ⚙️ Requirements

To get started with the repository, ensure your environment meets the following requirements:

- Python 3.8+
- PyTorch 2.0 (use Lightning AI)
- See `requirements.txt` for additional dependencies.

## 📈 Usage

To start with the data pre-precessing steps, follow the instructions:

1. Download TJH dataset from paper [An interpretable mortality prediction model for COVID-19 patients](https://www.nature.com/articles/s42256-020-0180-7), and you need to apply for the CDSL dataset if necessary. [Covid Data Save Lives Dataset](https://www.hmhospitales.com/prensa/notas-de-prensa/comunicado-covid-data-save-lives)
3. Run the pre-processing scripts `preprocess_{dataset}.ipynb` in `datasets/` folder.
4. Then you will have the 10-fold processed datasets in the required data format.

To start with the training or testing, use the following commands:

```bash
# Hyperparameter tuning
python dl_tune.py # for deep learning models
python ml_tune.py # for machine learning models

# Model training
python train.py
python train-dl.py
python train-ml.py

# Model testing
python test.py
python test-dl.py
python test-ml.py
```
