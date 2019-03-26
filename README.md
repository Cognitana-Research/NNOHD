[![License AGPL-3.0](https://img.shields.io/badge/License-AGPL--3-brightgreen.svg)](https://github.com/Cognitana-Research/NNOHD/blob/master/LICENSE)

# Unsupervised Artificial Neural Networks for Outlier Detection in High-Dimensional Data
# Implementation Release and Additional Material


Welcome to the additional material for the paper *Unsupervised Artificial Neural Networks for Outlier Detection in High-Dimensional Data* by Daniel Popovic, Edouard Fouché and Klemens Böhm, Karlsruhe Institute of Technology (KIT). This repository contains implementation releases of the three approaches *Autoencoder*, *Restricted Boltzmann Machine*, and *Self-Organising Map* for outlier detection with details to reproduce the experiments in the paper. In addition, it contains experimental results on further data sets and completes the tables, which are only partially shown in the paper for reasons of space.

This repository is released under the AGPLv3 license. Please see the [LICENSE](https://github.com/Cognitana-Research/NNOHD/blob/master/LICENSE) file for details.

## Quick Start

### Prerequisites

Download this repository to your local machine. It is recommended to use virtual environment to manage Python dependencies.
```bash
$ pip3 install -r requirements.txt
```
To re-enact the experiments, download the data sets as ZIP files at [Datasets Download](https://www.dropbox.com/sh/8ewsyor0bjrllzc/AADTWWD-FBgxKaNXR-sZfd0ua?dl=0) and unpack them.

### Run the Outlier Detector

The outlier detector runs in four different modes, using either Autoencoder, Restricted Boltzmann Machine, or Self-Organising Map alone, or run all three methods together. The detector supports two globally used optional parameters:
* `-w`: Writes the outlier detection results into CSV files. For each outlier detection method, a separate file is saved. The files contain an incremented data object ID, the real outlier label provided in the data set, and the method's respective outlier score.
* `-d`: The column delimiter for data sets in CSV format. The default delimiter is `;`.

#### Autoencoder

To run the Autoencoder alone, use the command line positional parameter *ae* to specify the method, and as the second positional parameter, the path and file name of the data set.
```bash
$ python detector.py ae mypath/mydataset.arff
```
For the Autoencoder, the following optional parameters exist:
* `-a`: The number of training epochs. Default is 20.
* `-e`: The encoding factor. Default is 0.8.

#### Restricted Boltzmann Machine

To run the Restricted Boltzmann Machine alone, use the positional parameter *rbm* to specify the method, and the data set path and file name as a second positional parameter.

For the Restricted Boltzmann Machine, the following optional parameters exist:
* `-r`: The number of training epochs. Default is 100.
* `-n`: The proportion of hidden neurons. Default is 0.8.

#### Self-Organising Map

To run the Self-Organising Map alone, use the positional parameter *som* to specify the method, and the data set path and file name as a second positional parameter.
```bash
$ python detector.py som mypath/mydataset.arff
```
For the Self-Organising Map, the following optional parameters exist:
* `-s`: The number of training epochs. Default is 100.
* `-g`: The grid size or the number of neuron rows and columns in the quadratic SOM grid. Default is 2.

#### All Methods

To run all three methods at once, use the positional parameter *all* to specify all three methods, and the data set path and file name as the second positional parameter.
```bash
$ python detector.py all mypath/mydataset.arff
```
Each optional parameter that exists for one of the three methods (see above) is also valid to use with all three methods at once. If an optional parameter is not specified, the default value is used.

### Data Set Structure

The outlier detector supports three file types for data sets: ARFF files (`.arff`), CSV files (`.csv`), and *MATLAB* files < version 7.3 (`.mat`).

#### ARFF Files

#### CSV Files

#### MATLAB Files

## Implementation Release

### Project Structure

#### Root Folder

Folders:

* `nnohd`: The implementation of the unsupervised neural networks for outlier detection.
* `images`: Contains images for this README.
* `datasets`: The data sets used for evaluation.

Files:

* `README.md`
* `LICENSE`: The license file. This repository is published under GNU AFFERO GENERAL PUBLIC LICENSE Version 3.
* `requirements.txt`: Refers to `app/requirements.txt`.

#### Application Structure

```
nnohd/
├── requirements.txt
├── ae.py
├── rbm.py
├── som.py
├── utils
│   └── preprocessing.py
```

### Dependencies

* [**Python**](https://www.python.org/) 3.6+
* [**Keras**](https://pypi.org/project/Keras/2.2.4/) 2.2.4
* [**Matplotlib**](https://pypi.org/project/matplotlib/2.2.2/) 2.2.2
* [**NumPy**](https://pypi.org/project/numpy/1.16.2/) 1.16.2
* [**Pandas**](https://pandas.pydata.org/) 0.24.2
* [**Scikit-learn**](https://pypi.org/project/scikit-learn/0.20.3/) 0.20.3
* [**SciPy**](https://pypi.org/project/scipy/1.2.1/) 1.2.1
* [**Somoclu**](https://pypi.org/project/somoclu/1.7.5/) 1.7.5
* [**Tensorflow**](https://pypi.org/project/tensorflow/1.13.1/) 1.13.1

## Additional Material
This section contains the complete data for the sections of the paper that are only partially filled due to space limitations.

### Evaluated Data Sets
In this section we list the complete set of evaluated data sets. The evaluated high-dimensional data sets can be found as ZIP files at [Datasets Download](https://www.dropbox.com/sh/8ewsyor0bjrllzc/AADTWWD-FBgxKaNXR-sZfd0ua?dl=0).

<details><summary>Show Table</summary>
<p>
  
Data Set | Dimensions | Data Objects | Outliers | Outlier Ratio
--- | --- | --- | --- | ---
Arrhythmia-2 | 259 | 248 | 4 | 1.61%
Arrhythmia-5 | 259 | 256 | 12 | 4.69%
Arrhythmia-10 | 259 | 271 | 27 | 9.96%
Arrhythmia-20 | 259 | 305 | 61 | 20.00%
Arrhythmia-46 | 259 | 450 | 206 | 45.78%
Cardio | 21 | 1,831 | 176 | 9.61%
Ecoli | 7 | 336 | 9 | 2.68%
InternetAds-2 | 1,555 | 1,630 | 32 | 1.96%
InternetAds-5 | 1,555 | 1,682 | 84 | 4.99%
InternetAds-10 | 1,555 | 1,775 | 177 | 9.97%
InternetAds-19 | 1,555 | 1,966 | 368 | 18.72%
ISOLET-2 | 617 | 2,449 | 50 | 2.00%
ISOLET-4 | 617 | 2,499 | 100 | 4.00%
ISOLET-8 | 617 | 2,599 | 200 | 7.70%
ISOLET-14 | 617 | 2,799 | 400 | 14.29%
ISOLET-20 | 617 | 2,999 | 600 | 20.00%
Lympho | 18 | 148 | 6 | 4.05%
MNIST | 100 | 7,603 | 700 | 9.21%
Musk | 166 | 3,062 | 97 | 3.12%
Optdigits | 64 | 5,216 | 150 | 2.88%
P53 | 5,408 | 16,592 | 143 | 0.86%
Pendigits | 16 | 6,870 | 156 | 2.27%
Seismic | 11 | 2,584 | 170 | 6.58%
Thyroid | 6 | 3,772 | 2.4%
Waveform | 21 | 3,509 | 166 | 4.73%
Yeast | 8 | 1,364 | 65 | 4.77%

</p></details>

### Dataset Evaluation

#### ROC AUC for High Dimensional Data Sets

<details><summary>Show Table</summary>
<p>

Data Set | AE | SOM | RBM | HiCS | LOF | FastABOD | LoOP | OC-SVM | KNN
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- 
Arrhythmia-2 | **80.22** | 76.33 | 49.04 | 50.56 | 76.74 | 76.84 | 75.00 | 77.66 | 71.88
Arrhythmia-5 | 78.16 | **81.49** | 50.36 | 58.81 | 80.46 | 49.51 | 79.99 | *81.47* | 68.72
Arrhythmia-10 | *83.84* | *83.74* | 48.23 | 65.12 | *83.45* | 83.35 | **84.24** | *83.26* | 76.20
Arrhythmia-20 | 71.01 | 71.33 | 49.93 | 50.71 | 70.47 | 71.14 | *71.70* | **72.38** | 67.52
Arrhythmia-46 | **74.93** | *74.79* | 46.99 | 58.96 | *74.41* | *74.17* | *74.10* | *74.13* | 70.72
InternetAds-2 | 43.36 | 66.12 | 46.93 | **99.84** | 71.64 | 76.49 | 77.14 | 64.18 | 81.23
InternetAds-5 | 40.14 | 73.36 | 50.27 | **99.87** | 78.63 | 78.75 | 82.56 | 74.76 | 69.35
InternetAds-10 | 39.38 | 69.42 | 49.75 | **79.25** | 74.71 | 74.41 | 77.73 | 71.25 | 65.21
InternetAds-19 | 36.32 | 69.69 | 50.14 | 47.33 | **74.01** | 72.40 | 69.78 | 62.43 | 59.54
ISOLET-2 | 96.85 | *99.28* | 92.28 | 79.71 | **99.58** | 93.09 | 98.23 | 92.05 | 94.66
ISOLET-4 | 94.72 | *98.49* | 87.24 | 79.79 | **99.34** | 86.44 | 95.08 | 89.75 | 83.89
ISOLET-8 | 95.16 | **98.29** | 87.19 | 78.14 | 83.81 | 75.37 | 68.71 | 89.19 | 82.90
ISOLET-14 | 92.67 | **94.96** | 85.50 | 78.16 | 65.61 | 75.60 | 62.16 | 81.61 | 80.00
ISOLET-20 | **90.16** | 87.13 | 87.19 | 77.82 | 60.04 | 73.28 | 55.00 | 77.06 | 77.40
MNIST | **82.06** | 81.07 | 49.87 | 51.74 | 80.34 | 54.35 | 71.66 | 76.46 | 72.74
Musk | **100.00** | **100.00** | 95.60 | *99.60* | 84.00 | 5.11 | 51.86 | 67.60 | 7.11
P53 | 60.63 | **67.17** | 64.76 | 62.09 | 61.99 | 62.92 | 61.99 | 61.27 | 62.56

</p></details>

#### PR AUC for High Dimensional Data Sets

<details><summary>Show Table</summary>
<p>

Data Set | AE | SOM | RBM | HiCS | LOF | FastABOD | LoOP | OC-SVM | KNN
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- 
Arrhythmia-2 | **29.37** | 27.94 | 1.66 | 1.64 | 3.83 | 3.98 | 3.51 | 4.90 | 3.04
Arrhythmia-5 | 30.61 | **36.15** | 5.00 | 4.14 | 30.62 | 29.53 | 29.85 | 30.56 | 22.37
Arrhythmia-10 | **48.54** | *48.24* | 10.31 | 13.84 | 45.36 | 42.49 | 45.55 | 40.55 | 33.24
Arrhythmia-20 | **51.97** | 50.62 | 20.35 | 15.78 | 49.12 | 48.14 | 48.90 | 49.92 | 46.98
Arrhythmia-46 | **75.32** | 74.54 | 44.98 | 44.67 | 74.36 | 72.83 | 71.63 | 73.68 | 71.50
InternetAds-2 | 1.58 | 32.78 | 1.91 | **94.52** | 34.86 | 32.79 | 3.51 | 4.90 | 3.04
InternetAds-5 | 5.12 | 45.86 | 5.21 | **97.67** | 52.56 | 40.25 | 51.12 | 33.54 | 32.87
InternetAds-10 | 7.69 | 47.89 | 10.12 | 41.13 | **51.93** | 35.74 | 46.09 | 29.40 | 35.11
InternetAds-19 | 14.29 | *54.67* | 18.91 | 1.44 | **55.24** | 41.86 | 44.48 | 29.46 | 33.64
ISOLET-2 | 44.45 | 66.93 | 29.51 | 4.47 | **70.95** | 29.54 | 51.78 | 25.65 | 42.91
ISOLET-4 | 49.97 | 69.76 | 33.59 | 9.08 | **74.21** | 24.07 | 47.62 | 47.61 | 23.96
ISOLET-8 | 59.60 | **78.61** | 40.50 | 14.36 | 43.81 | 21.71 | 31.45 | 46.55 | 28.59
ISOLET-14 | 64.48 | **75.40** | 53.84 | 29.06 | 40.03 | 33.68 | 27.15 | 47.53 | 36.62
ISOLET-20 | **64.26** | 63.55 | 40.50 | 36.92 | 34.56 | 37.72 | 25.59 | 48.72 | 41.45
MNIST | 30.13 | 27.40 | 9.26 | 9.99 | **33.95** | 14.05 | 24.74 | 25.49 | 27.96
Musk | **100.00** | **100.00** | 85.58 | 97.46 | 14.68 | 1.65 | 3.71 | 4.54 | 1.91
P53 | 1.04 | *1.29* | 1.20 | 1.25 | 1.06 | 1.13 | 1.06 | 1.06 | **1.30**

</p></details>

#### Visualization

##### ROC AUC and PR AUC for Arrhythmia-2

![](https://github.com/Cognitana-Research/NNOHD/blob/master/images/ROC_PR_Arrhythmia-2.png "ROC AUC and PR AUC for Arrhythmia-2")

##### ROC AUC and PR AUC for Arrhythmia-5

![](https://github.com/Cognitana-Research/NNOHD/blob/master/images/ROC_PR_Arrhythmia-5.png "ROC AUC and PR AUC for Arrhythmia-5")

##### ROC AUC and PR AUC for Arrhythmia-10

![](https://github.com/Cognitana-Research/NNOHD/blob/master/images/ROC_PR_Arrhythmia-10.png "ROC AUC and PR AUC for Arrhythmia-10")

##### ROC AUC and PR AUC for Arrhythmia-20

![](https://github.com/Cognitana-Research/NNOHD/blob/master/images/ROC_PR_Arrhythmia-20.png "ROC AUC and PR AUC for Arrhythmia-20")

##### ROC AUC and PR AUC for Arrhythmia-46

![](https://github.com/Cognitana-Research/NNOHD/blob/master/images/ROC_PR_Arrhythmia-46.png "ROC AUC and PR AUC for Arrhythmia-46")

##### ROC AUC and PR AUC for InternetAds-2

![](https://github.com/Cognitana-Research/NNOHD/blob/master/images/ROC_PR_InternetAds-2.png "ROC AUC and PR AUC for InternetAds-2")

##### ROC AUC and PR AUC for InternetAds-5

![](https://github.com/Cognitana-Research/NNOHD/blob/master/images/ROC_PR_InternetAds-5.png "ROC AUC and PR AUC for InternetAds-5")

##### ROC AUC and PR AUC for InternetAds-10

![](https://github.com/Cognitana-Research/NNOHD/blob/master/images/ROC_PR_InternetAds-10.png "ROC AUC and PR AUC for InternetAds-10")

##### ROC AUC and PR AUC for InternetAds-19

![](https://github.com/Cognitana-Research/NNOHD/blob/master/images/ROC_PR_InternetAds-19.png "ROC AUC and PR AUC for InternetAds-19")

##### ROC AUC and PR AUC for ISOLET-2

![](https://github.com/Cognitana-Research/NNOHD/blob/master/images/ROC_PR_ISOLET-2.png "ROC AUC and PR AUC for ISOLET-2")

##### ROC AUC and PR AUC for ISOLET-4

![](https://github.com/Cognitana-Research/NNOHD/blob/master/images/ROC_PR_ISOLET-4.png "ROC AUC and PR AUC for ISOLET-4")

##### ROC AUC and PR AUC for ISOLET-8

![](https://github.com/Cognitana-Research/NNOHD/blob/master/images/ROC_PR_ISOLET-8.png "ROC AUC and PR AUC for ISOLET-8")

##### ROC AUC and PR AUC for ISOLET-14

![](https://github.com/Cognitana-Research/NNOHD/blob/master/images/ROC_PR_ISOLET-14.png "ROC AUC and PR AUC for ISOLET-14")

##### ROC AUC and PR AUC for ISOLET-20

![](https://github.com/Cognitana-Research/NNOHD/blob/master/images/ROC_PR_ISOLET-20.png "ROC AUC and PR AUC for ISOLET-20")

##### ROC AUC and PR AUC for MNIST

![](https://github.com/Cognitana-Research/NNOHD/blob/master/images/ROC_PR_MNIST.png "ROC AUC and PR AUC for MNIST")

##### ROC AUC and PR AUC for Musk

![](https://github.com/Cognitana-Research/NNOHD/blob/master/images/ROC_PR_Musk.png "ROC AUC and PR AUC for Musk")

##### ROC AUC and PR AUC for P53

![](https://github.com/Cognitana-Research/NNOHD/blob/master/images/ROC_PR_P53.png "ROC AUC and PR AUC for P53")


### Parameter Evaluation
In this section we present the complete tables of parameter evaluation values for *Autoencoder*, *Restricted Boltzmann Machine*, and *Self-Organising Map*.


#### Parameter Evaluation for Autoencoder (AE)

<details><summary>Show Table</summary>
<p>

Parameter ![](https://github.com/Cognitana-Research/NNOHD/blob/master/images/epsilon.png "Epsilon") | Epochs ![](https://github.com/Cognitana-Research/NNOHD/blob/master/images/n_e.png "n^e") | ![](https://github.com/Cognitana-Research/NNOHD/blob/master/images/D_epsilon_n_e.png "D_{Epsilon,n^e}")
---|---|---
0.4 | 10 | 0.
0.4 | 20 | 0.
0.4 | 100 | 0.
0.4 | 1,000 | 0.
0.5 | 10 | 0.0519
0.5 | 20 | *0.0405*
0.5 | 100 | 0.0527
0.5 | 1,000 | 0.0820
0.6 | 10 | *0.0478*
0.6 | 20 | *0.0411*
0.6 | 100 | *0.0476*
0.6 | 1,000 | 0.0844
0.7 | 10 | *0.0501*
0.7 | 20 | *0.0412*
0.7 | 100 | 0.0527
0.7 | 1,000 | 0.0788
0.8 | 10 | *0.0409*
0.8 | 20 | **0.0403**
0.8 | 100 | 0.0512
0.8 | 1,000 | 0.0827
0.9 | 10 | *0.0424*
0.9 | 20 | *0.0412*
0.9 | 100 | 0.0539
0.9 | 1,000 | 0.0828

</p></details>

#### Parameter Evaluation for Restricted Boltzmann Machine (RBM)

<details><summary>Show Table</summary>
<p>

Parameter ![](https://github.com/Cognitana-Research/NNOHD/blob/master/images/delta.png "Delta") | Parameter ![](https://github.com/Cognitana-Research/NNOHD/blob/master/images/gamma.png "Gamma") | Epochs ![](https://github.com/Cognitana-Research/NNOHD/blob/master/images/n_e.png "n^e") | ![](https://github.com/Cognitana-Research/NNOHD/blob/master/images/D_delta_gamma_n_e.png "D_{delta,gamma,n^e}")
---|---|---|---
0.1 | 0.1 | 10 | 0.2318
0.1 | 0.1 | 20 | 0.2067
0.1 | 0.1 | 100 | 0.1700
0.1 | 0.1 | 1,000 | 0.1507
0.1 | 0.2 | 10 | 0.1221
0.1 | 0.2 | 20 | 0.0985
0.1 | 0.2 | 100 | 0.0994
0.1 | 0.2 | 1,000 | 0.2349
0.1 | 0.3 | 10 | 0.2038
0.1 | 0.3 | 20 | 0.1925
0.1 | 0.3 | 100 | 0.1453
0.1 | 0.3 | 1,000 | 0.1192
0.1 | 0.4 | 10 | 0.0971
0.1 | 0.4 | 20 | 0.1142
0.1 | 0.4 | 100 | 0.2313
0.1 | 0.4 | 1,000 | 0.2086
0.1 | 0.5 | 10 | 0.1857
0.1 | 0.5 | 20 | 0.1641
0.1 | 0.5 | 100 | 0.1136
0.1 | 0.5 | 1,000 | 0.0903
0.1 | 0.6 | 10 | 0.0920
0.1 | 0.6 | 20 | 0.0928
0.1 | 0.6 | 100 | 0.2010
0.1 | 0.6 | 1,000 | 0.1900
0.1 | 0.7 | 10 | 0.1587
0.1 | 0.7 | 20 | 0.1375
0.1 | 0.7 | 100 | 0.1000
0.1 | 0.7 | 1,000 | 0.0740
0.1 | 0.8 | 10 | 0.0778
0.1 | 0.8 | 20 | 0.2408
0.1 | 0.8 | 100 | 0.1873
0.1 | 0.8 | 1,000 | 0.1576
0.1 | 0.9 | 10 | 0.1426
0.1 | 0.9 | 20 | 0.1121
0.1 | 0.9 | 100 | 0.0764
0.1 | 0.9 | 1,000 | 0.0756
0.2 | 0.1 | 10 | 0.2412
0.2 | 0.1 | 20 | 0.2058
0.2 | 0.1 | 100 | 0.1660
0.2 | 0.1 | 1,000 | 0.1460
0.2 | 0.2 | 10 | 0.1305
0.2 | 0.2 | 20 | 0.1202
0.2 | 0.2 | 100 | 0.0955
0.2 | 0.2 | 1,000 | 0.2266
0.2 | 0.3 | 10 | 0.1975
0.2 | 0.3 | 20 | 0.1832
0.2 | 0.3 | 100 | 0.1439
0.2 | 0.3 | 1,000 | 0.1142
0.2 | 0.4 | 10 | 0.1031
0.2 | 0.4 | 20 | 0.0913
0.2 | 0.4 | 100 | 0.2398
0.2 | 0.4 | 1,000 | 0.2001
0.2 | 0.5 | 10 | 0.1883
0.2 | 0.5 | 20 | 0.1628
0.2 | 0.5 | 100 | 0.1143
0.2 | 0.5 | 1,000 | 0.1070
0.2 | 0.6 | 10 | 0.0751
0.2 | 0.6 | 20 | 0.0881
0.2 | 0.6 | 100 | 0.2084
0.2 | 0.6 | 1,000 | 0.1852
0.2 | 0.7 | 10 | 0.1508
0.2 | 0.7 | 20 | 0.1400
0.2 | 0.7 | 100 | 0.0953
0.2 | 0.7 | 1,000 | 0.0764
0.2 | 0.8 | 10 | 0.0923
0.2 | 0.8 | 20 | 0.2278
0.2 | 0.8 | 100 | 0.1755
0.2 | 0.8 | 1,000 | 0.1612
0.2 | 0.9 | 10 | 0.1429
0.2 | 0.9 | 20 | 0.1281
0.2 | 0.9 | 100 | 0.0822
0.2 | 0.9 | 1,000 | *0.0638*
0.3 | 0.1 | 10 | 0.2381
0.3 | 0.1 | 20 | 0.2174
0.3 | 0.1 | 100 | 0.1635
0.3 | 0.1 | 1,000 | 0.1473
0.3 | 0.2 | 10 | 0.1232
0.3 | 0.2 | 20 | 0.1111
0.3 | 0.2 | 100 | 0.0986
0.3 | 0.2 | 1,000 | 0.2351
0.3 | 0.3 | 10 | 0.2138
0.3 | 0.3 | 20 | 0.1846
0.3 | 0.3 | 100 | 0.1429
0.3 | 0.3 | 1,000 | 0.1249
0.3 | 0.4 | 10 | 0.0985
0.3 | 0.4 | 20 | 0.1104
0.3 | 0.4 | 100 | 0.2335
0.3 | 0.4 | 1,000 | 0.2191
0.3 | 0.5 | 10 | 0.1916
0.3 | 0.5 | 20 | 0.1622
0.3 | 0.5 | 100 | 0.1214
0.3 | 0.5 | 1,000 | 0.0978
0.3 | 0.6 | 10 | 0.0786
0.3 | 0.6 | 20 | *0.0657*
0.3 | 0.6 | 100 | 0.2042
0.3 | 0.6 | 1,000 | 0.1806
0.3 | 0.7 | 10 | 0.1547
0.3 | 0.7 | 20 | 0.1338
0.3 | 0.7 | 100 | 0.0894
0.3 | 0.7 | 1,000 | 0.0781
0.3 | 0.8 | 10 | 0.0839
0.3 | 0.8 | 20 | 0.2311
0.3 | 0.8 | 100 | 0.1837
0.3 | 0.8 | 1,000 | 0.1461
0.3 | 0.9 | 10 | 0.1375
0.3 | 0.9 | 20 | 0.1237
0.3 | 0.9 | 100 | 0.0844
0.3 | 0.9 | 1,000 | 0.0736
0.4 | 0.1 | 10 | 0.2188
0.4 | 0.1 | 20 | 0.2109
0.4 | 0.1 | 100 | 0.1754
0.4 | 0.1 | 1,000 | 0.1455
0.4 | 0.2 | 10 | 0.1344
0.4 | 0.2 | 20 | 0.1146
0.4 | 0.2 | 100 | 0.0997
0.4 | 0.2 | 1,000 | 0.2355
0.4 | 0.3 | 10 | 0.2075
0.4 | 0.3 | 20 | 0.1836
0.4 | 0.3 | 100 | 0.1391
0.4 | 0.3 | 1,000 | 0.1193
0.4 | 0.4 | 10 | 0.1073
0.4 | 0.4 | 20 | 0.1049
0.4 | 0.4 | 100 | 0.2301
0.4 | 0.4 | 1,000 | 0.2083
0.4 | 0.5 | 10 | 0.1914
0.4 | 0.5 | 20 | 0.1555
0.4 | 0.5 | 100 | 0.1074
0.4 | 0.5 | 1,000 | 0.0930
0.4 | 0.6 | 10 | 0.0995
0.4 | 0.6 | 20 | 0.0852
0.4 | 0.6 | 100 | 0.2117
0.4 | 0.6 | 1,000 | 0.1779
0.4 | 0.7 | 10 | 0.1607
0.4 | 0.7 | 20 | 0.1287
0.4 | 0.7 | 100 | 0.0874
0.4 | 0.7 | 1,000 | 0.0886
0.4 | 0.8 | 10 | 0.0781
0.4 | 0.8 | 20 | 0.2340
0.4 | 0.8 | 100 | 0.1793
0.4 | 0.8 | 1,000 | 0.1592
0.4 | 0.9 | 10 | 0.1420
0.4 | 0.9 | 20 | 0.1240
0.4 | 0.9 | 100 | 0.1061
0.4 | 0.9 | 1,000 | 0.0759
0.5 | 0.1 | 10 | 0.2396
0.5 | 0.1 | 20 | 0.2065
0.5 | 0.1 | 100 | 0.1644
0.5 | 0.1 | 1,000 | 0.1414
0.5 | 0.2 | 10 | 0.1308
0.5 | 0.2 | 20 | 0.1139
0.5 | 0.2 | 100 | 0.0935
0.5 | 0.2 | 1,000 | 0.2268
0.5 | 0.3 | 10 | 0.2150
0.5 | 0.3 | 20 | 0.1950
0.5 | 0.3 | 100 | 0.1395
0.5 | 0.3 | 1,000 | 0.1150
0.5 | 0.4 | 10 | 0.1029
0.5 | 0.4 | 20 | 0.1019
0.5 | 0.4 | 100 | 0.2335
0.5 | 0.4 | 1,000 | 0.2097
0.5 | 0.5 | 10 | 0.1811
0.5 | 0.5 | 20 | 0.1583
0.5 | 0.5 | 100 | 0.1071
0.5 | 0.5 | 1,000 | 0.1072
0.5 | 0.6 | 10 | 0.0834
0.5 | 0.6 | 20 | 0.0865
0.5 | 0.6 | 100 | 0.2110
0.5 | 0.6 | 1,000 | 0.1754
0.5 | 0.7 | 10 | 0.1629
0.5 | 0.7 | 20 | 0.1280
0.5 | 0.7 | 100 | 0.0971
0.5 | 0.7 | 1,000 | 0.0797
0.5 | 0.8 | 10 | 0.0784
0.5 | 0.8 | 20 | 0.2378
0.5 | 0.8 | 100 | 0.1833
0.5 | 0.8 | 1,000 | 0.1388
0.5 | 0.9 | 10 | 0.1331
0.5 | 0.9 | 20 | 0.1081
0.5 | 0.9 | 100 | 0.0901
0.5 | 0.9 | 1,000 | *0.0675*
0.6 | 0.1 | 10 | 0.2311
0.6 | 0.1 | 20 | 0.2155
0.6 | 0.1 | 100 | 0.1667
0.6 | 0.1 | 1,000 | 0.1556
0.6 | 0.2 | 10 | 0.1351
0.6 | 0.2 | 20 | 0.1135
0.6 | 0.2 | 100 | 0.1056
0.6 | 0.2 | 1,000 | 0.2303
0.6 | 0.3 | 10 | 0.2089
0.6 | 0.3 | 20 | 0.1915
0.6 | 0.3 | 100 | 0.1509
0.6 | 0.3 | 1,000 | 0.1183
0.6 | 0.4 | 10 | 0.1070
0.6 | 0.4 | 20 | 0.0856
0.6 | 0.4 | 100 | 0.2253
0.6 | 0.4 | 1,000 | 0.2090
0.6 | 0.5 | 10 | 0.1864
0.6 | 0.5 | 20 | 0.1631
0.6 | 0.5 | 100 | 0.1118
0.6 | 0.5 | 1,000 | 0.0971
0.6 | 0.6 | 10 | 0.0778
0.6 | 0.6 | 20 | 0.0817
0.6 | 0.6 | 100 | 0.2173
0.6 | 0.6 | 1,000 | 0.1775
0.6 | 0.7 | 10 | 0.1501
0.6 | 0.7 | 20 | 0.1241
0.6 | 0.7 | 100 | 0.0870
0.6 | 0.7 | 1,000 | 0.0797
0.6 | 0.8 | 10 | 0.0847
0.6 | 0.8 | 20 | 0.2358
0.6 | 0.8 | 100 | 0.1839
0.6 | 0.8 | 1,000 | 0.1587
0.6 | 0.9 | 10 | 0.1371
0.6 | 0.9 | 20 | 0.1132
0.6 | 0.9 | 100 | 0.0997
0.6 | 0.9 | 1,000 | 0.0863
0.7 | 0.1 | 10 | 0.2339
0.7 | 0.1 | 20 | 0.2093
0.7 | 0.1 | 100 | 0.1595
0.7 | 0.1 | 1,000 | 0.1483
0.7 | 0.2 | 10 | 0.1348
0.7 | 0.2 | 20 | 0.1075
0.7 | 0.2 | 100 | 0.0890
0.7 | 0.2 | 1,000 | 0.2187
0.7 | 0.3 | 10 | 0.2239
0.7 | 0.3 | 20 | 0.1798
0.7 | 0.3 | 100 | 0.1406
0.7 | 0.3 | 1,000 | 0.1169
0.7 | 0.4 | 10 | 0.0912
0.7 | 0.4 | 20 | 0.0944
0.7 | 0.4 | 100 | 0.2441
0.7 | 0.4 | 1,000 | 0.2151
0.7 | 0.5 | 10 | 0.1833
0.7 | 0.5 | 20 | 0.1663
0.7 | 0.5 | 100 | 0.1179
0.7 | 0.5 | 1,000 | 0.0972
0.7 | 0.6 | 10 | 0.0904
0.7 | 0.6 | 20 | 0.0869
0.7 | 0.6 | 100 | 0.2039
0.7 | 0.6 | 1,000 | 0.1943
0.7 | 0.7 | 10 | 0.1590
0.7 | 0.7 | 20 | 0.1289
0.7 | 0.7 | 100 | 0.0913
0.7 | 0.7 | 1,000 | 0.0811
0.7 | 0.8 | 10 | 0.0893
0.7 | 0.8 | 20 | 0.2375
0.7 | 0.8 | 100 | 0.1887
0.7 | 0.8 | 1,000 | 0.1484
0.7 | 0.9 | 10 | 0.1430
0.7 | 0.9 | 20 | 0.1059
0.7 | 0.9 | 100 | 0.0782
0.7 | 0.9 | 1,000 | 0.0755
0.8 | 0.1 | 10 | 0.2353
0.8 | 0.1 | 20 | 0.2178
0.8 | 0.1 | 100 | 0.1627
0.8 | 0.1 | 1,000 | 0.1498
0.8 | 0.2 | 10 | 0.1203
0.8 | 0.2 | 20 | 0.1233
0.8 | 0.2 | 100 | 0.0925
0.8 | 0.2 | 1,000 | 0.2357
0.8 | 0.3 | 10 | 0.2109
0.8 | 0.3 | 20 | 0.1863
0.8 | 0.3 | 100 | 0.1432
0.8 | 0.3 | 1,000 | 0.1246
0.8 | 0.4 | 10 | 0.1082
0.8 | 0.4 | 20 | 0.1023
0.8 | 0.4 | 100 | 0.2293
0.8 | 0.4 | 1,000 | 0.2125
0.8 | 0.5 | 10 | 0.1751
0.8 | 0.5 | 20 | 0.1630
0.8 | 0.5 | 100 | 0.1205
0.8 | 0.5 | 1,000 | 0.0936
0.8 | 0.6 | 10 | 0.0884
0.8 | 0.6 | 20 | 0.0804
0.8 | 0.6 | 100 | 0.2186
0.8 | 0.6 | 1,000 | 0.1776
0.8 | 0.7 | 10 | 0.1602
0.8 | 0.7 | 20 | 0.1250
0.8 | 0.7 | 100 | 0.0953
0.8 | 0.7 | 1,000 | 0.0816
0.8 | 0.8 | 10 | 0.0778
0.8 | 0.8 | 20 | 0.2266
0.8 | 0.8 | 100 | 0.1781
0.8 | 0.8 | 1,000 | 0.1522
0.8 | 0.9 | 10 | 0.1370
0.8 | 0.9 | 20 | 0.1122
0.8 | 0.9 | 100 | **0.0597**
0.8 | 0.9 | 1,000 | 0.0732
0.9 | 0.1 | 10 | 0.2355
0.9 | 0.1 | 20 | 0.2089
0.9 | 0.1 | 100 | 0.1710
0.9 | 0.1 | 1,000 | 0.1554
0.9 | 0.2 | 10 | 0.1355
0.9 | 0.2 | 20 | 0.1155
0.9 | 0.2 | 100 | 0.1068
0.9 | 0.2 | 1,000 | 0.2370
0.9 | 0.3 | 10 | 0.2120
0.9 | 0.3 | 20 | 0.1828
0.9 | 0.3 | 100 | 0.1432
0.9 | 0.3 | 1,000 | 0.1210
0.9 | 0.4 | 10 | 0.1080
0.9 | 0.4 | 20 | 0.0945
0.9 | 0.4 | 100 | 0.2296
0.9 | 0.4 | 1,000 | 0.2073
0.9 | 0.5 | 10 | 0.1874
0.9 | 0.5 | 20 | 0.1612
0.9 | 0.5 | 100 | 0.1059
0.9 | 0.5 | 1,000 | 0.0900
0.9 | 0.6 | 10 | 0.0940
0.9 | 0.6 | 20 | 0.0753
0.9 | 0.6 | 100 | 0.2092
0.9 | 0.6 | 1,000 | 0.1892
0.9 | 0.7 | 10 | 0.1628
0.9 | 0.7 | 20 | 0.1460
0.9 | 0.7 | 100 | 0.0971
0.9 | 0.7 | 1,000 | 0.0915
0.9 | 0.8 | 10 | 0.0786
0.9 | 0.8 | 20 | 0.2337
0.9 | 0.8 | 100 | 0.1866
0.9 | 0.8 | 1,000 | 0.1683
0.9 | 0.9 | 10 | 0.1431
0.9 | 0.9 | 20 | 0.1117
0.9 | 0.9 | 100 | 0.0765
0.9 | 0.9 | 1,000 | 0.0800

</p></details>

#### Parameter Evaluation for Self-Organising Map (SOM)

<details><summary>Show Table</summary>
<p>

Parameter ![](https://github.com/Cognitana-Research/NNOHD/blob/master/images/n.png "n") | Epochs ![](https://github.com/Cognitana-Research/NNOHD/blob/master/images/n_e.png "n^e") | ![](https://github.com/Cognitana-Research/NNOHD/blob/master/images/D_n_n_e.png "D_{n,n^e}")
---|---|---
1 | 10 | 0.0496
1 | 20 | 0.0649
1 | 100 | 0.0496
1 | 1,000 | 0.0496
2 | 10 | **0.0221**
2 | 20 | 0.0330
2 | 100 | *0.0223*
2 | 1,000 | *0.0223*
3 | 10 | *0.0315*
3 | 20 | 0.0466
3 | 100 | 0.0332
3 | 1,000 | 0.0330
4 | 10 | 0.0399
4 | 20 | 0.0509
4 | 100 | 0.0543
4 | 1,000 | 0.0554
5 | 10 | 0.0768
5 | 20 | 0.0932
5 | 100 | 0.0850
5 | 1,000 | 0.0859
6 | 10 | 0.0866
6 | 20 | 0.0906
6 | 100 | 0.0901
6 | 1,000 | 0.0890
7 | 10 | 0.1040
7 | 20 | 0.1186
7 | 100 | 0.1117
7 | 1,000 | 0.1132
8 | 10 | 0.0995
8 | 20 | 0.1103
8 | 100 | 0.1107
8 | 1,000 | 0.1073
9 | 10 | 0.1144
9 | 20 | 0.1207
9 | 100 | 0.1095
9 | 1,000 | 0.1103
10 | 10 | 0.1314
10 | 20 | 0.1425
10 | 100 | 0.1166
10 | 1,000 | 0.1144
11 | 10 | 0.1807
11 | 20 | 0.1724
11 | 100 | 0.1596
11 | 1,000 | 0.1642
12 | 10 | 0.1815
12 | 20 | 0.1771
12 | 100 | 0.1742
12 | 1,000 | 0.1667
13 | 10 | 0.1935
13 | 20 | 0.1919
13 | 100 | 0.1784
13 | 1,000 | 0.1826
14 | 10 | 0.1967
14 | 20 | 0.1898
14 | 100 | 0.1885
14 | 1,000 | 0.1969
15 | 10 | 0.2003
15 | 20 | 0.1965
15 | 100 | 0.2023
15 | 1,000 | 0.2083
16 | 10 | 0.2192
16 | 20 | 0.2230
16 | 100 | 0.2111
16 | 1,000 | 0.2175
17 | 10 | 0.2241
17 | 20 | 0.2218
17 | 100 | 0.2176
17 | 1,000 | 0.2075
18 | 10 | 0.2269
18 | 20 | 0.2271
18 | 100 | 0.2136
18 | 1,000 | 0.2234
19 | 10 | 0.2483
19 | 20 | 0.2331
19 | 100 | 0.2493
19 | 1,000 | 0.2291
20 | 10 | 0.2339
20 | 20 | 0.2227
20 | 100 | 0.2348
20 | 1,000 | 0.2346

</p></details>

## Acknowledgements

This repository features code and data sets from other projects. We would like to acknowledge the following contributions:

### Code

* [ELKI Data Mining Toolkit](https://github.com/elki-project/elki) by Erich Schubert, Alexander Koos, Tobias Emrich, Andreas Züﬂe, Klaus Arthur Schmid, Arthur Zimek. A Framework for Clustering Uncertain Data. Proceedings of the VLDB Endowment, 8(12): pp.1976–1979 (2015). [VLDB](http://www.vldb.org/pvldb/vol8/p1976-schubert.pdf).

* [Somoclu](https://github.com/peterwittek/somoclu) by Peter Wittek, Shi Chao Gao, Ik Soo Lim, Li Zhao. Somoclu: An Efficient Parallel Library for Self-Organizing Maps. Journal of Statistical Software, 78(9), pp.1-21 (2017). DOI:[10.18637/jss.v078.i09](https://doi.org/10.18637/jss.v078.i09). arXiv:[1305.1422](https://arxiv.org/abs/1305.1422).

### Data Sets

* [DAMI](http://www.dbs.ifi.lmu.de/research/outlier-evaluation/DAMI/) by Guilherme O. Campos, Arthur Zimek, Jörg Sander, Ricardo J. G. B. Campello, Barbora Micenková, Erich Schubert, Ira Assent, Michael E. Houle (2016). On the Evaluation of Unsupervised Outlier Detection: Measures, Datasets, and an Empirical Study. Data Mining and Knowledge
Discovery 30(4), Springer Nature, pp.891-927 (2016). DOI:[10.1007/s10618-015-0444-8](https://doi.org/10.1007/s10618-015-0444-8)

* [ODDS Library](http://odds.cs.stonybrook.edu) by Shebuti Rayana. Stony Brook University, Department of Computer Sciences, Stony Brook, NY (2016).

* [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml) by Dheeru Dua, Casey Graff. University of California, Irvine, School of Information and Computer Sciences (2017).
