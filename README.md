# Official open-source code of "Physics-guided neural network integrating uncertainty evolution for spatiotemporal wind speed prediction"

Accurate short-term wind forecasts are important for wind power operation, grid scheduling, and intelligent energy control, but wind speed changes are difficult to predict because they vary across space, time, and height. Here we show that a physics-guided neural network improves the prediction of three-dimensional wind speed fields by learning both data patterns and the physical consistency of wind evolution. The model combines information across time, space, and height, and uses physical constraints to guide the learning of wind speed, wind direction, and spatial structure. Tests show that the model predicts wind speed more accurately than current machine-learning methods within a twenty-four-hour forecast period at both ten-metre and one-hundred-metre heights. Compared with a leading numerical weather forecasting system, it gives more accurate wind speed forecasts within five hours and more accurate wind direction forecasts within twelve hours.  The model also exhibits strong regional generalization and temporal robustness, providing support for practical wind power applications.

# PD-WPN

[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Instructions to run on data](#Instructions-to-run-on-data)
- [License](#license)


# System Requirements
## Hardware requirements
`PD-WPN` requires a standard computer with enough RAM to support in-memory operations and a high-performance GPU to support fast operations on high-dimensional data.

## Software requirements
### OS Requirements
This package is supported for *Windows* and *Linux*. The package has been tested on the following systems:
+ Windows: Windows 10 22H2
+ Linux: Ubuntu 16.04

### Python Dependencies
`PD-WPN` mainly depends on the Python scientific stack.

```
einops==0.8.0
fbm==0.3.0
matplotlib==3.7.2
numpy==1.24.3
pandas==2.0.3
pmdarima==2.0.4
ptflops==0.7.3
pynvml==11.5.3
scikit_learn==1.5.1
scipy==1.10.1
seaborn==0.13.2
sympy==1.12
torch==2.3.1
torch_cluster==1.6.3
tqdm==4.66.4
tvm==1.0.0
xarray==2022.11.0
```

# Instructions to run on data

Due to the large size of the data and weight files, we host them on other data platforms, please download the relevant data from the link below.

Input data：[https://drive.google.com/drive/folders/1UL9s_8Hw5OosQpq7dbhSltL0VkQX3rRC1ULrRC?usp=drive_link](https://drive.google.com/drive/folders/1UL9s_8Hw5OosQpq7dbhSltL0VkQX3rRC1ULrRC?usp=drive_link)

Weight data：[https://drive.google.com/drive/folders/145mJluB1TqjFuxsfgfFWrG6sg94W4FmY145mFmY?usp=drive_link](https://drive.google.com/drive/folders/145mJluB1TqjFuxsfgfFWrG6sg94W4FmY145mFmY?usp=drive_link)

```
Put the downloaded data into the designated folder:

Execute utils/data_process.py to complete data processing and obtain model input;

Execute utils/time_process.py to generate the time series information corresponding to the samples;

Execute PD-WPN/train.py to train the PD-WPN model;

Execute main.py in any model folder to perform inference;

Execute evaluation.py in the model folder to complete the evaluation of the prediction results.
```

# License

This project is covered under the **MIT License**.
