# Official open-source code of "Physics-Informed Neural Network Based on Evolutionary Feature Uncertainty Fusion for Wind Speed Spatiotemporal Field Prediction"

To improve the medium and short-term prediction accuracy of the three-dimensional spatiotemporal field of wind speed while taking into account physical consistency and model efficiency, this project proposes a neural network architecture that combines the fusion of evolutionary feature uncertainty and physical information modeling, named PD-WPN. PD-WPN has achieved high-precision joint prediction and three-dimensional reconstruction of wind speed fields at five heights ranging from 10m to 100m. The prediction accuracy of wind direction within a 12-hour lead time is higher than that of the internationally leading ECMWF-HRES model. The prediction errors of wind speed at heights of 10 meters and 100 meters within a 24-hour lead time are significantly lower than other SOTA models. The migration experiments and robustness analyses show that this method has good regional generalization ability and time robustness. PD-WPN achieves a good balance between performance and efficiency,its performance in multi-height field modeling, spatial structure consistency and wind direction prediction proves that it has significant application potential in scenarios such as wind power dispatching, wind energy assessment and intelligent control.

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
`TDNN` mainly depends on the Python scientific stack.

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

Input data：[https://drive.google.com/drive/folders/1UL9s_8Hw5OosQpq7dbhSltL0VkQX3rRC?usp=drive_link](https://drive.google.com/drive/folders/1UL9s_8Hw5OosQpq7dbhSltL0VkQX3rRC?usp=drive_link)

Weight data：[https://drive.google.com/drive/folders/145mJluB1TqjFuxsfgfFWrG6sg94W4FmY?usp=drive_link](https://drive.google.com/drive/folders/145mJluB1TqjFuxsfgfFWrG6sg94W4FmY?usp=drive_link)

Put the downloaded data into the designated folder:

Execute utils/data_process.py to complete data processing and obtain model input;

Execute utils/time_process.py to generate the time series information corresponding to the samples;

Execute PD-WPN/train.py to train the PD-WPN model;

Execute main.py in any model folder to perform inference and use evaluation.py in the model folder to complete the evaluation of the prediction results.

# License

This project is covered under the **Apache 2.0 License**.
