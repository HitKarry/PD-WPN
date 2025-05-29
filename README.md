# Official open-source code of "Multivariate Meteorological Data Temporal Downscaling Based on High Freedom Dynamic Collaboration of Flows"

Temporal downscaling is one of the most challenging topics in meteorological data processing research. Traditional methods often face problems such as high computational costs and poor generalization capabilities. Frame interpolation methods based on deep learning have provided new ideas for the time downscaling of meteorological data. In this paper, a deep neural network for time downscaling of multivariate meteorological data is designed. It constructs data frames by independently estimating the kernel weights and offset vectors for each target pixel among different meteorological variables and dynamically fusing multivariate information. It generates output frames under the guidance of the feature space. Compared with other methods, the proposed model can handle a wide range of complex meteorological movements. The experimental results show that the proposed method has good performance, robustness, efficiency, scalability, and transferability in the downscaling of multivariate meteorological fields.

# TDNN

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

TDNN is a deep neural network model based on "High Freedom Dynamic Collaboration of Flows", which can handle the task of downscaling multivariate meteorological data. Open source data includes neural networks, testing frameworks, sample input data, and corresponding expected outputs.

- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Instructions to run on data](#Instructions-to-run-on-data)
- [License](#license)


# System Requirements
## Hardware requirements
`TDNN` requires a standard computer with enough RAM to support in-memory operations and a high-performance GPU to support fast operations on high-dimensional data.

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

Input data：[https://mega.nz/file/jIEAzAhI#_PWOKOwGBvpAF_yOpYe7uksy8LOmnZta6f2I55kk5fA](https://mega.nz/file/coYRWS5Z#pumyUUJRKGAVGAvtaT6e-grE9-sSHHxzBKBKrlYHTSA)

Output data：[https://mega.nz/file/fRVm1T7S#SWHCbu2tkFpME3Y-Eh7LWz15aVV4yI-4U-ZAqi_QpdA](https://mega.nz/file/50YVgbpZ#vkBv0853QdyRdP7bFhBWe_0rT6B2iWbivAA7h1fkmBM)

Weight data：[https://mega.nz/file/2cdBULiB#boGkh154f_97hbpbFIzYHl7j7iVLh_93gAeNH46L0EQ](https://mega.nz/file/584ChbAD#G48qGbpOSFxT8ZtUlbbkQAyJmxnH9BdtDVaY9XxAgnA)

Put the downloaded data into the specified folder, execute the main.py run and generate the resulting data. evaluate.py provides the functions necessary for quantitative evaluation of the data.

# License

This project is covered under the **Apache 2.0 License**.
