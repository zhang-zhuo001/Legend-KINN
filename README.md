# Legend-KINN: A Legendre Polynomial-Based Kolmogorov-Arnold-Informed Neural Network for Efficient PDE Solving

[//]: # (<p align="center">)

[//]: # (<a href="" alt="arXiv">)

[//]: # (    <img src="https://img.shields.io/badge/arXiv-2405.07992-b31b1b.svg?style=flat" /></a>)


<p align="center"><em>Zhuo Zhang, Xiong Xiong, Sen Zhang, Wei Wang, Yanxu Zhong, Canqun Yang, Xi Yang</em></p>

> Note: This project is currently under development and awaiting peer review of a research paper.

<p align="center">
<img src="https://github.com/zhang-zhuo001/misc/blob/main/run.png?raw=true" width="860"> <br>

</p>


This is a PyTorch implementation of Legend-KINN proposed by our paper "Legend-KINN: A Legendre Polynomial-Based Kolmogorov-Arnold-Informed Neural Network for Efficient PDE Solving". 

We are open-sourcing a partial implementation of the core code, pretrained weights, and some experimental results for the community to preview and use.

The complete code, detailed documentation, and final model weights will be released upon the paper's official acceptance.


## Updates
* 📝August, 2025: This repository has been created with a partial release of the code and weights.

* 📝May, 2025: The paper has been submitted for review.
---

## How to Use？

### 1. Requirements
PyTorch, numpy, and matplotlib.
### 2. Installation
Clone this repository and install the dependencies:

```bash
git clone https://github.com/zhang-zhuo001/Legend-KINN.git
cd Legend-KINN
```

Data preparation: For your convenience, we have provided an example dataset in the [data](/data/). folder. This data is structured to work directly with our scripts.

### 3. Quick Start ▶️
```python
python LegendKINN_cylinder.py
```

We've provided code for batch training. To begin, simply specify the parameters you wish to train in the nu_values dictionary within the main script.

For example, to train models with a range of nu values, you would define the dictionary like this:
```python
nu_values = {
    1.25e-2: '1.25e-2',
    1.25e-3: '1.25e-3',
    1e-2: '1e-2',
    2.5e-3: '2.5e-3',
    2e-2: '2e-2',
    2e-3: '2e-3',
    5e-3: '5e-3'
}
```

### 4. Validation and Visualization

To validate your trained models and visualize the results, you can use the `vis.py` script.

To customize which models you visualize, modify the `nu_mapping` dictionary within the `vis.py` script. For example, to visualize a specific set of nu values, use:

```python
nu_mapping = {
    1.25e-2: '1.25e-2',
    ...
    5e-3: '5e-3'
}
```

The script automatically handles file paths for checkpoints, data, and saving visualization results. The default paths are set to:

  * **Checkpoints**: `checkpoint/`
  * **Data**: `data/`
  * **Save Visualizations**: `vis_result/`

To run the evaluation and generate visualizations, simply execute the following command from the project root directory:

```bash
python vis.py
```

![LegendKINN second figure](https://github.com/zhang-zhuo001/misc/blob/main/cylinder_mlpvslegend%20(1).png?raw=true)
Figure 1: Flow field prediction visualization: our proposed method (Legend-KINN) vs. MLP in Flow Past a Circular Cylinder.

![LegendKINN third figure](https://github.com/zhang-zhuo001/misc/blob/main/flap_mlpvslegend.png?raw=true)
Figure 2: Flow field prediction visualization: our proposed method (Legend-KINN) vs. MLP in Flow Over a Narrow Column.

![LegendKINN third figure](https://github.com/zhang-zhuo001/misc/blob/main/forward_mlpvslegend.png?raw=true)
Figure 3: Flow field prediction visualization: our proposed method (Legend-KINN) vs. MLP in Flow Over a Forward-Facing Step.


## Models
We compared the training time and epochs needed for different methods to reach L2 relative errors of 50%, 20%, and 10%. (Note: "N/A" means the model failed to converge.)

![LegendKINN first figure](https://github.com/zhang-zhuo001/misc/blob/main/time_cylinder.png?raw=true)
Figure 4: Comparison of Time and Epochs for Different Methods to Achieve 50%, 20%, and 10% L2 Relative Error in Flow Past a Circular Cylinder.

The results show Legend-KINN consistently has the best convergence, achieving target errors with significantly less time and fewer epochs than other methods.

## Models
### Legend-KINN trained on Cylinder (flow around a circular cylinder)
| Model                                                                                                    | Layers | Params | FLOP  | Log |
|:---------------------------------------------------------------------------------------------------------|   :---:    |  :---: |  :---:  |  :---:  |
| [MLP (nu=0.01)](https://github.com/zhang-zhuo001/Legend-KINN/releases/download/model/mlp_0.01.pth)         | [2, 46, 46, 46, 3] | 4.603 | 285,568 | [log](https://github.com/zhang-zhuo001/Legend-KINN/releases/download/model/0.01_mlp.csv) |
| [MLP (nu=0.02)](https://github.com/zhang-zhuo001/Legend-KINN/releases/download/model/mlp_0.02.pth)          | [2, 46, 46, 46, 3] | 4.603 | 285,568 |  [log](https://github.com/zhang-zhuo001/Legend-KINN/releases/download/model/0.02_mlp.csv) |
| [Legend-KINN (nu=0.01)](https://github.com/zhang-zhuo001/Legend-KINN/releases/download/model/legend_0.01.pth)  | [2, 20, 20, 20, 3] | 4,500 | 65,536 | [log](https://github.com/zhang-zhuo001/Legend-KINN/releases/download/model/0.01_legend.csv) |
| [Legend-KINN (nu=0.02)](https://github.com/zhang-zhuo001/Legend-KINN/releases/download/model/legend_0.02.pth) | [2, 20, 20, 20, 3] | 4,500 | 65,536 | [log](https://github.com/zhang-zhuo001/Legend-KINN/releases/download/model/0.02_legend.csv) |




## Bibtex
```
@article{zhang2025legend,
  title={Legend-KINN: A Legendre Polynomial-Based Kolmogorov-Arnold-Informed Neural Network for Efficient PDE Solving},
  author={Zhang, Zhuo and Xiong, Xiong and Zhang, Sen and Wang, Wei and Zhong, Yanxu and Yang, Canqun and Yang, Xi},
  journal={arXiv preprint arXiv:XXXX.XXXXX (Coming Soon)},
  year={2025}
}
```
## Contact

We welcome academic discussions and collaboration with researchers and developers interested in this field. If you have any ideas or suggestions for collaboration, please don't hesitate to reach out to us at [zhangzhuo@nudt.edu.cn](zhangzhuo@nudt.edu.cn).

## Acknowledgment
We are grateful for the computing resources provided by the National Supercomputing Center in Tianjin and the National Supercomputing Center in Guangzhou. 

Our implementation is based on several excellent open-source projects:


  * [PINN](https://github.com/maziarraissi/PINNs)
  * [TSONN](https://github.com/Cao-WenBo/TSONN)
  * [KINN](https://github.com/yizheng-wang/Research-on-Solving-Partial-Differential-Equations-of-Solid-Mechanics-Based-on-PINN)
  * [KAN](https://github.com/KindXiaoming/pykan)

