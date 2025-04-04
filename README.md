<p align="center">
<h1 align="center">
  <img src="./assets/icon.png" alt="icon" style="width: 30px; vertical-align: -4px; margin-right: 3px;">
    PIG: Physics-Informed Gaussians as Adaptive Parametric Mesh Representations
</h1>
<h3 align="center">
    ICLR 2025
</h3>
   <p align="center">
    <a href="https://github.com/NamGyuKang">Namgyu Kang</a>
    ·
    <a href="https://jaeminoh.github.io/">Jaemin Oh</a>
    ·
    <a href="https://www.youngjoonhong.com/">Youngjoon Hong</a>
    ·
    <a href="https://silverbottlep.github.io/">Eunbyung Park</a>    
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2412.05994">Paper</a> | <a href="https://namgyukang.github.io/Physics-Informed-Gaussians/">Project Page</a> </h3>
  <div align="center"></div>
</p>

<p align="center">
  <a href="">
    <img src="./assets/teaser.jpg" alt="Logo" width="100%">
  </a>
</p>

https://github.com/user-attachments/assets/ddfd6949-f459-40b8-bcb6-67558b980c09

https://github.com/user-attachments/assets/e6289032-aa43-4adc-a62f-8b015a0bf73f

https://github.com/user-attachments/assets/443ed3b0-f44c-49ae-8cac-f48e86704120

## Quick Start

## 1. Installation

### Clone Physics-Informed-Gaussians repo

```
git clone https://github.com/NamGyuKang/Physics-Informed-Gaussians.git
cd Physics-Informed-Gaussians
```

### Create JAX environment (Flow-Mixing, Klein-Gordon, Nonliner-Diffusion Eq.)
Please follow the steps in the Jax_gpu_version_installation.txt file to install JAX GPU version.

### Create Pytorch environment (Helmholtz Eq.)
The code is tested with Python (3.8, 3.9) and PyTorch (1.11, 11.2) with CUDA (>=11.3). 
You can create an anaconda environment with those requirements by running:

- if you use CUDA 11.3, Pytorch 1.11, Python 3.9, `conda env create -f CUDA_11_3_Pytorch_1_11_Py_3_9.yml`
- or with CUDA 11.6, Pytorch 1.12, Python 3.8, `conda env create -f CUDA_11_6_Pytorch_1_12_Py_3_8.yml`
- and then `conda activate pig`

## 2. Run the code in each folder

- `CUDA_VISIBLE_DEVICES=0 bash flow_mixing3d_pig.sh`
- `CUDA_VISIBLE_DEVICES=0 bash helmholtz2d_pig.sh`
- `CUDA_VISIBLE_DEVICES=0 bash klein_gordon3d_pig.sh`
- `CUDA_VISIBLE_DEVICES=0 bash diffusion3d_pig.sh`

## Citation
If you find this code useful in your research, please consider citing us!

```bibtex
@article{kang2024pig,
  title={PIG: Physics-Informed Gaussians as Adaptive Parametric Mesh Representations},
  author={Kang, Namgyu and Oh, Jaemin and Hong, Youngjoon and Park, Eunbyung},
  journal={arXiv preprint arXiv:2412.05994},
  year={2024}
}
```

## Contact
Contact [Namgyu Kang](mailto:kangnamgyu27@gmail.com) if you have any further questions.

## Acknowledgements
This project is built on top of several outstanding repositories: [SPINN](https://github.com/stnamjef/SPINN), [PIXEL](https://github.com/NamGyuKang/PIXEL), [JAXPI](https://github.com/PredictiveIntelligenceLab/jaxpi). We thank the original authors for opensourcing their excellent work.

