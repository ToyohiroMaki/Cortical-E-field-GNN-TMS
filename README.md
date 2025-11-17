# Cortical Surface Electric Field Estimation for Real-Time TMS with Graph Neural Networks

---

## Description

- This repository contains the source code for the "Cortical Surface Electric Field Estimation for Real-Time TMS with Graph Neural Networks" (Maki et al., 2025).
- All imaging data used in this study is open and publicly available at the respective project webpage: the Human Connectome Project (https://www.humanconnectome.org/study/hcp-young-adult).

## Sample Inference Demo

To test the model pipeline without preparing any MRI or mesh data, you can run a simple demo using publicly available anatomical templates.
We provide a sample script that loads the sample T1-weighted brain image and performs E-field inference on a cortical mesh with a sample coil configuration.

### Requirements

```bash
mamba env create -f requirements.yml
conda activate sample
```

### Run the demo
```shell
python demo_infer.py
```

### Visualization
```shell
gmsh ./sample_out/result.msh
```

## Citation
```bibtex
@article{maki2025cortical,
  title={Cortical surface electric field estimation for real-time TMS with graph neural networks},
  author={Maki, Toyohiro and Yokota, Tatsuya and Hirata, Akimasa and Hontani, Hidekata},
  journal={Physics in Medicine and Biology},
  year={2025}
}
```
