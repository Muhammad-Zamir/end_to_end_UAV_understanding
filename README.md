# End_to_End Anti-UAV intetnt understanding*

This repository is the offical implementation of [**End_to_End Anti-UAV intetnt understanding**](https://arxiv.org/abs/).

## Installation

[**Python>=3.10.0**](https://www.python.org/) is required 

```bash
conda create -n "environment_name" python=3.12
conda activate "environment_name" 
pip install -r requirements.txt
```

## Data Preparation

We currently support [**NPS**](https://engineering.purdue.edu/~bouman/UAV_Dataset/), [**TDUAV**](https://huggingface.co/datasets/yifwang/MM-AntiUAV/tree/main), and [**Anti-UAV**](https://github.com/ZhaoJ9014/Anti-UAV) datasets. Follow the instructions below to prepare datasets.

* **Ground-Truth**: We recommend to leverage with the ground truth (for NPS ) prepared by Dog-Fight paper [Download](https://github.com/mwaseema/Drone-Detection):

* **Dataset - preprocessing**: Simply extract frames and convert dataset into yolo format, and make sure the path for dataset videos and annotations are correct:

## Training

You can change dataset path and parameters setting from, [config.py](https://github.com/Muhammad-Zamir/end_to_end_UAV_understanding/blob/main/config.py) or from the default parameter file [config.py](https://github.com/Muhammad-Zamir/end_to_end_UAV_understanding/blob/main/config.py).
* **Training**

```bash
python trainer.py --dataset MultiUAV --epochs 50 --output_dir ./checkpoints

```

## Testing
Run commands below to compute evaluation results (BLEU1, BLEU@, BLEU#, BLEU4, SPICE, ACC, METEOR).

```bash
python val.py \
    --dataset MultiUAV \
    --checkpoint ./checkpoints/capita_multiuav/best_model.pth \
    --output_dir ./eval_results
```
Change the dataset and checkpoint path according to your own local setup.

* **Ablation study**: To do Ablations study search the keyword "Ablation" in the [model.py](https://github.com/Muhammad-Zamir/end_to_end_UAV_understanding/blob/main/model.) file and un comment the ablation lines and comment the original line:


* **LLM Choice**: To change the LLM  model, just rename the model name in [config.py](https://github.com/Muhammad-Zamir/end_to_end_UAV_understanding/blob/main/config.py):

## Pretrained Weights

- [x] Please find the pre-trained weights of TDUAV dataset from [Google Drive](https://drive.google.com/drive/folders/1WmEStgIurUSdGDp-udN1C9SNuCD4Gl6b)


## Acknowledgment

...........

## Citation

If you find this work useful in your research, please kindly cite the paper:

```
waiting
```



