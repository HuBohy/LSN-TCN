## :book: Laughs and Smiles detector based on Temporal Convolutional Network (ACII 2022)

> [[Paper](https://ieeexplore.ieee.org/abstract/document/9953896)] <br>
> [Hugo Bohy](https://scholar.google.com/citations?user=szbVZxcAAAAJ&hl=en&oi=ao), [Kevin El Haddad](https://scholar.google.com/citations?user=S3Q9SAsAAAAJ&hl=en), [Thierry Dutoit](https://scholar.google.com/citations?user=xxpvjOUAAAAJ&hl=en) <br>
> Numediart Institute
> University of Mons

### NDC-ME Dataset
Please contact [Kevin El Haddad](kevin.elhaddad@umons.ac.be) for details about the dataset.

## :wrench: Dependencies and Installation
- Python >= 3.9
- [Pytorch >= 2.0](https://pytorch.org/get-started/locally/)
- Option: NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

### Installation
1. Clone repo
    ```bash
    git clone https://github.com/HuBohy/LSN-TCN.git
    cd LSN-TCN
    ```

2. Install dependent packages

    ```bash
    pip install -r requirements.txt
    ```

## :computer: Training
Our LSN-TCN checkpoints can be found under the following link: [Drive](https://drive.google.com/drive/folders/1dFUjKKMw8VgKY_n2iOqbQqY4Yuz3RVdx?usp=drive_link)

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --data-video path/to/video/data --data-audio path/to/audio/data --label-path path/to/labels --mode fusion 
```
The result will be stored in ```logs/fusion/```.

## :scroll: Acknowledgement

 Our LSN-TCN implementation is inspired by [MS-TCN](https://arxiv.org/abs/1903.01945). We appreciate the authors of [MS-TCN](https://arxiv.org/abs/1903.01945) for making their codes available to public.

## :scroll: BibTeX
If you use this work in your project, please cite:
```
@inproceedings{bohy2022new,
  title={A new perspective on smiling and laughter detection: Intensity levels matter},
  author={Bohy, Hugo and El Haddad, Kevin and Dutoit, Thierry},
  booktitle={2022 10th International Conference on Affective Computing and Intelligent Interaction (ACII)},
  pages={1--8},
  year={2022},
  organization={IEEE}
}
```

### :e-mail: Contact

If you have any question or collaboration need, please email `hugo.bohy@umons.ac.be`.