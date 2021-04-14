# Exposing GAN-generated Faces Using Inconsistent Corneal Specular Highlights
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/)

Shu Hu, Yuezun Li, and Siwei Lyu
_________________

This repository is the official implementation of our paper 
"Exposing GAN-generated Faces Using Inconsistent Corneal Specular Highlights", 
which has been accepted by **ICASSP 2021**. 

[<a href="https://medium.com/@heizi.lyu/to-tell-lies-look-into-the-eyes-634bc889866a" target="_blank">Blog</a>] 
[<a href="https://cse.buffalo.edu/ubmdfl/projects/GAN_detect_iris/GAN_Iris.html" target="_blank">Project page</a>] 
[<a href="http://www.buffalo.edu/ubnow/stories/2021/03/deepfake-o-meter.html" target="_blank">News</a>]

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## How to run the code

#### Directly run the code on one of our provided image:

```setup
python main.py
```

#### Run the code on your own images:

```setup
python main.py -input <test image path> -output <path for saving result>
```

## Some results

```train
python AoRR/plot_aggregate_interpretation.py
```

<p align="center">
    <img src="gan_detect_iris_for_submit/outputs/seed000000_iris_final.jpg" height="400" width= "800">
</p>

## Citation
Please kindly consider citing our paper in your publications. 
```bash
@article{hu2020exposing,
  title={Exposing GAN-generated Faces Using Inconsistent Corneal Specular Highlights},
  author={Hu, Shu and Li, Yuezun and Lyu, Siwei},
  journal={arXiv preprint arXiv:2009.11924},
  year={2020}
}
```