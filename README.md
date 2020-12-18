# SuperGlue PyTorch Implementation

## This SuperGlue implementation is intended for an application of Physarum Dynamics LP solver. Physarum Dynamics is a very fast and efficient LP solver that has the potential to be used to replace the original Sinkhorn Algorithm in SuperGlue. 

* Physarum Dynamics full paper PDF: [Physarum Powered Differentiable Linear Programming Layers and Applications](https://arxiv.org/abs/2004.14539).
* [SuperGlue with Physarum Dynamics implementation](https://github.com/HeatherJiaZG/Superglue-with-Physarum-Dynamics)

## Introduction
<p align="center">
  <img src="assets/superglue1.png" width="400"/>
</p>

 The SuperGlue network is a Graph Neural Network combined with an Optimal Matching layer that is trained to perform matching on two sets of sparse image features. SuperGlue operates as a "middle-end," performing context aggregation, matching, and filtering in a single end-to-end architecture. 

 Correspondences across images have some constraints:
 * A keypoint can have at most a single correspondence in the another image.
 * Some keypoints will be unmatched due to occlusion and failure of the detector.

 SuperGlue aims to find all correspondences between reprojections of the same points and identifying keypoints that have no matches. There are two main components in SuperGlue architecture: Attentional Graph Neural Network and Optimal Matching Layer.

<p align="center">
  <img src="assets/superglue2.png"/>
</p>

This repo includes PyTorch code for training the SuperGlue matching network on top of [SIFT](http://www.diva-portal.org/smash/get/diva2:480321/FULLTEXT02) keypoints and descriptors.
For more details, please see:
* Full paper PDF: [SuperGlue: Learning Feature Matching with Graph Neural Networks](https://arxiv.org/abs/1911.11763).

## Dependencies
* Python 3
* PyTorch >= 1.1
* OpenCV >= 3.4 (4.1.2.30 recommended for best GUI keyboard interaction, see this [note](#additional-notes))
* Matplotlib >= 3.1
* NumPy >= 1.18

Simply run the following command: `pip3 install numpy opencv-python torch matplotlib`

Or create a conda environment by `conda install --name myenv --file superglue.txt`

## Contents
There are two main top-level scripts in this repo:

1. `train.py` : trains the superglue model.
2. `load_data.py`: reads images from files and creates pairs. It generates keypoints, descriptors and ground truth matches which will be used in training.

### Download Data
Download the COCO2014 dataset files for training
```
wget http://images.cocodataset.org/zips/train2014.zip
```
Download the validation set
```
wget http://images.cocodataset.org/zips/val2014.zip
```
Download the test set
```
wget http://images.cocodataset.org/zips/test2014.zip
```

### Training Directions

To train the SuperGlue with default parameters, run the following command:

```sh
python train.py
```

### Additional useful command line parameters
* Use `--epoch` to set the number of epochs (default: `20`).
* Use `--train_path` to set the path to the directory of training images.
* Use `--eval_output_dir` to set the path to the directory in which the visualizations is written (default: `dump_match_pairs/`).
* Use `--show_keypoints` to visualize the detected keypoints (default: `False`).
* Use `--viz_extension` to set the visualization file extension (default: `png`). Use pdf for highest-quality.

### Visualization Demo
The matches are colored by their predicted confidence in a jet colormap (Red: more confident, Blue: less confident).

You should see images like this inside of `dump_match_pairs/`

<img src="assets/8349_matches.png" width="800">
<img src="assets/4599_matches2.png" width="800">
<img src="assets/2799_matches.png" width="800">
<img src="assets/3849_matches2.png" width="800">
<img src="assets/3949_matches.png" width="800">


