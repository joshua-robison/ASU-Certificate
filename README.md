# ASU Machine Learning and Artificial Intelligence

This is a series of Jupyter notebooks and Python files containing my graduate school coursework.

## Table of Contents
- [Bayesian Modeling](#bayes-model)
- [k-Means Clustering](#k-means-model)
- [Convolutional Neural Network](#cnn-model)
- [Dynamic Bayesian Network](#dynamic-bayes-model)
- [Robot Neural Network](#collision-prediction-model)
- [Planning Domain Definition Language](#packman-pddl)

### [Bayes Model](https://github.com/joshua-robison/ASU-Certificate/blob/master/Notebooks/bayes_model.ipynb)
#### Data Used: [MNIST Image Data](https://github.com/joshua-robison/ASU-Certificate/blob/master/Datasets/bayes_data)
This notebook explains how to analyze a dataset, visualize features, generate training parameters, and model a bayesian classifier.

### [k-Means Model](https://github.com/joshua-robison/ASU-Certificate/blob/master/Notebooks/clustering_model.ipynb)
#### Data Used: [Data Points](https://github.com/joshua-robison/ASU-Certificate/blob/master/Datasets/)
This notebook demonstrates how to associate random data into clusters (based on some criteria) and visualize the convergence of a solution.

### [CNN Model](https://github.com/joshua-robison/ASU-Certificate/blob/master/Notebooks/cnn_model.ipynb)
#### Helper: [mnist.py](https://github.com/joshua-robison/ASU-Certificate/blob/master/Notebooks/mnist.py)
#### Data Used: [MNIST Image Data](https://github.com/joshua-robison/ASU-Certificate/blob/master/Datasets/cnn_data)
This notebook demonstrates how to construct a Convolutional Neural Network. Then it loads in MNIST image data, trains, tests and evaluates
on a subset of the data. For improved results, you could use more data, increase the number of epochs, and change the model structure.
This is simply an introduction and proof of concept.

### [Dynamic Bayes Model](https://github.com/joshua-robison/ASU-Certificate/blob/master/Notebooks/dynamic_bayes_network.ipynb)
This notebook explains how to analyze a problem and configure a dynamic bayesian network to simulate a stochastic process.

### [Collision Prediction Model](https://github.com/joshua-robison/ASU-Certificate/blob/master/Notebooks/RobotModel/)
This is a series of files that simulates a two dimensional robot equipped with sensors. The robot can randomly navigate the environment
collecting simulation data. This data is then pre-processed and split into training and validation sets. A neural network
model is initialized, trained, and evaluated with this data. Then the simulation can be re-run testing the proficiency of
our trained model. In order to maximize performance, more simulation data should be collected.

A [HOW-TO](https://github.com/joshua-robison/ASU-Certificate/blob/master/Notebooks/RobotModel/HOWTO.txt) is included in this folder with the workflow outlined.

### [Packman PDDL](https://github.com/joshua-robison/ASU-Certificate/blob/master/Notebooks/Packman/)
This contains a series of files using the Planning Domain Definition Language to solve "packman" puzzles. These files can be run at the following site:
http://editor.planning.domains/#

