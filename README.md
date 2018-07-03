# Toxic_TextCNN

This branch is to deploy the inference to movidius.

## Getting Started

clone this project.

### Prerequisites

```
python3
numpy
tensorflow_gpu
matplotlib
pandas
tensorflow

packages listed in requirements.txt.
```

## Usage

Explain how to run the automated tests for this system

### To train


```
make train
```

### To create the inference graph

```
make infer
```

### To compile the graph

```
cd graphs;
mvNCCompile toxic_textcnn_inference.meta -s 12 -in input -on output -o toxic_textcnn_inference.graph
```

### To try to run, in the main directory.
make run

## Notes
The makefile is messy; some things still reference old Makefile that I modified.
In general, pickle files are used to avoid redoing the preprocessing. If you
need to redo the preprocessing, make clean first.
