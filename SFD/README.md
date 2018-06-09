# Test instructions for S³FD: Single Shot Scale-invariant Face Detector

### Introduction

S³FD is a real-time face detector, which performs superiorly on various scales of faces with a single deep neural network, especially for small faces. For more details, please refer to their original [arXiv paper](https://arxiv.org/abs/1708.05237).

### Preparation

1. To install the proper version of Caffe, follow the [SSD Installation Instructions](./SSD-install.md)

2. Download the authors' [pre-trained model](https://drive.google.com/open?id=1CboBIsjcDQ-FC1rMES6IjTl6sYQDoD6u).
    
### Datasets

You can also download them from:
1. [AFW](http://www.ics.uci.edu/~xzhu/face/)
2. [PASCAL face (train/validataion)](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html) and [PASCAL face (test)](http://host.robots.ox.ac.uk:8080/eval/challenges/voc2012/)

### Test SFD detector

The test scripts are located in `$SFD_ROOT/sfd_test_code/`

#### Test SFD on AFW, PASCAL Face

To test with those datasets, use the `test.py` script, for example:

```
python2.7 test.py -d AFW -p ../datasets/AFW/testimages
python2.7 test.py -d PASCAL -p ../datasets/PASCAL/VOCdevkit/VOC2012/JPEGImages
```

Remember to change paths accordingly. Run `test.py -h` to see a list of valid arguments.

The test will output, in 'output' dir for each dataset, which can be used with the `face-eval` tool to plot the precision-recall curves. 

### Running evaluation benchmarks

Download the [EVALUATION TOOLBOX](https://bitbucket.org/marcopede/face-eval) for evaluation. 

#### Plotting Precision-Recall curves for AFW and PASCAL

1. Copy the previously generated file `./output/{AFW,PASCAL}/sfd_{afw,pascal}_dets.txt` into `face-eval/detections/{AFW,PASCAL}` respectively.

2. Run:

    ```
    cd ${face_eval_dir}
    # For AFW
    python2.7 plot_AP.py --dataset AFW
    # For PASCAL
    python2.7 plot_AP.py --dataset PASCAL 
    ```

3. Two PDFs files will be generated, `AFW_final.pdf` and `PASCAL_final.pdf` with their respectives Precision-Recall curves.