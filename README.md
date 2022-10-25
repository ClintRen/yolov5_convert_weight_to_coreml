## Introduction

A script convert .pt file of [yolov5](https://github.com/ultralytics/yolov5) weight to .coreml file of apple app platform. It is easy to add decode layer and NMS to the model.

- **easy to use**

  You just setup environment and run the script with your model parameters, then you can get some coreml weight files.

- **Support of quantization**

  The Fp16 and Int8 quantization is supported. In addition, MAC is not required by quantization.

## Installation

The [yolov5](https://github.com/ultralytics/yolov5) is required to run this script. You should follow the instructions to deploy the yolov5 runtime environment ([yolov5 v6.2](https://github.com/ultralytics/yolov5/tree/v6.2) has been tested in python3.8). 

Also, the coremltools is required.

```shell
pip install coremltools==6.0
```

## Getting Started

There are explanations for converting model parameters in [convert.py](convert.py).

It is worth noting that the parameter of yolov5_repo in [convert.py](convert.py) is required. It is a path of the yolov5 repo.

## For Example
In this repo, you can run the following command to get some coreml weight files of yolov5s.

```shell
python convert.py --yolov5-repo /path/to/yolov5 --weight yolov5s.pt --img-size 640 --quantize
```

yolov5s.mlmodel, yolov5s_FP16.mlmodel and yolov5s_Int8.mlmodel will be generated in the weight directory.
## Detection
The pictures below are detected by the yolov5s_Int8.mlmodel in preview on MAC.

Please give a star If it helps you, thanks. 

![markdown picture](pictures/zidane_res_Int8.png)

![markdown picture](pictures/bus_res_Int8.png)

