# Product Label Recognition with YOLOv8, EasyOCR, and BERT
</br>

## Data

The video used can be downloaded [here](https://youtube.com/shorts/iXvZrCz1Tmw?feature=shared).

## Models

A YOLOv8 pretrained model was used to detect product labels.

A product labels detector was used to detect product labels. The model was trained with YOLOv8 using [this dataset](https://universe.roboflow.com/lamar-university-venef/bottles-label-detection) and following this [step-by-step tutorial on how to train an object detector with on your custom data](https://github.com/sam02425/modeltraining-on-local-machine.git).


## Dependencies

The SORT module needs to be downloaded from [this repository](https://github.com/abewley/sort) as mentioned in the [video](https://youtu.be/fyJB1t0o0ms?t=1120).

To install the required dependencies, run:
```bash
$ pip install -r requirements.txt
