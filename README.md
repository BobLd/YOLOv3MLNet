# YOLO v3 in MLNet
Use the YOLO v3 algorithms for object detection in C# using ML.Net. We start with a Torch model, then converting it to ONNX format and use it in ML.Net.

This is a case study on a document layout YOLO trained model. The model can be found in the following Medium article: [Object Detection — Document Layout Analysis Using Monk AI](https://medium.com/towards-artificial-intelligence/object-detection-document-layout-analysis-using-monk-object-detection-toolkit-6c57200bde5).

**Another case study, based on [this](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov3) YOLO v3 model is available [here](https://github.com/BobLd/YOLOv3MLNet/tree/master/YOLOV3MLNetSO).**

## Main differences
- The ONNX conversion removes 1 feature which is the *objectness score*, p<sub>c</sub>. The original model has (5 + classes) features for each bounding box, the ONNX model has (4 + classes) features per bounding box. We will use the class probability as a proxy for the *objectness score* when performing the Non-maximum Suppression (NMS) step. This is a known issue, more info [here](https://github.com/ultralytics/yolov3/issues/750).
- Image resizing is not optimised, and will always yield 416x416 size image. This is not the case in the original model (see this issue: [RECTANGULAR INFERENCE](https://github.com/ultralytics/yolov3/issues/232)).

# Train and export to ONNX in Python
This is based on this article [Object Detection — Document Layout Analysis Using Monk AI](https://medium.com/towards-artificial-intelligence/object-detection-document-layout-analysis-using-monk-object-detection-toolkit-6c57200bde5).

## Load the model
```python
import os
import sys
from IPython.display import Image
sys.path.append("../Monk_Object_Detection/7_yolov3/lib")
from infer_detector import Infer

gtf = Infer()

f = open("dla_yolov3/classes.txt")
class_list = f.readlines()
f.close()

model_name = "yolov3"
weights = "dla_yolov3/dla_yolov3.pt"
gtf.Model(model_name, class_list, weights, use_gpu=False, input_size=(416, 416))
```

## Test the model
```python
img_path = "test_square.jpg"
gtf.Predict(img_path, conf_thres=0.2, iou_thres=0.5)
Image(filename='output/test_square.jpg')
```

## Export the model
You need to set `ONNX_EXPORT = True` in `...\Monk_Object_Detection\7_yolov3\lib\models.py` before loading the model.

We name the input layer `image` and the 2 ouput layers `classes`, `bboxes`. This is not needed but helps the clarity.

```python
import torch
import torchvision.models as models

dummy_input = torch.randn(1, 3, 416, 416) # Create the right input shape (e.g. for an image)
dummy_input = torch.nn.Sigmoid()(dummy_input) # limit between 0 and 1 (superfluous?)
torch.onnx.export(gtf.system_dict["local"]["model"],
                  dummy_input, 
                  "dla_yolov3.onnx",
                  input_names=["image"],
                  output_names=["classes", "bboxes"],
                  opset_version=9)
```

# Check exported model with Netron
The ONNX model can be viewed in [Netron](https://www.electronjs.org/apps/netron). Our model looks like this:
![neutron](https://github.com/BobLd/YOLOv3MLNet/blob/master/netron.png)

- The input layer size is [1 x 3 x 416 x 416]. This corresponds to 1 batch size x 3 colors x 416 pixels height x 416 pixel width (more info about fixed batch size [here](https://github.com/ultralytics/yolov3/issues/1030)).

As per this [article](https://medium.com/analytics-vidhya/yolo-v3-theory-explained-33100f6d193):
> For an image of size 416 x 416, YOLO predicts ((52 x 52) + (26 x 26) + 13 x 13)) x 3 = 10,647 bounding boxes.
- The `bboxes` output layer is of size [10,647 x 4]. This corresponds to 10,647 bounding boxes x 4 bounding box coordinates (x, y, h, w).
- The `classes` output layer is of size [10,647 x 18]. This corresponds to 10,647 bounding boxes x 18 classes (this model has only 18 classes).

Hence, each bounding box has (4 + classes) = 22 features. The total number of prediction in this model is 22 x 10,647.

**NB**: The ONNX conversion removes 1 feature which is the *objectness score*, p<sub>c</sub>. The original model has (5 + classes) features for each bounding box. We will use the class probability as a proxy for the *objectness score*.

![medium-explanation](https://miro.medium.com/max/700/1*6KLkWAWCINb8kVNuPRaDMQ.png)

More information can be found in this article: [YOLO v3 theory explained](https://medium.com/analytics-vidhya/yolo-v3-theory-explained-33100f6d193)

# Load model in C#

# Predict in C#
![output](YOLOv3MLNet/Assets/Output/PMC5055614_00001._processed.jpg)

# Resources
- https://medium.com/towards-artificial-intelligence/object-detection-document-layout-analysis-using-monk-object-detection-toolkit-6c57200bde5
- https://medium.com/analytics-vidhya/yolo-v3-theory-explained-33100f6d193
- https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c
- https://michhar.github.io/convert-pytorch-onnx/
