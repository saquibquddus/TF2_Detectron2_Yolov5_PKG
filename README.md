# Package for TF2 , Detectron2 and Yolov5 Inferencing.
This packge can be useful to do inferencing on TF2, Detectron2 and yolov5.
All the supported libraries are already added in the package.<br>
User can select five pretrained model from each framework to do prediction.<br>
## TF2
1. ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8<br>
2. ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8<br>
3. efficientdet_d4_coco17_tpu-32<br>
4. faster_rcnn_resnet101_v1_1024x1024_coco17_tpu-8<br>
5. efficientdet_d7_coco17_tpu-32<br>

## Detectron2
1. faster_rcnn_R_50_FPN_1x<br>
2. faster_rcnn_R_50_FPN_3x<br>
3. faster_rcnn_R_101_FPN_3x<br>
4. retinanet_R_50_FPN_1x<br>
5. faster_rcnn_R_50_C4_1x<br>

## Yolov5
1. yolov5n
2. yolov5s
3. yolov5m
4. yolov5l
5. yolov5x



## How to Run?
Step 1: 
```python
pip install TDY-PKG
```
Step 2:
Copy the templates, utils folder and app.py file in any directory from the [Github link](https://github.com/saquibquddus/TF2_Detectron2_Yolov5_PKG)

Step 3:
Run app.py file.
```python
python app.py
```

Step 4:
Go to : http://127.0.0.1:9502/

Step 5:
Upload Image and select Framework and Model. 
![UI Image](https://github.com/saquibquddus/TF2_Detectron2_Yolov5_PKG/blob/master/UI.JPG?raw=true)

Step 6:
Click On Predict.(The model will get download in runtime to do prediction)
![Predicted UI image](https://github.com/saquibquddus/TF2_Detectron2_Yolov5_PKG/blob/master/Predicted_UI.JPG?raw=true)



