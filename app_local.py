from flask import Flask, request, jsonify, render_template,Response
import os
from asyncio.windows_events import NULL
from flask_cors import CORS, cross_origin
import time

from src.detectron2_webapp.ObjectDetector import DetectorDetectron
from src.detectron2_webapp.com_ineuron_utils.utils import decodeImagedetectron

from src.tf2_webapp.com_in_ineuron_ai_utils.utils import decodeImagetf2
from src.tf2_webapp.detect import PredictorTF2
from src.tf2_webapp.object_detection.utils import label_map_util


from src.yolo_webapp.com_ineuron_apparel.com_ineuron_utils.utils import decodeImageyolov5
from src.yolo_webapp.detect import DetectorYolov5


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


# @cross_origin()
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"


@app.route("/")
def home():
    return render_template("index.html")

#tf2
@app.route("/predict", methods=['POST','GET'])
@cross_origin()
def predictRoute():
    try:
        framework= request.json['framework']
        model = request.json['model']
        image = request.json['image']
        # detection = request.json['detection']
        if framework=="" or model=="" or image=="":
            raise Exception("PLEASE SELECT THE IMAGE, FRAMEWORK OR MODEL")
            
        if framework == 'TF2':
            start = time.time()
            obj_detect = PredictorTF2(model)
            decodeImagetf2(image, clApp.filename)
            result = obj_detect.run_inference()
            end = time.time()
            time_taken =end-start
            print("Time Taken to execute the model",time_taken)

        elif framework == 'Detectron2':
            start = time.time()
            obj_detect = DetectorDetectron(clApp.filename, model)
            decodeImagedetectron(image, clApp.filename)
            result = obj_detect.inference(clApp.filename)
            end = time.time()
            time_taken =end-start
            print("Time Taken to execute the model",time_taken)
        
        elif framework == 'YOLOv5':
            start = time.time()
            obj_detect = DetectorYolov5(clApp.filename,model)
            decodeImageyolov5(image, clApp.filename)
            result = obj_detect.detect_action()
            end = time.time()
            time_taken =end-start
            print("Time Taken to execute the model",time_taken)

    except ValueError as val:
        print(val)
        return Response("Value not found inside  json data")
    except KeyError:
        return Response("Key value error incorrect key passed")
    except Exception as e:
        print(e)
        result = "Invalid input"

    return jsonify(result)


if __name__ == "__main__":
    clApp = ClientApp()
    port = 9502
    app.run(host='127.0.0.1', port=port)