import os
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS, Flag
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession



flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416','path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')


flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show image output')




def process_frame(imagePath,saved_model_loaded,infer,STRIDES, ANCHORS, NUM_CLASS, XYSCALE,input_size):

    

    frame = cv2.imread(imagePath)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)

    frame_size = frame.shape[:2]
    image_data = cv2.resize(frame, (input_size, input_size))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    
    
    batch_data = tf.constant(image_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=FLAGS.iou,
        score_threshold=FLAGS.score
    )
    
    # convert data to numpy arrays and slice out unused elements
    num_objects = valid_detections.numpy()[0]
    bboxes = boxes.numpy()[0]
    bboxes = bboxes[0:int(num_objects)]
    scores = scores.numpy()[0]
    scores = scores[0:int(num_objects)]
    classes = classes.numpy()[0]
    classes = classes[0:int(num_objects)]

    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
    original_h, original_w, _ = frame.shape
    bboxes = utils.format_boxes(bboxes, original_h, original_w)
    
    # store all predictions in one parameter for simplicity when calling functions
    pred_bbox = [bboxes, scores, classes, num_objects]
    
    # read in all class names from config
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)
    
    # by default allow all classes in .names file
    allowed_classes = list(class_names.values())        
    
    # custom allowed classes (uncomment line below to customize tracker for only people)
    #allowed_classes = ['person']
    # loop through objects and use class index to get class name, allow only classes in allowed_classes list
    names = []
    deleted_indx = []
    for i in range(num_objects):
        class_indx = int(classes[i])
        class_name = class_names[class_indx]
        if class_name not in allowed_classes:
            deleted_indx.append(i)
        else:
            names.append(class_name)
    names = np.array(names)
    count = len(names)
    
    # delete detections that are not in allowed_classes
    bboxes = np.delete(bboxes, deleted_indx, axis=0)
    scores = np.delete(scores, deleted_indx, axis=0)

    
    
    

    for index in range(len(classes)):
        class_name=class_names[classes[index]]
        bbox=bboxes[index]
        col=int(bbox[0])
        row=int(bbox[1])
        width=col+int(bbox[2])
        heigth=row+ int(bbox[3])
        cv2.rectangle(frame, (col,row), (width, heigth), (255,0,255), 2)
     
        cv2.putText(frame, class_name,(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
    
    result = np.asarray(frame)
    result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    if not FLAGS.dont_show:
        cv2.imshow("Output Video", result)        
    
    if cv2.waitKey(0) & 0xFF == ord('q'): 
        cv2.destroyAllWindows()
        return
        
def configure(_argv):

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    

   
    # load standard tensorflow saved model
    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    return saved_model_loaded,infer,input_size,STRIDES, ANCHORS, NUM_CLASS, XYSCALE
    

def main(_argv):
    """
    python object_tracker.py 
    --weights ./checkpoints/yolov4-tiny-416
    --model yolov4 
    --video ./data/video/test.mp4 
    --output ./outputs/tiny.avi 
    --tiny
    """
    FLAGS.weights= "./checkpoints/yolov4-tiny-416"
    FLAGS.model="yolov4"
    FLAGS.tiny =True
    

    saved_model_loaded,infer,input_size,STRIDES, ANCHORS, NUM_CLASS, XYSCALE =configure(_argv)
    imagePath="./data/testImage.jpg"
    process_frame(imagePath,saved_model_loaded,infer,STRIDES, ANCHORS, NUM_CLASS, XYSCALE,input_size)
    print("Fine")

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass