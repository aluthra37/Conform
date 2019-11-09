import argparse
import logging
import time
import math

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    
    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    parser.add_argument('--exercise', type=str,
                        help='Exercise you are going to complete')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)

    refHuman = None
    count = 0
    elbowRX = 0
    while True:
        ret_val, image = cam.read()
        if count < 7:
            cv2.putText(image,
                    "Im: %d" % count,
                    (40, 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)        
        logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        
        differenceShoulderElbowXR = 0
        differenceShoulderElbowYR = 0
        differenceElbowWristXR = 0
        differenceElbowWristYR = 0
        angleSER = 0
        angleEWR = 0
        if len(humans) > 0:
            if count == 7:
                refHuman = humans[0]
                
            if args.exercise == "Curls":
                if elbowRX is not None and humans[0].body_parts.get(3) is not None:
                    difference = elbowRX - humans[0].body_parts.get(3).x
                    print("%f d %f elbocurrent %f elbowpast " % (difference, humans[0].body_parts.get(3).x,  elbowRX))
                    elbowRX = humans[0].body_parts[3].x
                    
                if humans[0].body_parts.get(2) is not None and humans[0].body_parts.get(3) is not None and humans[0].body_parts.get(4) is not None:
                    differenceShoulderElbowXR = humans[0].body_parts[2].x - humans[0].body_parts[3].x 
                    differenceShoulderElbowYR = humans[0].body_parts[2].y - humans[0].body_parts[3].y 
                    angleSER = math.atan2(differenceShoulderElbowXR, differenceShoulderElbowYR)
                    differenceElbowWristXR = humans[0].body_parts[3].y - humans[0].body_parts[4].y 
                    differenceElbowWristYR =  humans[0].body_parts[3].x - humans[0].body_parts[4].x 
                    angleEWR = math.atan2(differenceElbowWristXR, differenceElbowWristYR)
                    #if count > 7:
                    #    refdifferenceShoulderElbowXR = refHuman.body_parts[2].x - refHuman.body_parts[3].x 
                    #    refdifferenceShoulderElbowYR = refHuman.body_parts[2].y - refHuman.body_parts[3].y 
                    #    refangleSER = math.atan2(refdifferenceShoulderElbowXR, refdifferenceShoulderElbowYR)
                    #    refdifferenceElbowWristXR = refHuman.body_parts[3].y - refHuman.body_parts[4].y 
                    #    refdifferenceElbowWristYR =  refHuman.body_parts[3].x - refHuman.body_parts[4].x 
                    #    refangleEWR = math.atan2(refdifferenceElbowWristXR, refdifferenceElbowWristYR)

                    
                
        logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
                
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')
        count+=1
        
        


    cv2.destroyAllWindows()
