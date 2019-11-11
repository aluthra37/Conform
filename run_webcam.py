import argparse
import logging
import time
import math
import sys
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
    #TODO change this back
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)
    #setting this value drastically increases frame rate
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')

    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    #mobilenet_thin results in best framerate
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    
    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    #necessary for specifying which exercise is being completed.
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
    #determine if a good baseline was found
    found = False
    #reference human to compare against
    refHuman = None
    #current framecount
    count = 0
    elbowRX = 0
    while True:
        ret_val, image = cam.read()
        cv2.Flip(image, flipMode=-1)
        #place the current frame onto the image
        cv2.putText(image,
                    "Im: %d" % count,
                    (30, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)      
        logger.debug('image process+')
        #get the humans
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        #difference between shoulder and elbow X coordinates on the right side
        differenceShoulderElbowXR = 0
        #difference between should and elbow Y coordinates on the right side
        differenceShoulderElbowYR = 0
        #difference between elbow and wrist x coordinates on the right side
        differenceElbowWristXR = 0
        #difference between elbow and wrist y coordinates on the right sdie
        differenceElbowWristYR = 0
        #angle between Shoulder Elbow on right side
        angleSER = 0
        #angle between Elbow and wrist on the right side
        angleEWR = 0
        #make sure we have at least one valid registered human
        if len(humans) > 0:
            #Exercise selection if statements
            if args.exercise == "Curls":
                #print all releveant values, mostly just for debugging
                print("%d %s %s %s %s %s %s" % (count ,humans[0].body_parts.get(2), humans[0].body_parts.get(3), humans[0].body_parts.get(4) ,humans[0].body_parts.get(5), humans[0].body_parts.get(6), humans[0].body_parts.get(7)))
                #Set the reference human, give the user at least 10 frames to be established, and then make sure it has notbeen found previously
                if found is False and count > 10:
                    #check all releveant bodyparts regarding the curl (Shoulder, elbow, wrist both sides)
                    if humans[0].body_parts.get(2) is not None and humans[0].body_parts.get(3) is not None and humans[0].body_parts.get(4) is not None and humans[0].body_parts.get(5) is not None and humans[0].body_parts.get(6) is not None and humans[0].body_parts.get(7) is not None:
                        #Set the reference human value and make sure we do not execute this again
                        refHuman = humans[0]
                        found = True
                #begin the form comparison against the reference human (and maybe the previous form)
                elif found is True:
                    #commented out code regarding the difference between the previous values
                    #if elbowRX is not None and humans[0].body_parts.get(3) is not None:
                    #    difference = elbowRX - humans[0].body_parts.get(3).x
                    #    print("%f d %f elbocurrent %f elbowpast " % (difference, humans[0].body_parts.get(3).x,  elbowRX))
                    #    elbowRX = humans[0].body_parts[3].x
                    # Make sure the current human has all of the required comparison points
                    if humans[0].body_parts.get(2) is not None and humans[0].body_parts.get(3) is not None and humans[0].body_parts.get(4) is not None:
                        #finding the angle between the body parts
                        differenceShoulderElbowXR = humans[0].body_parts[2].x - humans[0].body_parts[3].x 
                        differenceShoulderElbowYR = humans[0].body_parts[2].y - humans[0].body_parts[3].y 
                        angleSER = math.atan2(differenceShoulderElbowXR, differenceShoulderElbowYR)
                        differenceElbowWristXR = humans[0].body_parts[3].y - humans[0].body_parts[4].y 
                        differenceElbowWristYR =  humans[0].body_parts[3].x - humans[0].body_parts[4].x 
                        angleEWR = math.atan2(differenceElbowWristXR, differenceElbowWristYR)
                        #finding the angle of the reference human
                        refdifferenceShoulderElbowXR = refHuman.body_parts[2].x - refHuman.body_parts[3].x 
                        refdifferenceShoulderElbowYR = refHuman.body_parts[2].y - refHuman.body_parts[3].y 
                        refangleSER = math.atan2(refdifferenceShoulderElbowXR, refdifferenceShoulderElbowYR)
                        refdifferenceElbowWristXR = refHuman.body_parts[3].y - refHuman.body_parts[4].y 
                        refdifferenceElbowWristYR =  refHuman.body_parts[3].x - refHuman.body_parts[4].x 
                        refangleEWR = math.atan2(refdifferenceElbowWristXR, refdifferenceElbowWristYR)
                        
        logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        logger.debug('show+')
        #placing the lines in the middle, remove when done
        cv2.line(image, (int(image.shape[1]/2), 0), (int(image.shape[1]/2), int(image.shape[0])), (255, 0, 0), 20, 20)
        cv2.line(image, (0, int(image.shape[0]/2)), (int(image.shape[1]), int(image.shape[0]/2)), (255, 0, 0), 20, 20)
        #Place the FPS tags in the top left corner, removeable when done
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        #show the final image    
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')
        #increment the framecount
        count+=1
        



    cv2.destroyAllWindows()
