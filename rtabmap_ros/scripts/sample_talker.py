#!/usr/bin/env python
# license removed for brevity

from __future__ import absolute_import, division, print_function
import rospy
# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from monodepth_model import *
from monodepth_dataloader import *
from average_gradients import *
import sys
# numpy and scipy
from scipy.ndimage import filters

# Ros libraries
import roslib
import rospy
import rospkg
# Ros Messages
from sensor_msgs.msg import Image
import yaml
from sensor_msgs.msg import CameraInfo


#parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')

#parser.add_argument('--encoder',          type=str,   help='type of encoder, vgg or resnet50', default='vgg')
#parser.add_argument('--input_dir',       type=str,   help='path to the rgb images', default="./rgb_sync/")
#parser.add_argument('--checkpoint_path',  type=str,   help='path to a specific checkpoint to load', required=True)
#parser.add_argument('--input_height',     type=int,   help='input height', default=256)
#parser.add_argument('--input_width',      type=int,   help='input width', default=512)
#parser.add_argument('--output_dir',      type=str,   help='output directory for depth images', default="./depth_sync_sample")

#parser.add_argument("--filename", help="Path to yaml file containing " +\
#                                             "camera calibration data",default = 'rgbdatasets.yaml')
#args = parser.parse_args()


input_height = 256
input_width = 512
encoder = "vgg"

rospack = rospkg.RosPack()
path = rospack.get_path('rtabmap_ros')

input_dir = os.path.join(path,"scripts/data/rgb_sync")
output_dir = os.path.join(path,"scripts/data/depth_sync")
yaml_fname = os.path.join(path,"scripts/rgbdatasets2.yaml" )

def yaml_to_CameraInfo(yaml_fname):
    """
    Parse a yaml file containing camera calibration data (as produced by 
    rosrun camera_calibration cameracalibrator.py) into a 
    sensor_msgs/CameraInfo msg.
    
    Parameters
    ----------
    yaml_fname : str
        Path to yaml file containing camera calibration data
    Returns
    -------
    camera_info_msg : sensor_msgs.msg.CameraInfo
        A sensor_msgs.msg.CameraInfo message containing the camera calibration
        data
    """
    # Load data from file
    with open(yaml_fname, "r") as file_handle:
        calib_data = yaml.load(file_handle)
    # Parse
    camera_info_msg = CameraInfo()
    camera_info_msg.width = calib_data["image_width"]
    camera_info_msg.height = calib_data["image_height"]
    camera_info_msg.K = calib_data["camera_matrix"]["data"]
    camera_info_msg.D = calib_data["distortion_coefficients"]["data"]
    camera_info_msg.R = calib_data["rectification_matrix"]["data"]
    camera_info_msg.P = calib_data["projection_matrix"]["data"]
    #camera_info_msg.distortion_model = calib_data["distortion_model"]
    return camera_info_msg

def talker():
    
    #Initialize publishers
    image_pub_rgb = rospy.Publisher("/rgb/image",Image,queue_size = 10)
    image_pub_depth = rospy.Publisher("/depth/image",Image,queue_size = 10)
    camera_info_pub = rospy.Publisher("/rgb/camera_info", CameraInfo, queue_size=10)


    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        print(input_dir)
        print(output_dir)
        rgb_filenames = [os.path.join(input_dir,f) for f in list(os.walk(input_dir))[0][2]]
        depth_filenames = [os.path.join(output_dir,f) for f in list(os.walk(output_dir))[0][2]]
        print("Reading images and converting them to the correct size.")
        process_time = 0
        image_saving_time = 0
        print("Converting to depth images and storing in the output directory")
        camera_info_msg = yaml_to_CameraInfo(yaml_fname)
        try:
            for i in range(len(rgb_filenames)):
                rgb_image = scipy.misc.imread(rgb_filenames[i], mode="RGB")
                original_height, original_width, num_channels = rgb_image.shape
                rgb_image = scipy.misc.imresize(rgb_image, [input_height, input_width], interp='lanczos')
                depth_image = scipy.misc.imread(depth_filenames[i])
                
                #Form the rgb message
                msg_rgb = Image()
                msg_rgb.header.stamp = rospy.Time.now()
                #msg_rgb.format = "png"
                msg_rgb.data = np.array(cv2.imencode('.png', rgb_image)[1]).tostring()
                
                #Form the depth message
                msg_depth = Image()
                msg_depth.header.stamp = rospy.Time.now()
                #msg_depth.format = "png"
                msg_depth.data = np.array(cv2.imencode('.png', depth_image)[1]).tostring()
                
                # Publish both images
                rospy.loginfo("Publishing rgb image")
                image_pub_rgb.publish(msg_rgb)
                rospy.loginfo("Publishing depth image")
                image_pub_depth.publish(msg_depth)
                camera_info_pub.publish(camera_info_msg)
                rospy.loginfo("Publishing camerainfo")
                rate.sleep()
                
        except IndexError,KeyboardInterrupt:
            pass

    
    
    
        
        
        
        

if __name__ == '__main__':
    try:
        talker()
    except KeyboardInterrupt:
        pass