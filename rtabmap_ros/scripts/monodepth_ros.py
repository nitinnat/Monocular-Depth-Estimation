# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

from __future__ import absolute_import, division, print_function

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

# Ros Messages
from sensor_msgs.msg import Image

parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')

parser.add_argument('--encoder',          type=str,   help='type of encoder, vgg or resnet50', default='vgg')
parser.add_argument('--input_dir',       type=str,   help='path to the rgb images', default="./rgb_sync/")
parser.add_argument('--checkpoint_path',  type=str,   help='path to a specific checkpoint to load', required=True)
parser.add_argument('--input_height',     type=int,   help='input height', default=256)
parser.add_argument('--input_width',      type=int,   help='input width', default=512)
parser.add_argument('--output_dir',      type=str,   help='output directory for depth images', default="./depth_sync_sample")
args = parser.parse_args()


#For the ros publisher
VERBOSE=False

class image_feature:

    def __init__(self):
        '''Initialize ros publisher, ros subscriber'''
        # topic where we publish
        #Load parameters
        self.params = monodepth_parameters(
            encoder=args.encoder,
            height=args.input_height,
            width=args.input_width,
            batch_size=2,
            num_threads=1,
            num_epochs=1,
            do_stereo=False,
            wrap_mode="border",
            use_deconv=False,
            alpha_image_loss=0,
            disp_gradient_loss_weight=0,
            lr_loss_weight=0,
            full_summary=False)

        #Initialize both the publishers for the rgb and depth images
        self.image_pub_rgb = rospy.Publisher("/camera/rgb/image_rect_color",Image)
        self.image_pub_depth = rospy.Publisher("/camera/depth_registered/image_raw",Image)
        # self.bridge = CvBridge()

        # subscribed Topic
        #self.subscriber = rospy.Subscriber("/camera/image/compressed",
        #    CompressedImage, self.callback,  queue_size = 1)
        #if VERBOSE :
        #    print "subscribed to /camera/image/compressed"

    def callback():
        #Change this to a loop over all image files in the rgb folder
        rgb_filenames = [os.path.join(args.input_dir,f) for f in list(os.walk(args.input_dir))[0][2]]
        depth_filenames = [os.path.join(args.output_dir,f) for f in list(os.walk(args.output_dir))[0][2]]
        print("Reading images and converting them to the correct size.")

        process_time = 0
        image_saving_time = 0
        print("Converting to depth images and storing in the output directory")

        for i in range(len(rgb_filenames)):
            rgb_image = scipy.misc.imread(rgb_filenames[i], mode="RGB")
            original_height, original_width, num_channels = input_image.shape
            input_image = scipy.misc.imresize(input_image, [args.input_height, args.input_width], interp='lanczos')

            depth_image = scipy.misc.imread(depth_filenames[i])
            
            #Form the rgb message
            msg_rgb = Image()
            msg_rgb.header.stamp = rospy.Time.now()
            msg_rgb.format = "png"
            msg_rgb.data = np.array(cv2.imencode('.png', rgb_image)[1]).tostring()
            
            #Form the depth message
            msg_depth = Image()
            msg_depth.header.stamp = rospy.Time.now()
            msg_depth.format = "png"
            msg_depth.data = np.array(cv2.imencode('.png', depth_image)[1]).tostring()
            
            # Publish both images
            print("Publishing rgb image")
            self.image_pub_rgb.publish(msg_rgb)
            print("Publishing depth image")
            self.image_pub_depth.publish(msg_depth)
            

def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def main(args):
    '''Initializes and cleanup ros node'''
    ic = image_feature()
    rospy.init_node('image_feature', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")

if __name__ == '__main__':
    main(sys.argv)
