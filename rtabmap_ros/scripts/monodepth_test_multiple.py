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
        self.image_pub_rgb = rospy.Publisher("/rgb/image/",Image)
        self.image_pub_depth = rospy.Publisher("/depth/image/",Image)
        # self.bridge = CvBridge()

        # subscribed Topic
        #self.subscriber = rospy.Subscriber("/camera/image/compressed",
        #    CompressedImage, self.callback,  queue_size = 1)
        #if VERBOSE :
        #    print "subscribed to /camera/image/compressed"

    def callback():


        """Test function."""

        left  = tf.placeholder(tf.float32, [2, args.input_height, args.input_width, 3])
        model = MonodepthModel(self.params, "test", left, None)

        input_images = []
        #Change this to a loop over all image files in the rgb folder
        print(args.input_dir)
        image_files = [f for f in list(os.walk(args.input_dir))[0][2]]
        image_paths = [os.path.join(args.input_dir,f) for f in image_files]
        print("Reading images and converting them to the correct size.")


        # SESSION
        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)

        # SAVER
        train_saver = tf.train.Saver()

        # INIT
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        # RESTORE
        restore_path = args.checkpoint_path.split(".")[0]
        train_saver.restore(sess, restore_path)
        process_time = 0
        image_saving_time = 0
        print("Converting to depth images and storing in the output directory")

        for i, filepath in tqdm(enumerate(image_paths)):

            input_image = scipy.misc.imread(filepath, mode="RGB")
            original_height, original_width, num_channels = input_image.shape
            input_image = scipy.misc.imresize(input_image, [args.input_height, args.input_width], interp='lanczos')

            input_image2 = input_image.astype(np.float32) / 255
            input_image_pair = np.stack((input_image2, np.fliplr(input_image2)), 0)
            
            tic = time.time()
            disp = sess.run(model.disp_left_est[0], feed_dict={left: input_image_pair})
            disp_pp = post_process_disparity(disp.squeeze()).astype(np.float32)

            disp_pp = 1. / disp_pp
            
            ##disp_to_img = 65535 - disp_to_img
            toc = time.time()
            
            output_name = os.path.join(args.output_dir,image_files[i])
            #np.save(os.path.join(args.output_dir, "{}_disp.npy".format(output_name)), disp_pp)
            tic = time.time()

            print(disp_pp)
            disp_to_img = scipy.misc.imresize(disp_pp.squeeze(), [original_height, original_width])
            #print(disp_to_img)
            disp_to_img = disp_to_img / 255.
            #print(disp_to_img)

            disp_to_img *= 65535

            #disp_to_img = disp_to_img * 65535.0 * 0.8347744631791324
            #print(disp_to_img)
            

            #disp_to_img = 65535.0 - disp_to_img
            #print(disp_to_img)
            
            disp_to_img = disp_to_img.astype(np.uint16)
            print(disp_to_img)
            print(disp_to_img.dtype)

            

            ###CONVERSION TO DEPTH FROM DISPARITY HERE
            #disp_to_img = 1. / disp_to_img  #Invert image
            #
            #print(disp_pp)
            #
            #print(disp_pp)
            #disp_to_img *= 255
            #print(disp_to_img)
            #disp_to_img = disp_to_img.astype(np.uint16)
            #print(disp_to_img)
            
            #disp_to_img = disp_to_img.astype(np.uint16)
            #print(disp_to_img)
            #disp_to_img = disp_to_img.astype(np.uint16)
            #COnvert image to depth from disparity
            ##disp_to_img = 1. / disp_to_img
            #print(disp_to_img.dtype,disp_to_img.shape)
            process_time += (toc - tic)
            #plt.imsave(output_name, disp_to_img, cmap="gray")

            #Form the rgb message
            msg_rgb = Image()
            msg_rgb.header.stamp = rospy.Time.now()
            msg_rgb.format = "png"
            msg_rgb.data = np.array(cv2.imencode('.png', input_image)[1]).tostring()
            
            #Form the depth message
            msg_depth = Image()
            msg_depth.header.stamp = rospy.Time.now()
            msg_depth.format = "png"
            msg_depth.data = np.array(cv2.imencode('.png', disp_to_img)[1]).tostring()
            
            # Publish both images
            print("Publishing rgb image")
            self.image_pub_rgb.publish(msg_rgb)
            print("Publishing depth image")
            self.image_pub_depth.publish(msg_depth)
            #cv2.imwrite(output_name,disp_to_img)
            
    #print("It took {} seconds to compute depth maps from the images.".format(process_time))
    #print("It took {} seconds to save the images.".format(image_saving_time))
    #print("Time to compute depth map per image", process_time/len(input_images))
    #print('done!')


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
