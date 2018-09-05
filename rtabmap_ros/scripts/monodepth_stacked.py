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
#np.set_printoptions(threshold='nan')
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

parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')

parser.add_argument('--encoder',          type=str,   help='type of encoder, vgg or resnet50', default='vgg')
parser.add_argument('--input_dir',       type=str,   help='path to the rgb images', default="./rgb_kitti")
parser.add_argument('--ground_dir',       type=str,   help='path to the depth ground truth images', default="./depth_kitti_ground_truth")
parser.add_argument('--checkpoint_path',  type=str,   help='path to a specific checkpoint to load', required=True)
parser.add_argument('--input_height',     type=int,   help='input height', default=256)
parser.add_argument('--input_width',      type=int,   help='input width', default=512)
parser.add_argument('--output_dir',      type=str,   help='output directory for depth images', default="./depth_kitti_cityscapes_scaled")
args = parser.parse_args()

std_threshold = 5

def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def test_simple(params):
    """Test function."""

    left  = tf.placeholder(tf.float32, [2, args.input_height, args.input_width, 3])
    model = MonodepthModel(params, "test", left, None)


    #----------------------------------------------------
    #Load rgb images and convert into the required format
    #for monodepth
    #----------------------------------------------------
    input_images = []
    image_files = [f for f in list(os.walk(args.input_dir))[0][2]]
    image_paths = [os.path.join(args.input_dir,f) for f in image_files]
    print("Reading images and converting them to the correct size.")

    tic = time.time()
    for filepath in tqdm(image_paths):
        input_image = scipy.misc.imread(filepath, mode="RGB")
        original_height, original_width, num_channels = input_image.shape
        input_image = scipy.misc.imresize(input_image, [args.input_height, args.input_width], interp='lanczos')
        input_image = input_image.astype(np.float32) / 255
        input_image_pair = np.stack((input_image, np.fliplr(input_image)), 0)
        input_images.append(input_image_pair)
    #----------------------------------------------------
    #Load ground truth images to help find scaling factor
    #----------------------------------------------------
    ground_truth_image_files = [f for f in list(os.walk(args.ground_dir))[0][2]]
    ground_truth_image_paths = [os.path.join(args.ground_dir,f) for f in ground_truth_image_files]
    ground_images = []
    for filepath in ground_truth_image_paths:
        ground_input_image = scipy.misc.imread(filepath)
        ground_images.append(ground_input_image)
    print("Shape of ground truth images is ", str(ground_input_image.shape))
    

    #----------------------------------------------------
    #Restore tensorflow checkpoint
    #----------------------------------------------------
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
    
    overall_scale_map = np.zeros_like(ground_images[0]).astype(float)
    disparity_images = []
    #----------------------------------------------------
    #Convert to depth images using the monodepth model
    #----------------------------------------------------
    print("Converting to depth images and storing in the output directory")
    for i,input_image_pair in tqdm(enumerate(input_images)):

        tic = time.time()
        disp = sess.run(model.disp_left_est[0], feed_dict={left: input_image_pair})
        disp_pp = post_process_disparity(disp.squeeze()).astype(np.float32)
        toc = time.time()
        
        
        #np.save(os.path.join(args.output_dir, "{}_disp.npy".format(output_name)), disp_pp)
        tic = time.time()
        #Invert and scale
        #disp_pp = 0.54 * 721 / (1242 * disp_pp)
        disp_pp = 1. /disp_pp
        disp_to_img = scipy.misc.imresize(disp_pp.squeeze(), [original_height, original_width],mode = 'I')
        print("Squeezing image to ",str(disp_to_img.shape))


        #print(np.amin(disp_to_img),np.amax(disp_to_img))
        #print(disp_to_img)

        ##Thresholding
        #high_values_flags = disp_to_img > 100  # Where values are low
        #disp_to_img[high_values_flags] = 255
        #low_values_flags = disp_to_img < 5
        #disp_to_img[low_values_flags] = 0

        #disp_to_img = 525. / disp_to_img
        disp_to_img = disp_to_img / 255.
        disp_to_img *= 65535
        disp_to_img = disp_to_img.astype(np.uint16)

        #----------------------------------------------------
        #Check obtained depth image with original ground truth
        #image to obtain scaling factor
        #----------------------------------------------------
        print("\n")
        print("Ground truth shape is {}, and datatype is {}".format(str(ground_images[i].shape), ground_images[i].dtype))
        print("Monodepth depth image shape is {}, and datatype is {}".format(str(disp_to_img.shape), disp_to_img.dtype))
        #print(ground_images[i])
        #print(disp_to_img)
        print("\n")
        print("Min value of ground truth: {} and max value of ground truth: {}".format(np.amin(ground_images[i]),np.amax(ground_images[i])))
        print("Min value of monodepth image: {} and max value of monodepth image: {}".format(np.amin(disp_to_img[i]),np.amax(disp_to_img[i])))
        print("\n")
        scale_map = np.divide(ground_images[i].astype(float)+1,disp_to_img.astype(float) + 1)
        print("\n")
        print("Scale map is", scale_map)
        print("Min value of scale_map: {} and max value of scale_map: {}".format(np.amin(scale_map),np.amax(scale_map)))

        ##Add scale map to overall and compute the sum
        overall_scale_map += scale_map
        #print("overall_scale_map is")
        #print(overall_scale_map)
        

        process_time += (toc - tic)

        disparity_images.append(disp_to_img)
        #cv2.imshow('image',disp_to_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
        #plt.imsave(output_name, disp_to_img, cmap="gray")
    
    #----------------------------------------------------
    #Perform scaling with average of overall scale map
    #----------------------------------------------------
    average_scale_map = overall_scale_map/len(ground_images)
    print("Average scale map")
    print(average_scale_map)
    depth_images = [np.multiply(d,average_scale_map) for d in disparity_images]
    depth_images = [d.astype(np.uint16) for d in depth_images]
    
    #----------------------------------------------------
    #Save images into the output directory
    #----------------------------------------------------
    print("Saving depth images...")
    for i in range(len(image_files)):
        output_name = os.path.join(args.output_dir,image_files[i])
        print("Saving in ",output_name)
        cv2.imwrite(output_name,depth_images[i])
    

def main(_):

    params = monodepth_parameters(
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

    test_simple(params)

if __name__ == '__main__':
    tf.app.run()
