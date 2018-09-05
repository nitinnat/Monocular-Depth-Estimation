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

parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')

parser.add_argument('--encoder',          type=str,   help='type of encoder, vgg or resnet50', default='vgg')
parser.add_argument('--input_dir',       type=str,   help='path to the rgb images', default="./rgb_kitti")
parser.add_argument('--checkpoint_path',  type=str,   help='path to a specific checkpoint to load', required=True)
parser.add_argument('--input_height',     type=int,   help='input height', default=256)
parser.add_argument('--input_width',      type=int,   help='input width', default=512)
parser.add_argument('--output_dir',      type=str,   help='output directory for depth images', default="./depth_kitti_cityscapes")
args = parser.parse_args()

STD_THRESOLD = 5

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

    input_images = []
    #Change this to a loop over all image files in the rgb folder
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
        
    toc = time.time()
    print("It took {} seconds to process the images.".format(toc-tic))
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
    disp_memory = [] #Stores the current and future disparity images for future loading
    for i in tqdm(range(1,len(input_images)-1)):

        #Load 3 images, one previous, one current, one next
        input_image_pair_prev = input_images[i-1]
        input_image_pair_cur = input_images[i]
        input_image_pair_post = input_images[i+1]

        
        
        #Run the model for each image

        #Load from memory if exists
        if disp_memory != []:
            disp_prev, disp_cur = disp_memory[0], disp_memory[1]
        else:
            disp_prev = sess.run(model.disp_left_est[0], feed_dict={left: input_image_pair_prev})
            disp_cur = sess.run(model.disp_left_est[0], feed_dict={left: input_image_pair_cur})
        disp_post = sess.run(model.disp_left_est[0], feed_dict={left: input_image_pair_post})
        
        #Save the current and post images for loading further on
        disp_memory = [disp_cur,disp_post]

        
        #Process the three images
        disp_pp_prev = post_process_disparity(disp_prev.squeeze()).astype(np.float32)
        disp_pp_cur = post_process_disparity(disp_cur.squeeze()).astype(np.float32)
        disp_pp_post = post_process_disparity(disp_post.squeeze()).astype(np.float32)
        
        output_name = os.path.join(args.output_dir,image_files[i])
        #np.save(os.path.join(args.output_dir, "{}_disp.npy".format(output_name)), disp_pp)


        #Invert disparity images
        disp_pp_prev = 1. /disp_pp_prev
        disp_pp_cur = 1. /disp_pp_cur
        disp_pp_post = 1. /disp_pp_post



        disp_to_img_prev = scipy.misc.imresize(disp_pp_prev.squeeze(), [original_height, original_width],mode = 'I')
        disp_to_img_cur = scipy.misc.imresize(disp_pp_cur.squeeze(), [original_height, original_width],mode = 'I')
        disp_to_img_post = scipy.misc.imresize(disp_pp_post.squeeze(), [original_height, original_width],mode = 'I')
        #print(np.amin(disp_to_img),np.amax(disp_to_img))
        #print(disp_to_img)

        """
        Check the standard deviation of the 3 pixels at each location. Set current image's pixel value
        to 0 if variance is higher than a threshold
        """
        #1st step - concatenate the images in the third axis

        disp_to_img_prev = np.reshape(disp_to_img_prev,[original_height,original_width,1])
        disp_to_img_cur = np.reshape(disp_to_img_cur,[original_height,original_width,1])
        disp_to_img_post = np.reshape(disp_to_img_post,[original_height,original_width,1])
        three_channel_image = np.concatenate((disp_to_img_prev,disp_to_img_cur,disp_to_img_post),axis=2)

        #2nd step find the standard deviation at each location
        thresholded_image = np.zeros_like(input_image_pair_cur)

        tup = input_image_pair_cur.shape
        height1, weight1 = tup[0], tup[1]
        for i in range(0,height1):
            for j in range(0,weight1):
                std_temp = np.std(three_channel_image[i,j,:])
                if std_temp > STD_THRESOLD:
                    thresholded_image[i,j] = 0
                else:
                    thresholded_image[i,j] = input_image_pair_cur[i,j]
        print(thresholded_image.shape)
        break   
        disp_to_img = thresholded_image
        ##Thresholding
        #high_values_flags = disp_to_img > 100  # Where values are low
        #disp_to_img[high_values_flags] = 255
        #low_values_flags = disp_to_img < 10
        #disp_to_img[low_values_flags] = 0

        #disp_to_img = 525. / disp_to_img
        disp_to_img = disp_to_img / 255.
        disp_to_img *= 65535
        disp_to_img = disp_to_img.astype(np.uint16)

        


        #cv2.imshow('image',disp_to_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
        #plt.imsave(output_name, disp_to_img, cmap="gray")
        print("Saving in ",output_name)
        cv2.imwrite(output_name,disp_to_img)
        image_saving_time += (time.time() - tic)
    print("It took {} seconds to compute depth maps from the images.".format(process_time))
    print("It took {} seconds to save the images.".format(image_saving_time))
    print("Time to compute depth map per image", process_time/len(input_images))
    print('done!')

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
