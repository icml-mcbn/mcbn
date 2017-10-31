import numpy as np
import os.path
import scipy
import argparse
import scipy.io as sio
import matplotlib
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import cv2
import sys

# Make sure that caffe is on the python path:
caffe_root = '/SegNet/caffe-segnet/'            # Change this to the absolute directoy to SegNet Caffe
sys.path.insert(0, caffe_root + 'python')
trials = 5 # number of sampling for calculating the mean and variance

import caffe

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--colours', type=str, required=True)
parser.add_argument('--testdata', type=str, required=True)
parser.add_argument('--traindata', type=str, required=True)
args = parser.parse_args()

caffe.set_mode_gpu()

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)

input_shape = net.blobs['data'].data.shape

label_colours = cv2.imread(args.colours).astype(np.uint8)


def read_input_image(input_image_file):
    input_image_raw = caffe.io.load_image(input_image_file)
    input_image = caffe.io.resize_image(input_image_raw, (input_shape[2],input_shape[3]))
    input_image = input_image*255
    input_image = input_image.transpose((2,0,1))
    input_image = input_image[(2,1,0),:,:]
    input_image = np.asarray([input_image])
    return input_image, input_image_raw

train_image_files = []
test_image_files = []
ground_truth_files = []

# read through the training data for constructing the random batches
with open(args.traindata) as f:
    for line in f:
        input_image_file, ground_truth_file = line.split()
        train_image_files.append(input_image_file)

train_images_shape = list(input_shape)
train_images_shape[0] = len(train_image_files)
train_input_images = np.zeros(train_images_shape)

for i in range(0, len(train_image_files)):
    input_image, _ = read_input_image(train_image_files[i])
    train_input_images[i, :, :, :] = input_image

# read through the test data for evaluating MCBN
with open(args.testdata) as f:
    for line in f:
        input_image_file, ground_truth_file = line.split()
        test_image_files.append(input_image_file)
        ground_truth_files.append(ground_truth_file)

for i in range(len(test_image_files)):

    test_input_image, test_input_image_raw = read_input_image(test_image_files[i])
    ground_truth_file = ground_truth_files[i]
    ground_truth = cv2.imread(ground_truth_file, 0)

    print('image #'+str(i)+': '+test_image_files[i])

    predicteds = []
    for t in range(trials):
        ids = np.random.choice(train_input_images.shape[0], input_shape[0]-1)
        other_train_input_images = train_input_images[ids, :, :, :]

        input_batch = np.concatenate([test_input_image, other_train_input_images])
        print input_batch.shape

        out = net.forward_all(data=input_batch)
        predicted = net.blobs['prob'].data[0:1,:,:,:]
        predicteds = predicted if predicteds == [] else np.concatenate([predicteds,predicted])
        

    print predicteds.shape

    output = np.mean(predicteds,axis=0)
    uncertainty = np.var(predicteds,axis=0)
    ind = np.argmax(output, axis=0)

    segmentation_ind_3ch = np.resize(ind,(3,input_shape[2],input_shape[3]))
    segmentation_ind_3ch = segmentation_ind_3ch.transpose(1,2,0).astype(np.uint8)
    segmentation_rgb = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)

    gt_ind_3ch = np.resize(ground_truth,(3,input_shape[2],input_shape[3]))
    gt_ind_3ch = gt_ind_3ch.transpose(1,2,0).astype(np.uint8)
    gt_rgb = np.zeros(gt_ind_3ch.shape, dtype=np.uint8)

    cv2.LUT(segmentation_ind_3ch,label_colours,segmentation_rgb)
    cv2.LUT(gt_ind_3ch,label_colours,gt_rgb)
    
    uncertainty = np.transpose(uncertainty, (1,2,0))

    average_unc = np.mean(uncertainty,axis=2)
    min_average_unc = np.min(average_unc)
    max_average_unc = np.max(average_unc)
    max_unc = np.max(uncertainty)

    plt.imshow(test_input_image_raw,vmin=0, vmax=255)
    plt.figure()
    plt.imshow(segmentation_rgb,vmin=0, vmax=255)
    plt.figure()
    plt.imshow(gt_rgb,vmin=0, vmax=255)
    plt.set_cmap('bone_r')
    plt.figure()
    plt.imshow(average_unc,vmin=0, vmax=max_average_unc)
    plt.show()

    # uncomment to save results
    # scipy.misc.toimage(segmentation_rgb, cmin=0.0, cmax=255.0).save('images/'+str(i)+'_segnet_segmentation_mcbn'+str(trials)+'.png')
    # cm = matplotlib.pyplot.get_cmap('bone_r') 
    # matplotlib.image.imsave('images/'+str(i)+'_segnet_uncertainty'+str(trials)+'.png',average_unc,cmap=cm, vmin=0, vmax=max_average_unc)

    print 'Processed: ', input_image_file

print 'Success!'

