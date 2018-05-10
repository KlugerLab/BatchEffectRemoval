'''
Created on Dec 5, 2016

@author: urishaham
'''

import os
import numpy as np
import matplotlib
from Calibration_Util import FileIO as io
from Calibration_Util import DataHandler as dh
import argparse
#detect display
havedisplay = "DISPLAY" in os.environ
# havedisplay = False
#if we have a display use a plotting backend
if havedisplay:
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')

parse = argparse.ArgumentParser(description='command line interface for mmd net')
parse.add_argument(
    '--source_dir',
    type=str,
    default=os.path.join(io.DeepLearningRoot(),'Data/Person1Day1_3month.csv'),
    help='Directory of the source dataset')
parse.add_argument(
    '--target_dir',
    type=str,
    default=os.path.join(io.DeepLearningRoot(),'Data/Person1Day2_3month.csv'),
    help='Directory of the source dataset')
parse.add_argument(
    '--epochs',
    type=int,
    default=500,
    help='Number of epochs to run for')
parse.add_argument(
    '--denoise',
    type=bool,
    default=False,
    help='Whether or not to denoise the datasets')
parse.add_argument(
    '--ae_latent_dim',
    type=int,
    default=25,
    help='Size of autoencoder latent layer, should denoising be necessary')
parse.add_argument(
    '--layer_sizes',
    type=str,
    default='25_25_25',
    help='Widths of MMDNet resnet blocks, separated by underscores (e.g. 25_25_25)')
parse.add_argument(
    '--l2_penalty',
    type=float,
    default=1e-2,
    help='Lambda for L2 regularization term in MMDNet loss')
parse.add_argument(
    '--initial_lr',
    type=float,
    default=1e-3,
    help='Learning rate to start with at beginning of MMDNet training')
parse.add_argument(
    '--lr_decay',
    type=float,
    default=0.97,
    help='Decay rate of MMDNet learning rate, such that lr at epoch x is lr_decay ^ x')

FLAGS = parse.parse_args()

source = np.genfromtxt(FLAGS.source_dir, delimiter=',', skip_header=0)
target = np.genfromtxt(FLAGS.target_dir, delimiter=',', skip_header=0)

# pre-process data: log transformation, a standard practice with CyTOF data
target = dh.preProcessCytofData(target)
source = dh.preProcessCytofData(source)

from MMD_ResNet import MMDNet
data_dim = source.shape[-1]
mmdnet = MMDNet(data_dim,
        epochs=FLAGS.epochs,
        denoise=FLAGS.denoise,
        ae_latent_dim=FLAGS.ae_latent_dim,
        layer_sizes=[int(x) for x in FLAGS.layer_sizes.split('_')],
        l2_penalty=FLAGS.l2_penalty)
mmdnet.build_model()
mmdnet.fit(source, target, FLAGS.initial_lr, FLAGS.lr_decay)
mmdnet.evaluate(source, target)
