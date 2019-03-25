#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Created on Tue Mar 14 18:07:32 2017

@author: Daniel Popovic
"""

import argparse
import os
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Artificial neural network-based \
                                 outlier detection.')

# Positional arguments
parser.add_argument('method', 
                    choices=['ae', 'rbm', 'som', 'all'], 
                    help='The outlier detection method to be used. \
                    Allowed values are "ae" for Autoencoder, "rbm" for \
                    Restricted Boltzmann Machine, "som" for \
                    Self-Organising Map, and "all" for all three methods \
                    at once.')
parser.add_argument('dataset', 
                    help='The path to and the name of the data set \
                    file. Allowed file formats are ARFF (.arff), CSV (.csv), \
                    and MATLAB < v7.3 (.mat). See the repository README.MD \
                    (https://github.com/Cognitana-Research/NNOHD) for details \
                    about the supported file formats.')

# Optional arguments
parser.add_argument('-w', 
                    '--writerankings', 
                    action='store_true',
                    help='Write the outlier scores to CSV file(s). For each \
                    method, a separate file is created that contains the \
                    an incremental data object ID, the real outlier label, \
                    and the predicted outlier score for each data object.')
parser.add_argument('-d', 
                    '--delimiter', 
                    default=';',
                    help='The delimiter for CSV data set files. Default \
                    is ";".')
parser.add_argument('-a', 
                    '--aeepochs', 
                    type=int, 
                    default=20, 
                    help='The number of training epochs for the Autoencoder. \
                    Default is 20.')
parser.add_argument('-e', 
                    '--aeencodingfactor', 
                    type=float, 
                    default=0.8, 
                    help='The encoding factor for the Autoencoder hidden \
                    layers. Allowed values between 0.0 and 1.0. Defaul is \
                    0.8')
parser.add_argument('-r', 
                    '--rbmepochs', 
                    type=int, 
                    default=100, 
                    help='The number of training epochs for the Restricted \
                    Boltzmann Machine. Default is 100.')
parser.add_argument('-n', 
                    '--rbmhiddenneurons', 
                    type=float, 
                    default=0.8, 
                    help='The share of hidden neurons for the Restricted \
                    Boltzmann Machine. Allowed values between 0.0 and 1.0. \
                    Default is 0.9')
parser.add_argument('-s', 
                    '--somepochs', 
                    type=int, 
                    default=10, 
                    help='The number of training epochs for the \
                    Self-Organising Map. Default is 10.')
parser.add_argument('-g', '--somgrid', 
                    type=float, 
                    default=2, 
                    help='The grid size for the Self-Organising Map. A value \
                    of g leads to a rectangular grid of g rows x g columns. \
                    Default is 2.')

args = parser.parse_args()

save_rankings = args.writerankings
pathname, filename = os.path.split(args.dataset)

if args.method in ('ae', 'all'):
    from autoencoder import Autoencoder
    
    ae = Autoencoder(filename, 
                     pathname, 
                     args.aeencodingfactor, 
                     args.aeepochs, 
                     args.delimiter)
    
    print('----------')
    print('Autoencoder results on data set {}:'.format(filename))
    print('----------')
    print('Number of dimensions: {}'.format(ae.dimensions_count))
    print('Number of data objects: {}'.format(ae.objects_count))
    print('Number of outliers: {}'.format(ae.outlier_count))
    print('----------')
    print('ROC AUC: {}'.format(ae.roc_auc))
    print('PR AUC: {}'.format(ae.pr_auc))
    print('F1 Score: {}'.format(ae.f1_score))
    print('----------\n')
    if save_rankings:
        with open('ae_{}epochs_{}encodingfactor_{}.csv'.format(
                args.aeepochs,
                args.aeencodingfactor,
                filename), 'w') as aefile:
            # Save outlier ranking descending
            aefile.write('Data object ID;Real outlier label;Predicted outlier score\n')
            counter = 0
            for value in ae.data['dist']:
                aefile.write('{};{};{}\n'.format(
                        counter, 
                        ae.labels[counter], 
                        value))
                counter += 1
            print('Saved as ae_{}epochs_{}encodingfactor_{}.csv\n'.format(
                    args.aeepochs,
                    args.aeencodingfactor,
                    filename))

elif args.method in ('rbm', 'all'):
    from rbm import RBM
    
    rbm = RBM(filename,
              pathname,
              args.rbmepochs,
              args.rbmhiddenneurons,
              args.delimiter)

    print('----------')
    print('RBM results on data set {}:'.format(filename))
    print('----------')
    print('Number of dimensions: {}'.format(rbm.dimensions_count))
    print('Number of data objects: {}'.format(rbm.objects_count))
    print('Number of outliers: {}'.format(rbm.outlier_count))
    print('----------')
    print('ROC AUC: {}'.format(rbm.roc_auc))
    print('PR AUC: {}'.format(rbm.pr_auc))
    print('F1 Score: {}'.format(rbm.f1_score))
    print('----------\n')
    if save_rankings:
        with open('rbm_{}epochs_{}hidden_{}.csv'.format(
                args.rbmepochs,
                args.rbmhiddenneurons,
                filename), 'w') as rbmfile:
            # Save outlier ranking descending
            rbmfile.write('Data object number;Real outlier label;Predicted outlier score\n')
            counter = 0
            for value in rbm.fe:
                rbmfile.write('{};{};{}\n'.format(
                        counter, 
                        rbm.labels[counter],
                        value))
                counter += 1
            print('Saved as rbm_{}epochs_{}hidden_{}.csv\n'.format(
                    args.rbmepochs,
                    args.rbmhiddenneurons,
                    filename))

elif args.method in ('som', 'all'):
    from som import SOM
    
    som = SOM(filename,
              pathname,
              args.somepochs,
              args.somgrid,
              args.somgrid,
              args.delimiter)
    
    print('----------')
    print('SOM results on data set {}:'.format(filename))
    print('----------')
    print('Number of dimensions: {}'.format(som.dimensions_count))
    print('Number of data objects: {}'.format(som.objects_count))
    print('Number of outliers: {}'.format(som.outlier_count))
    print('----------')
    print('ROC AUC: {}'.format(som.roc_auc))
    print('PR AUC: {}'.format(som.pr_auc))
    print('F1 Score: {}'.format(som.f1_score))
    print('----------\n')
    if save_rankings:
        with open('som_{}epochs_{}grid_{}.csv'.format(
                args.somepochs,
                args.somgrid,
                filename), 'w') as somfile:
            # Save outlier ranking descending
            somfile.write('Data object number;Real outlier label;Predicted outlier score\n')
            counter = 0
            for value in som.qe:
                somfile.write('{};{};{}\n'.format(
                        counter, 
                        som.labels[counter],
                        value))
                counter += 1
            print('Saved as som_{}epochs_{}grid_{}.csv\n'.format(
                    args.somepochs,
                    args.somgrid,
                    filename))
    
