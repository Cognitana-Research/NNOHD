#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 18:33:23 2017

@author: Daniel Popovic
"""

import numpy as np
import pandas as pd
from scipy.io import arff, loadmat
from sklearn import preprocessing


class Preprocessor(object):
    """
    Preprocess data files to be used as training and test data for neural 
    networks. Creating an instance of this class with the filename (string) 
    parameter will load that file, fill the class instance parameters based on
    its content and normalize the contained data to a value range of [0, 1].
    
    Parameters
    ----------
    data : pandas.DataFrame
        Contains the data read from the input data file without class labels, 
        before normalization.
    data_normalized : pandas.DataFrame
        Contains the data read from the input file without class labels, after 
        normalization of each attribute
        to the range [0, 1].
    meta : scipy.io.arff.MetaData
        Contains useful meta information from datasets about the attributes 
        and types in the data
        and the original class range before normalization, if available.
    labels : pandas.Series
        Contains the class labels for the data.
    objects_count : numpy.int64
        The number of objects (rows) in the data.
    dimensions_count : numpy.int64
        The number of dimensions (columns) in the data.
    inlier_count : numpy.int64
        The number of inliers in the data according to the class labels.
    outlier_count : numpy.int64
        The number of outliers in the data according to the class labels.
    
    Notes
    -----
    Currently supported file types: ARFF .arff, MATLAB .mat < version 7.3, CSV 
    .csv .txt
    """
    def __init__(self, 
                 filename, 
                 with_labels=True, 
                 with_header=False, 
                 delimiter=","):
        if filename.endswith('.arff'):
            self._load_arff_data(filename)
            self._preprocess(self.data)
        elif filename.endswith('.txt') or filename.endswith('.csv'):
            self._load_csv_data(filename, with_labels, with_header, delimiter)
            self._preprocess(self.data)
        elif filename.endswith('.mat'):
            self._load_mat_data(filename)
            self._preprocess(self.data)
        elif filename.endswith('.h5'):
            self._load_hdf5_data(filename)
            self._preprocess(self.data)
        self.inlier_count = self.labels.value_counts()[0]
        self.outlier_count = self.labels.value_counts()[1]
        self.dimensions_count = self.data.shape[1]
        self.objects_count = self.data.shape[0]

    
    def _load_arff_data(self, 
                        filename):
        """
        Load training data and meta info from .arff file, including class 
        label, and fill class instance variables.
        
        Parameters
        ----------
        filename : string
            The full path and filename for the .arff data file to be loaded.
        """
        self.data, self.meta = arff.loadarff(filename)
        self.data = pd.DataFrame(self.data)
        self.data = self.data.sample(frac=1)
        # For .arff files from the LMU data repository: delete 'ID' column and 
        # transform 'outlier' column (no/yes) to 'label' column (0/1)
        if "id" in self.data.columns:
            del self.data["id"]
        if "outlier" in self.data.columns:
            self.data["outlier"][self.data["outlier"] == b"'yes'"] = 1
            self.data["outlier"][self.data["outlier"] == b"'no'"] = 0
            labels = np.array(self.data["outlier"], dtype=np.int)
            del self.data["outlier"]
            self.data["class"] = labels

        self.labels = self.data['class'].astype(int)
        self.labels[self.labels != 0] = 1
        #class_labels = pd.value_counts(self.labels.values, sort=False)
    
        del self.data['class']
        
        #self.inlier_count = class_labels[0]
        #self.outlier_count = class_labels[1]
        #self.dimensions_count = len(self.meta.names()) - 1
        #self.objects_count = len(self.labels)

    def _load_csv_data(self, 
                       filename, 
                       with_labels=True, 
                       with_header=False, 
                       delimiter=","):
        """
        Load training data from .csv or .txt file in CSV format with specified 
        delimiter (default ",") and fill class instance variables.
        
        Parameters
        ----------
        filename: string
            The full path and filename for the .csv or .txt file to be loaded.
        with_labels: boolean, default True
            Indicates whether or not class labels are included in the file. 
            Labels have to be in the last column.
        delimiter: string, default ","
            The delimiter character for the CSV data.
        """
        self.data = pd.read_csv(filename, 
                                sep=delimiter, 
                                header=None, 
                                names=None)
        self.data = self.data.sample(frac=1)
        if with_header:
            self.data = self.data.iloc[2:]
        if with_labels:
            self.labels = self.data[self.data.columns[-1]].astype(int)
            del self.data[self.data.columns[-1]]
        else:
            self.labels = np.zeros(len(self.data.index))

        #class_labels = pd.value_counts(self.labels.values, sort=False)
        
    def _load_mat_data(self, filename):
        """
        Load training data from Matlab .mat file and fill class instance 
        variables (Matlab version 7.2 or lower).
        
        Parameters
        ----------
        filename: string
            The full path and filename for the .mat file to be loaded.
        """
        matdict = loadmat(filename)
        self.data = pd.DataFrame(matdict['X'])
        #self.data = self.data.sample(frac=1)
        self.labels = pd.Series([row[0].astype(int) for row in matdict['y']])

    def _load_hdf5_data(self, filename):
        """
        Load training data from HDF5 .h5 file and fill class instance 
        variables. For Matlab .mat file with version > 7.2: Rename the file to 
        .h5, as Matlab uses HDF5 format from version 7.3 onwards.
        
        Parameters
        ----------
        filename: string
            The full path and filename for the .h5 file to be loaded.
        """

    def _preprocess(self, data):
        """
        Proprocess data to normalize all attributes to a range [0, 1], using 
        sklearn.preprocessing.MinMaxScaler.
        
        Parameters
        ----------
        data : pandas.DataFrame
            The data set to be normalized.
        """
        min_max_scaler = preprocessing.MinMaxScaler()
        np_scaled = min_max_scaler.fit_transform(data)
        self.data_normalized = pd.DataFrame(np_scaled)
        self.data_normalized = self.data_normalized.astype('float32')
       
