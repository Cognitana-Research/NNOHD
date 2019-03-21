# -*- coding: utf-8 -*-

"""
Created on Thu Apr 30 06:34:43 2017

@author: Daniel Popovic
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import somoclu
from utils.preprocessing import Preprocessor
from sklearn.metrics import roc_curve, auc, precision_recall_curve, log_loss
from sklearn import preprocessing as skpre
import scipy.spatial.distance as spdist

class SOM():
    """
    Self-Organising Map for outlier detection.
    
    Uses Somoclu (http://peterwittek.github.io/somoclu/) SOM implementation as 
    base for outlier detection. The implementation generates the SOM and 
    performs training on the specified data set within initialization
    to facilitate batch runs for different data sets and SOM parameter sets.
    
    Parameters
    ----------
    filename : str
        The filename of the data set on which to perform outlier detection. 
        Currently supported file types:
        ARFF .arff, MATLAB .mat < version 7.3, CSV .csv .txt
    filepath : str, optional
        Specifies the path where the data set is located.
        Default: 'datasets'.
    rows : int, optional 
        The number of rows in the map.
        Default: 5
    columns : int, optional 
        The number of columns in the map.
        Default: 5
    epochs : int, optional 
        The number of training epochs to perform on the specified data set.
        Default: 10
    delimiter : str, optional 
        The column delimiter for data sets loaded from CSV files.
        Default: ','
    initialization : {'pca', 'random'}, optional 
        The initialization method for the SOM neuron weights.
            * 'random': Random selection for the initial weights
            * 'pca': Initialization from the first subspace spanned by the 
                     first two eigenvectors of the correlation matrix (default)
    error : {'euclid', 'max', 'manhattan', 'cosine', 'correlation', 'angle'}, 
            optional 
        The error function to be used to compute the distance of data objects 
        from their BMU to obtain their outlier scores.
            * 'euclid': Euclidean distance error (default)
            * 'max': Maximum absolute distance error of all dimensions
            * 'manhattan': Manhattan distance error
            * 'cosine': Cosine distance error
            * 'correlation': Correlation error
            * 'angle': Angle-based error
    maptype : {'planar', 'toroid'}, optional 
        The map topology of the SOM.
            * 'planar': A planar map (default)
            * 'toroid': A toroid map
    gridtype : {'rectangular', 'hexagonal'}, optional 
        The grid shape of the SOM.
            * 'rectangular': A rectangular grid (default)
            * 'hexagonal': A hexagonal grid
    """
    
    def __init__(self, filename, 
                 filepath="datasets", 
                 rows=5, 
                 columns=5,
                 epochs=10, 
                 delimiter=",", 
                 initialization="pca", 
                 error="euclid", 
                 maptype="planar", 
                 gridtype="rectangular"):
        """
        Constructor for the SOM class.
        """
        self.filename = filename
        self.filepath = filepath
        file = "{}/{}".format(filepath,filename)
        self.epochs = epochs
        self.rows, self.columns = rows, columns
        self.maptype = maptype
        self.gridtype = gridtype
        input_data = Preprocessor(file, delimiter=delimiter)
        self.data = input_data.data
        self.data_norm = input_data.data_normalized.values
        self.labels = input_data.labels.values
        self.objects_count = input_data.objects_count
        self.inlier_count = input_data.inlier_count
        self.outlier_count = input_data.outlier_count
        self.dimensions_count = input_data.dimensions_count
        self.som = somoclu.Somoclu(columns, rows, compactsupport=False,
                                   initialization=initialization,
                                   maptype=maptype, gridtype=gridtype)
        self.som.train(self.data_norm, epochs=epochs)
        self.bmus = self.som.get_bmus(self.som.get_surface_state())
        self.qe = []
        self.qebmu = []
        self.bmulocations = []
        if error == "euclid":
            for i in range(self.bmus.shape[0]):
                self.bmulocations.append(self.som.codebook[
                        self.bmus[i][1]][self.bmus[i][0]])
                qerror = np.linalg.norm(
                        self.bmulocations[i] - self.data_norm[i])
                self.qe.append(qerror)
        elif error == "max":
            for i in range(self.bmus.shape[0]):
                self.bmulocations.append(self.som.codebook[
                        self.bmus[i][1]][self.bmus[i][0]])
                qerror = np.amax(self.bmulocations[i] - self.data_norm[i])
                self.qe.append(qerror)
        elif error == "manhattan":
            for i in range(self.bmus.shape[0]):
                self.bmulocations.append(self.som.codebook[
                        self.bmus[i][1]][self.bmus[i][0]])
                qerror = spdist.cityblock(self.bmulocations[i], 
                                          self.data_norm[i])
                self.qe.append(qerror)
        elif error == "cosine":
            for i in range(self.bmus.shape[0]):
                self.bmulocations.append(self.som.codebook[
                        self.bmus[i][1]][self.bmus[i][0]])
                qerror = spdist.cosine(self.bmulocations[i], 
                                       self.data_norm[i])
                self.qe.append(qerror)
        elif error == "correlation":
            for i in range(self.bmus.shape[0]):
                self.bmulocations.append(self.som.codebook[
                        self.bmus[i][1]][self.bmus[i][0]])
                qerror = spdist.correlation(self.bmulocations[i], 
                                            self.data_norm[i])
                self.qe.append(qerror)
        elif error == "angle":
            for i in range(self.bmus.shape[0]):
                self.bmulocations.append(self.som.codebook[
                        self.bmus[i][1]][self.bmus[i][0]])
                qerror = np.arccos(
                        np.clip(np.dot(
                                (self.data_norm[i] \
                                 / np.linalg.norm(self.data_norm[i])), 
                                (self.bmulocations[i] \
                                 / np.linalg.norm(self.bmulocations[i]))),
                            -1.0, 1.0))
                self.qe.append(qerror)
        minmax_scaler = skpre.MinMaxScaler()
        standard_scaler = skpre.StandardScaler()
        self.qe_scaled = minmax_scaler.fit_transform(
                np.array(self.qe).reshape(-1,1))
        self.qe_scaled_std = standard_scaler.fit_transform(
                np.array(self.qe).reshape(-1,1))
        self.fpr, self.tpr, self.thresholds = roc_curve(self.labels, self.qe)
        self.roc_auc = auc(self.fpr, self.tpr)
        self.precision, self.recall, self.threshold_pr = \
            precision_recall_curve(self.labels, self.qe)
        self.pr_auc = auc(self.recall, self.precision)
        self.f1_scores = 2 * (self.precision * self.recall) \
            / (self.precision + self.recall)
        self.f1_score = np.max(np.nan_to_num(self.f1_scores))
        self.logloss = log_loss(self.labels, self.qe)
        self.data['qe'] = self.qe

    
    def mid_matrix(self, 
                   codebook):
        """
        Compute the median interneuron distance matrix (MID matrix).
        
        Parameters
        ----------
        codebook : 2D numpy.array of float32
            The SOM codebook of weights.
        
        Returns
        -------
        2D numpy.array of float32
            The MID matrix.
        """
        mid = np.zeros((self.rows,self.columns))
        mid[0,0] = np.median([np.linalg.norm(codebook[0,0]-codebook[0,1]),
                             np.linalg.norm(codebook[0,0]-codebook[1,1]),
                             np.linalg.norm(codebook[0,0]-codebook[1,0])])
        for row in range(1, self.rows-1):
            mid[row,0] = np.median([
                    np.linalg.norm(codebook[row,0]-codebook[row-1,0]),
                    np.linalg.norm(codebook[row,0]-codebook[row-1,1]),
                    np.linalg.norm(codebook[row,0]-codebook[row,1]),
                    np.linalg.norm(codebook[row,0]-codebook[row+1,1]),
                    np.linalg.norm(codebook[row,0]-codebook[row+1,0])])
        for col in range(1, self.columns-1):
            mid[0,col] = np.median([
                    np.linalg.norm(codebook[0,col]-codebook[0,col-1]),
                    np.linalg.norm(codebook[0,col]-codebook[1,col-1]),
                    np.linalg.norm(codebook[0,col]-codebook[1,col]),
                    np.linalg.norm(codebook[0,col]-codebook[1,col+1]),
                    np.linalg.norm(codebook[0,col]-codebook[0,col+1])])
        mid[self.rows-1,0] = np.median(
                [np.linalg.norm(codebook[self.rows-1,0] \
                                -codebook[self.rows-2,0]),
                 np.linalg.norm(codebook[self.rows-1,0] \
                                -codebook[self.rows-2,1]),
                 np.linalg.norm(codebook[self.rows-1,0] \
                                -codebook[self.rows-1,1])])
        for row in range(1, self.rows-1):
            for col in range(1, self.columns-1):
                mid[row,col] = np.median(
                        [np.linalg.norm(codebook[row,col] \
                                        -codebook[row-1,col-1]),
                         np.linalg.norm(codebook[row,col] \
                                        -codebook[row-1,col]),
                         np.linalg.norm(codebook[row,col] \
                                        -codebook[row-1,col+1]),
                         np.linalg.norm(codebook[row,col] \
                                        -codebook[row,col+1]),
                         np.linalg.norm(codebook[row,col] \
                                        -codebook[row+1,col+1]),
                         np.linalg.norm(codebook[row,col] \
                                        -codebook[row+1,col]),
                         np.linalg.norm(codebook[row,col] \
                                        -codebook[row+1,col-1]),
                         np.linalg.norm(codebook[row,col] \
                                        -codebook[row,col-1])])
        mid[0,self.columns-1] = np.median(
                [np.linalg.norm(codebook[0,self.columns-1] \
                                -codebook[0,self.columns-2]),
                 np.linalg.norm(codebook[0,self.columns-1] \
                                -codebook[1,self.columns-2]),
                 np.linalg.norm(codebook[0,self.columns-1] \
                                -codebook[1,self.columns-1])])
        for row in range(1, self.rows-1):
            mid[row,self.columns-1] = np.median(
                    [np.linalg.norm(codebook[row,self.columns-1] \
                                    -codebook[row-1,self.columns-1]),
                     np.linalg.norm(codebook[row,self.columns-1] \
                                    -codebook[row-1,self.columns-2]),
                     np.linalg.norm(codebook[row,self.columns-1] \
                                    -codebook[row,self.columns-2]),
                     np.linalg.norm(codebook[row,self.columns-1] \
                                    -codebook[row+1,self.columns-2]),
                     np.linalg.norm(codebook[row,self.columns-1] \
                                    -codebook[row+1,self.columns-1])])
        for col in range(1, self.columns-1):
            mid[self.rows-1,col] = np.median(
                    [np.linalg.norm(codebook[self.rows-1,col] \
                                    -codebook[self.rows-1,col-1]),
                     np.linalg.norm(codebook[self.rows-1,col] \
                                    -codebook[self.rows-2,col-1]),
                     np.linalg.norm(codebook[self.rows-1,col] \
                                    -codebook[self.rows-2,col]),
                     np.linalg.norm(codebook[self.rows-1,col] \
                                    -codebook[self.rows-2,col+1]),
                     np.linalg.norm(codebook[self.rows-1,col] \
                                    -codebook[self.rows-1,col+1])])
        mid[self.rows-1,self.columns-1] = np.median(
                [np.linalg.norm(codebook[self.rows-1,self.columns-1] \
                                -codebook[self.rows-1,self.columns-2]),
                 np.linalg.norm(codebook[self.rows-1,self.columns-1] \
                                -codebook[self.rows-2,self.columns-2]),
                 np.linalg.norm(codebook[self.rows-1,self.columns-1] \
                                -codebook[self.rows-2,self.columns-1])])
        return mid
    
    def mean_bmu(self, 
                 codebook):
        """
        Compute the mean of the SOM neuron weights.
        
        Parameters
        ----------
        codebook : 2D numpy.array of float32
            The SOM codebook of weights.
        
        Returns
        -------
        1D numpy.array of float32
            The mean of the SOM neuron weights.
        """
        flat_codebook = self.som.codebook.reshape((self.rows*self.columns, 
                                                   self.dimensions_count))
        codebook_mean = flat_codebook.mean(axis=0)
        #print(codebook_mean)
        return codebook_mean
    
    def median_bmu(self, 
                   codebook):
        """
        Compute the median of the SOM neuron weights.
        
        Parameters
        ----------
        codebook : 2D numpy.array of float32
            The SOM codebook of weights.
        
        Returns
        -------
        1D numpy.array of float32
            The median of the SOM neuron weights.
        """
        flat_codebook = self.som.codebook.reshape((self.rows*self.columns, 
                                                   self.dimensions_count))
        codebook_median = np.median(flat_codebook, axis=0)
        #print(codebook_median)
        return codebook_median
    
    def mean_nd_matrix(self, 
                       codebook):
        """
        Compute the distance of each SOM neuron weight to the mean of all SOM 
        neuron weights.
        
        Parameters
        ----------
        codebook : 2D numpy.array of float32
            The SOM codebook of weights.
        
        Returns
        -------
        2D numpy.array of float32
            The mean neuron distance matrix of the SOM neuron weights.
        """
        mnd = np.zeros((self.rows,self.columns))
        mean_neuron = np.mean(np.reshape(codebook, 
                                         (self.rows*self.columns, 
                                          self.dimensions_count)))
        for row in range(self.rows):
            for col in range(self.columns):
                mnd[row,col] = np.mean(abs(codebook[row,col,:] - mean_neuron))
        return mnd
    
    def median_nd_matrix(self, 
                         codebook):
        """
        Compute the distance of each SOM neuron weight to the median of all 
        SOM neuron weights.
        
        Parameters
        ----------
        codebook : 2D numpy.array of float32
            The SOM codebook of weights.
        
        Returns
        -------
        2D numpy.array of float32
            The median neuron distance matrix of the SOM neuron weights.
        """
        mnd = np.zeros((self.rows,self.columns))
        median_neuron = np.median(np.reshape(codebook, 
                                             (self.rows*self.columns, 
                                              self.dimensions_count)))
        for row in range(self.rows):
            for col in range(self.columns):
                mnd[row,col] = np.median(
                        abs(codebook[row,col,:] - median_neuron))
        return mnd

    def save_labels_qes(self):
        """
        Save the outlier labels and the quantization error for each data 
        object as CSV file.
        """
        with open('som_labels_qes_{}rows_{}cols_{}epochs_{}.csv'.format(
                self.rows, 
                self.columns,
                self.epochs, 
                self.filename),'w') as somfile:    
            for i in range(len(self.labels)):
                somfile.write("{},{},{}{}".format(i, 
                              self.labels[i], 
                              self.qe[i], 
                              "\n"))
    
    def save_bmu_locations(self):
        """
        Save the locations of the BMUs in the SOM grid (zero-based row and 
        column index) as CSV file.
        """
        with open('som_bmulocations_{}rows_{}cols_{}epochs_{}.csv'.format(
                self.rows, 
                self.columns,
                self.epochs, 
                self.filename),'w') as somfile:    
            for i in range(len(self.bmulocations)):
                somfile.write("{}{}".format(self.bmulocations[i], "\n"))
    
    def save_codebook(self):
        """
        Save the SOM neuron weights as CSV file.
        """
        with open('som_codebook_{}rows_{}cols_{}epochs_{}.csv'.format(
                self.rows, 
                self.columns,
                self.epochs, 
                self.filename),'w') as somfile:    
            for i in range(len(self.som.codebook)):
                somfile.write("{}{}".format(self.som.codebook[i], "\n"))
    
    def plot_point_qe_per_dimension(self, 
                                    point_index, 
                                    save_file=True, 
                                    plot_qe=True, 
                                    plot_bmu=False, 
                                    plot_data=False):
        """
        Plot and/or save the quantization errors per dimension for a specified 
        data object.
        
        Parameters
        ----------
        point_index : int
            The zero-based index in the data set of the data object.
        save_file : bool, optional 
            Select to save the quantization errors per dimension as CSV file.
            Default: True
        plot_qe : bool, optional 
            Select to plot the quantization errors per dimension and save the 
            plot as PNG file.
            Default: True
        plot_bmu : bool, optional 
            Select to also plot the weight value per dimension of the BMU.
            Default: False
        plot_data : bool, optional 
            Select to also plot the data object value per dimension.
            Default: False
        
        Returns
        -------
        2D numpy.array of float32
            The quantization errors per dimension for the data object.
        """
        i = point_index
        qes_per_dim = np.array(abs(
                self.som.codebook[
                        self.bmus[i][1]][
                                self.bmus[i][0]] - self.data_norm[i]))
        bmu_per_dim = np.array(self.som.codebook[
                self.bmus[i][1]][self.bmus[i][0]])
        if save_file:
            with open('som_qes_per_dim_{}rows_{}cols_{}epochs_{}_point{}\
                      .csv'.format(self.rows, 
                      self.columns,
                      self.epochs, 
                      self.filename, 
                      i), 'w') as somfile:
                for line in qes_per_dim:
                    somfile.write(str(line) + '\n')
        if plot_qe:
            plt.figure(figsize=(6,3))
            plt.plot(qes_per_dim)
            if plot_bmu:
                plt.plot(bmu_per_dim, c='r')
            if plot_data:
                plt.plot(self.data_norm[i], c='g')
            plt.xlim((0,self.dimensions_count))
            plt.xlabel('Dimension')
            plt.ylabel('Quantization error')
            plt.title("Quantization error in each dimension \
                      for data object {}".format(i))
            plt.tight_layout()
            plt.savefig("SOM_QE_per_dim_point_{}_{}.pdf".format(i, 
                        self.filename), format="pdf")
        return qes_per_dim
        
    def plot_roc_auc(self):
        """
        Plot the receiver operating characteristic (ROC) curve.
        """
        plt.figure(figsize=(10,6))
        plt.plot(self.fpr, 
                 self.tpr, 
                 color='red', 
                 label='AUC = %0.2f)' % self.roc_auc)
        plt.xlim((0,1))
        plt.ylim((0,1))
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('FP-rate')
        plt.ylabel('TP-rate')
        plt.title('ROC for SOM grid {}*{}, {} epochs, {} data set'.format(
                self.rows, 
                self.columns, 
                self.epochs, 
                self.filename))
        plt.legend(loc="lower right")
        plt.show()
    
    def plot_pr_auc(self):
        """
        Plot the precision recall (PR) curve.
        """
        plt.figure(figsize=(10,6))
        plt.plot(self.recall, 
                 self.precision, 
                 color='red', 
                 label='AUC = %0.2f' % self.pr_auc)
        plt.xlim((0,1))
        plt.ylim((0,1))
        #plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR for SOM grid {}*{}, {} epochs, {} data set'.format(
                self.rows, 
                self.columns, 
                self.epochs, 
                self.filename))
        plt.legend(loc="lower right")
        plt.show()
    
    def plot_3d_points_and_neurons(self, 
                                   dimensions):
        """
        Plot the data objects and SOM neuron weights in a 3D scatter plot and 
        save as SVG.
        
        Parameters
        ----------
        dimensions : list of int
            A list of three dimensions for the plot.
        """
        fig3d = plt.figure(figsize=(10,7)).gca(projection='3d')
        levels = [0,1,2]
        colors = ['#348ABD', '#A60628']
        cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors)
        

        fig3d.scatter(self.data_norm[:,dimensions[0]], 
                      self.data_norm[:,dimensions[1]], 
                      zs=self.data_norm[:,dimensions[2]], 
                      cmap=cmap, 
                      c=self.labels)
        for i in range(self.rows):
            for j in range(self.columns):
                    fig3d.scatter(self.som.codebook[i,j,dimensions[0]],
                                  self.som.codebook[i,j,dimensions[1]], 
                                  zs=self.som.codebook[i,j,dimensions[2]], 
                                  c='k')
        fig3d.set_edgecolors = fig3d.set_facecolors = lambda *args:None
        fig3d.set_xlabel('x', labelpad=10)
        fig3d.set_ylabel('y', labelpad=10)
        fig3d.set_zlabel('z', labelpad=10)
        plt.savefig("som_3d_plot_outliers_neurons.pdf")
    
    def plot_3d_points(self, 
                       dimensions):
        """
        Plot the data objects and SOM neuron weights in a 3D scatter plot and 
        save as SVG.
        
        Parameters
        ----------
        dimensions : list of int
            A list of three dimensions for the plot.
        """
        fig3d = plt.figure(figsize=(6,3)).gca(projection='3d')
        levels = [0,1,2]
        colors = ['#348ABD', 'r']
        cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors)
        

        fig3d.scatter(self.data_norm[:,dimensions[0]], 
                      self.data_norm[:,dimensions[1]], 
                      zs=self.data_norm[:,dimensions[2]], 
                      cmap=cmap, 
                      c=self.labels)
        fig3d.set_xlabel('x', labelpad=10)
        fig3d.set_ylabel('y', labelpad=10)
        fig3d.set_zlabel('z', labelpad=10)
        plt.tight_layout()
        plt.title("3D visualization in dimensions {}, {} and {}".format(
                dimensions[0], 
                dimensions[1], 
                dimensions[2]))
        plt.savefig("som_3d_plot_x_outliers_dim{}-{}-{}_{}.pdf".format(
                dimensions[0], 
                dimensions[1], 
                dimensions[2], 
                self.filename))

    def plot_2d_points_3_dimensions(self, 
                                    dimensions):
        """
        Plot the data objects and SOM neuron weights in 3 2D scatter plots, 
        one for each combination of the three specified dimensions.
        
        Parameters
        ----------
        dimensions : list of int
            A list of three dimensions for the plots.
        """
        # Create 1x3 sub plots
        gs = gridspec.GridSpec(1, 3, wspace=0.4)
        
        plt.figure(figsize=(11,2.5))
        
        ax = plt.subplot(gs[0, 0]) # row 0, col 0
        plt.scatter(self.data_norm[:,dimensions[0]], 
                    self.data_norm[:,dimensions[1]])
        plt.xlabel('x')
        plt.ylabel('y')
        
        ax = plt.subplot(gs[0, 1]) # row 0, col 1
        plt.scatter(self.data_norm[:,dimensions[1]], 
                    self.data_norm[:,dimensions[2]])
        plt.xlabel('y')
        plt.ylabel('z')
        
        ax = plt.subplot(gs[0, 2]) # row 0, col 2
        plt.scatter(self.data_norm[:,dimensions[0]], 
                    self.data_norm[:,dimensions[2]])
        plt.xlabel('x')
        plt.ylabel('z')
        
        plt.show()
    
    def plot_outlier_score(self):
        """
        Plot the outlier score for each data object in the data set and save 
        as PDF.
        """
        plt.figure(figsize=(6,3))
        levels = [0,1,2]
        colors = ['#348ABD', '#A60628']
        cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors)
        plt.scatter(self.data.index, 
                    self.data['qe'], 
                    c=self.labels, 
                    cmap=cmap, 
                    edgecolor='black', s=15)
        plt.xlabel('Index')
        plt.ylabel('Quantization error')
        plt.xlim((0,self.objects_count))
        plt.title("Quantization error for each data object")
        plt.tight_layout()
        plt.savefig("som_outlier_scores-{}.pdf".format(self.filename), 
                    format="pdf")
    
    def print_f1_log_loss(self):
        """
        Print the F1 score and the log loss to the console.
        """
        f_one_score = 2 * (self.precision * self.recall) / \
                          (self.precision + self.recall)
        logloss = log_loss(self.labels, self.qe)
        print("F1-Score: {}".format(np.mean(f_one_score)))
        
        print("Log loss: {}".format(logloss))

