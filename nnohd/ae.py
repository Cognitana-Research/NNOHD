# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 18:07:32 2017

@author: Daniel Popovic
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import matplotlib.colors
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as k
import numpy as np
import scipy.spatial.distance as spdist
from utils.preprocessing import Preprocessor


class Autoencoder():
    """
    Autoencoder for outlier detection using TensorFlow and Keras.
    
    The autoencoder can be configured to have 1, 3 or 5 hidden layers.
    
    Parameters
    ----------
    filename : str
        The filename of the data set on which to perform outlier detection. 
        Currently supported file types:
        ARFF .arff, MATLAB .mat < version 7.3, CSV .csv .txt
    filepath : str, optional 
        Specifies the path where the data set is located.
        Default: 'datasets'
    encoding_factor : float32, optional 
        Specifies the percentage of the data set dimensions that is used as 
        the number of hidden neurons.
        Default: 0.8
    epochs : int, optional 
        Specifies the number of training epochs.
        Default: 100
    hidden_layers : int, optional 
        The number of hidden layers.
        Default: 1
    activation_functions : 1D array of str, optional 
        Specifies an array of activation functions for the layers. The array 
        must contain as many items as there are layer transitions, which is
        hidden_layers+1. Possible activation functions:
            * 'relu': Rectified linear unit function
            * 'sigmoid': Sigmoid function
            * 'tanh': Tangens hyperbolic function
            * 'elu': Exponential linear unit function
            * 'selu': Scaled exponential linear unit function
            * 'softplus': Softplus function
            * 'softsign': Softsign function
            * 'hard_sigmoid': Hard sigmoid function
            * 'linear': Linear function
        Default: ['relu', 'sigmoid']
    optimizer : {'adadelta', 'sgd', 'rmsprop', 'adagrad', 'adam', 'adamax', \
                 'nadam'}, optional 
        The optimization algorithm. Possible optimizers:
            * 'sgd': Stochastic gradient descent optimizer
            * 'rmsprop': RMSProp optimizer
            * 'adagrad': Adagrad optimizer
            * 'adadelta': Adadelta optimizer
            * 'adam': Adam optimizer
            * 'adamax': Adamax optimizer
            * 'nadam': Nesterov Adam optimizer
        Default: 'adadelta'
    loss_function : {'binary_crossentropy', 'mean_squared_error', \
                     'mean_absolute_error', \
                     'mean_absolute_percentage_error', \
                     'mean_squared_logarithmic_error', 'hinge', \
                     'squared_hinge', 'logcosh', \
                     'kullback_leibler_divergence', 'poisson'}, optional
        The loss function. Possible loss functions:
            * 'binary_crossentropy': Binary cross-entropy
            * 'mean_squared_error': Mean squared error
            * 'mean_absolute_error': Mean absolute error
            * 'mean_absolute_percentage_error': Mean absolute percentage error
            * 'mean_squared_logarithmic_error': Mean squared logarithmic error
            * 'hinge': Hinge loss function
            * 'squared_hinge': Squared hinge loss
            * 'logcosh': Logarithmic cosine hyperbolic loss
            * 'kullback_leibler_divergence': Kullback-Leibler divergence
            * 'poisson': Poisson loss function
        Default: 'binary_crossentropy'
    multi_encode : bool, optional 
        Specifies whether or not to use the encoding_factor also for deeper 
        layers. If False, the hidden layers from 2 onwards use the same number 
        of neurons as the first hidden layer.
        Default: False
    error : {'euclid', 'max', 'manhattan', 'cosine', 'correlation', 'angle'}, \
        optional 
    The error function to be used to compute the distance of data objects from
    their reconstruction at the output layer to obtain their outlier scores.
        * 'euclid': Euclidean distance error (default)
        * 'max': Maximum absolute distance error of all dimensions
        * 'manhattan': Manhattan distance error
        * 'cosine': Cosine distance error
        * 'correlation': Correlation error
        * 'angle': Angle-based error
    delimiter : str, optional
    The column delimiter for data sets loaded from CSV files.
    Default: ','
    """
    
    def __init__(self, 
                 filename, 
                 filepath="datasets", 
                 encoding_factor=0.8, 
                 epochs=100, 
                 hidden_layers=1,
                 multi_encode=False, 
                 activation_functions=None, 
                 optimizer=None, 
                 loss_function=None,
                 error='euclid', 
                 delimiter=",", 
                 with_header=False):
        """
        Constructor for Autoencoder class.
        """
        print(k.tensorflow_backend._get_available_gpus())
        self.filename = filename
        self.filepath = filepath
        file = "{}/{}".format(filepath,filename)
        self.epochs = epochs
        self.encoding_factor = encoding_factor
        self.error = error
        input_data = Preprocessor(file, 
                                  delimiter=delimiter, 
                                  with_header=with_header)
        self.data = input_data.data
        self.data_norm = input_data.data_normalized
        self.labels = input_data.labels.values
        self.objects_count = input_data.objects_count
        self.inlier_count = input_data.inlier_count
        self.outlier_count = input_data.outlier_count
        self.dimensions_count = input_data.dimensions_count
        self.compressed_dimensions = int(np.floor(
                self.dimensions_count * self.encoding_factor))
        if optimizer == None:
            optimizer = 'adadelta'
        if loss_function == None:
            loss_function = 'binary_crossentropy'
        if hidden_layers == 1:
            if activation_functions == None:
                activation_functions = ['relu', 'sigmoid']
            self.autoencoder, self.encoder, self.decoder = \
                self.prepare_autoencoder_1hl(encoding_factor, 
                                             activation_functions,
                                             optimizer,
                                             loss_function)
        elif hidden_layers == 2:
            if activation_functions == None:
                activation_functions = ['relu', 'relu', 'sigmoid']
            self.autoencoder, self.encoder, self.decoder = \
                self.prepare_autoencoder_2hl(encoding_factor,
                                             activation_functions,
                                             optimizer,
                                             loss_function)
        elif hidden_layers == 3:
            if activation_functions == None:
                activation_functions = ['relu', 'sigmoid', 'relu', 'sigmoid']
            if multi_encode:
                self.autoencoder, self.encoder, self.decoder = \
                    self.prepare_autoencoder_3hl_multi_encode(
                            encoding_factor,
                            activation_functions,
                            optimizer,
                            loss_function)
            else:
                self.autoencoder, self.encoder, self.decoder = \
                    self.prepare_autoencoder_3hl(encoding_factor,
                                                 activation_functions,
                                                 optimizer,
                                                 loss_function)
        else: # for all other cases, 5 hidden layers are used
            if activation_functions == None:
                activation_functions = ['relu', 
                                        'sigmoid', 
                                        'relu', 
                                        'sigmoid', 
                                        'relu', 
                                        'sigmoid']
            if multi_encode:
                self.autoencoder, self.encoder, self.decoder = \
                    self.prepare_autoencoder_5hl_multi_encode(
                            encoding_factor,
                            activation_functions,
                            optimizer,
                            loss_function)
            else:
                self.autoencoder, self.encoder, self.decoder = \
                    self.prepare_autoencoder_5hl(encoding_factor,
                                                 activation_functions,
                                                 optimizer,
                                                 loss_function)
        self.data['naivedist'] = self.compute_distance()
        
        # fit the model
        self.autoencoder.fit(self.data_norm.values, self.data_norm.values,
                        epochs=self.epochs,
                        batch_size=self.dimensions_count,
                        shuffle=True,
                        verbose=0)
        
        ## evaluation
        self.data['dist'] = self.compute_distance()
        self.roc_auc, self.fpr, self.tpr = self.compute_roc_auc()
        self.pr_auc, self.precision, self.recall = self.compute_pr_auc()
        self.f1_scores = 2 * (self.precision * self.recall) \
            / (self.precision + self.recall)
        self.f1_score = np.max(np.nan_to_num(self.f1_scores))
        
    def prepare_autoencoder_1hl(self, encoding_factor=0.8,
                                activations=['relu', 'sigmoid'],
                                optimizer='adadelta',
                                loss='binary_crossentropy'):
        """
        Generate the Keras layers and model for autoencoder with 1 hidden 
        layer.
        
        Parameters
        ----------
        encoding_factor : float32, optional 
            The encoding factor for the hidden layer.
            Default: 0.8
        activations : 1D array of str, optional 
            The array of activation functions for hidden layer and 
            output layer.
            Default: ['relu', 'sigmoid']
        optimizer : str, optional 
            The optimizer algorithm.
            Default: 'adadelta'
        loss : str, optional 
            The loss function.
            Default: 'binary_crossentropy'
        
        Returns
        -------
        keras.model, keras.model, keras.model
            The Keras models for the autoencoder, the encoder and the decoder
        """
        encoding_dim = int(np.floor(self.dimensions_count * encoding_factor))
        
        input = Input(shape=(self.dimensions_count,))
        encoded = Dense(encoding_dim, activation=activations[0])(input)
        decoded = Dense(self.dimensions_count, 
                        activation=activations[1])(encoded)
        
        autoencoder = Model(inputs=input, outputs=decoded)
        
        encoder = Model(inputs=input, outputs=encoded)
        
        encoded_input = Input(shape=(encoding_dim,))
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(inputs=encoded_input, 
                        outputs=decoder_layer(encoded_input))

        autoencoder.compile(optimizer=optimizer, loss=loss)
        return autoencoder, encoder, decoder
    
    def prepare_autoencoder_2hl(self, encoding_factor=0.8,
                                activations=['relu', 'relu', 'sigmoid'],
                                optimizer='adadelta',
                                loss='binary_crossentropy'):
        """
        Generate the Keras layers and model for autoencoder with 2 hidden 
        layers.

        Parameters
        ----------
        encoding_factor : float32, optional 
            The encoding factor for the hidden layers.
            Default: 0.8
        activations : 1D array of str, optional 
            The array of activation functions for hidden layers and 
            output layer.
            Default: ['relu', 'relu', 'sigmoid']
        optimizer : str, optional 
            The optimizer algorithm.
            Default: 'adadelta'
        loss : str, optional 
            The loss function.
            Default: 'binary_crossentropy'
        
        Returns
        -------
        keras.model, keras.model, keras.model
            The Keras models for the autoencoder, the encoder and the decoder
        """
        encoding_dim = int(np.floor(self.dimensions_count * encoding_factor))
        
        input = Input(shape=(self.dimensions_count,))
        encoded = Dense(encoding_dim, activation=activations[0])(input)
        hidden_1 = Dense(encoding_dim, activation=activations[1])(encoded)
        decoded = Dense(self.dimensions_count, 
                        activation=activations[2])(hidden_1)
        
        autoencoder = Model(inputs=input, outputs=decoded)
        
        encoder = Model(inputs=input, outputs=encoded)
        
        encoded_input = Input(shape=(encoding_dim,))
        hidden_layer = autoencoder.layers[-2]
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(inputs=encoded_input, 
                        outputs=decoder_layer(hidden_layer(encoded_input)))

        autoencoder.compile(optimizer=optimizer, loss=loss)
        return autoencoder, encoder, decoder
    
    def prepare_autoencoder_3hl(self, 
                                encoding_factor=0.8,
                                activations=['relu', 
                                             'sigmoid', 
                                             'relu', 
                                             'sigmoid'],
                                optimizer='adadelta', 
                                loss='binary_crossentropy'):
        """
        Generate the Keras layers and model for autoencoder with 3 hidden 
        layers without multi-encoding.
        
        Parameters
        ----------
        encoding_factor : float32, optional 
            The encoding factor for the hidden layers.
            Default: 0.8
        activations : 1D array of str, optional 
            The array of activation functions for hidden layers and 
            output layer.
            Default: ['relu', 'sigmoid', 'relu', 'sigmoid']
        optimizer : str, optional 
            The optimizer algorithm.
            Default: 'adadelta'
        loss : str, optional 
            The loss function.
            Default: 'binary_crossentropy'
        
        Returns
        -------
        keras.model, keras.model, keras.model
            The Keras models for the autoencoder, the encoder and the decoder
        """
        encoding_dim = int(np.floor(self.dimensions_count * encoding_factor))
        
        input = Input(shape=(self.dimensions_count,))
        encoded = Dense(encoding_dim, activation=activations[0])(input)
        hidden_1 = Dense(
                int(np.floor(self.dimensions_count * encoding_factor)), 
                activation=activations[1])(encoded)
        hidden_2 = Dense(encoding_dim, activation=activations[2])(hidden_1)
        decoded = Dense(self.dimensions_count, 
                        activation=activations[3])(hidden_2)
        
        autoencoder = Model(inputs=input, outputs=decoded)
        
        encoder = Model(inputs=input, outputs=encoded)
        
        encoded_input = Input(shape=(encoding_dim,))
        hidden_layer_1 = autoencoder.layers[-3]
        hidden_layer_2 = autoencoder.layers[-2]
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(inputs=encoded_input, 
                        outputs=decoder_layer(hidden_layer_2(hidden_layer_1(
                                encoded_input))))

        autoencoder.compile(optimizer=optimizer, loss=loss)
        return autoencoder, encoder, decoder

    def prepare_autoencoder_3hl_multi_encode(
            self, 
            encoding_factor=0.8,
            activations=['relu', 'sigmoid', 'relu', 'sigmoid'],
            optimizer='adadelta', loss='binary_crossentropy'):
        """
         Generate the Keras layers and model for autoencoder with 3 hidden 
         layers with multi-encoding.
       
        Parameters
        ----------
        encoding_factor : float32, optional 
            The encoding factor for the hidden layers.
            Default: 0.8
        activations : 1D array of str, optional 
            The array of activation functions for hidden layers and 
            output layer.
            Default: ['relu', 'sigmoid', 'relu', 'sigmoid']
        optimizer : str, optional 
            The optimizer algorithm.
            Default: 'adadelta'
        loss : str, optional 
            The loss function.
            Default: 'binary_crossentropy'
        
        Returns
        -------
        keras.model, keras.model, keras.model
            The Keras models for the autoencoder, the encoder and the decoder
        """
        encoding_dim = int(np.floor(self.dimensions_count * encoding_factor))
        
        input = Input(shape=(self.dimensions_count,))
        encoded = Dense(encoding_dim, activation=activations[0])(input)
        hidden_1 = Dense(int(np.floor(self.encoding_dim * encoding_factor)), 
                         activation=activations[1])(encoded)
        hidden_2 = Dense(encoding_dim, activation=activations[2])(hidden_1)
        decoded = Dense(self.dimensions_count, 
                        activation=activations[3])(hidden_2)
        
        autoencoder = Model(inputs=input, outputs=decoded)
        
        encoder = Model(inputs=input, outputs=encoded)
        
        encoded_input = Input(shape=(encoding_dim,))
        hidden_layer_1 = autoencoder.layers[-3]
        hidden_layer_2 = autoencoder.layers[-2]
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(
                inputs=encoded_input, 
                outputs=decoder_layer(hidden_layer_2(hidden_layer_1(
                        encoded_input))))

        autoencoder.compile(optimizer=optimizer, loss=loss)
        return autoencoder, encoder, decoder
    
    def prepare_autoencoder_5hl(
            self, 
            encoding_factor=0.8,
            activations=['relu', 'sigmoid', 'relu', 
                         'sigmoid', 'relu', 'sigmoid'],
            optimizer='adadelta', 
            loss='binary_crossentropy'):
        """
        Generate the Keras layers and model for autoencoder with 5 hidden 
        layers without multi-encoding.
        
        Parameters
        ----------
        encoding_factor : float32, optional 
            The encoding factor for the hidden layers.
            Default: 0.8
        activations : 1D array of str, optional 
            The array of activation functions for hidden layers and 
            output layer.
            Default: ['relu', 'sigmoid', 'relu', 'sigmoid', 'relu', 'sigmoid']
        optimizer : str, optional 
            The optimizer algorithm.
            Default: 'adadelta'
        loss : str, optional 
            The loss function.
            Default: 'binary_crossentropy'
        
        Returns
        -------
        keras.model, keras.model, keras.model
            The Keras models for the autoencoder, the encoder and the decoder
        """
        encoding_dim = int(np.floor(self.dimensions_count * encoding_factor))
        
        input = Input(shape=(self.dimensions_count,))
        encoded = Dense(encoding_dim, activation=activations[0])(input)
        hidden_1 = Dense(
                int(np.floor(self.dimensions_count * encoding_factor)), 
                activation=activations[1])(encoded)
        hidden_2 = Dense(
                int(np.floor(encoding_dim * encoding_factor)), 
                activation=activations[2])(hidden_1)
        hidden_3 = Dense(encoding_dim, activation=activations[3])(hidden_2)
        hidden_4 = Dense(encoding_dim, activation=activations[4])(hidden_3)
        decoded = Dense(self.dimensions_count, 
                        activation=activations[3])(hidden_4)
        
        autoencoder = Model(inputs=input, outputs=decoded)
        
        encoder = Model(inputs=input, outputs=encoded)
        
        encoded_input = Input(shape=(encoding_dim,))
        hidden_layer_1 = autoencoder.layers[-5]
        hidden_layer_2 = autoencoder.layers[-4]
        hidden_layer_3 = autoencoder.layers[-3]
        hidden_layer_4 = autoencoder.layers[-2]
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(inputs=encoded_input, outputs=decoder_layer(
                hidden_layer_4(hidden_layer_3(
                        hidden_layer_2(hidden_layer_1(encoded_input))))))

        autoencoder.compile(optimizer=optimizer, loss=loss)
        return autoencoder, encoder, decoder
    
    def prepare_autoencoder_5hl_multi_encode(
            self, 
            encoding_factor=0.8,
            activations=['relu', 'sigmoid', 'relu', 
                         'sigmoid', 'relu', 'sigmoid'],
            optimizer='adadelta', 
            loss='binary_crossentropy'):
        """
        Generate the Keras layers and model for autoencoder with 5 hidden 
        layers with multi-encoding.
        
        Parameters
        ----------
        encoding_factor : float32, optional 
            The encoding factor for the hidden layers.
            Default: 0.8
        activations : 1D array of str, optional 
            The array of activation functions for hidden layers and 
            output layer.
            Default: ['relu', 'sigmoid', 'relu', 'sigmoid', 'relu', 'sigmoid']
        optimizer : str, optional 
            The optimizer algorithm.
            Default: 'adadelta'
        loss : str, optional 
            The loss function.
            Default: 'binary_crossentropy'
        
        Returns
        -------
        keras.model, keras.model, keras.model
            The Keras models for the autoencoder, the encoder and the decoder
        """
        encoding_dim = int(np.floor(self.dimensions_count * encoding_factor))
        
        input = Input(shape=(self.dimensions_count,))
        encoded = Dense(encoding_dim, activation=activations[0])(input)
        hidden_1 = Dense(
                int(np.floor(encoding_dim * encoding_factor)), 
                activation=activations[1])(encoded)
        hidden_2 = Dense(
                int(np.floor(encoding_dim * encoding_dim * encoding_factor)), 
                activation=activations[2])(hidden_1)
        hidden_3 = Dense(
                int(np.floor(encoding_dim * encoding_factor)), 
                activation=activations[3])(hidden_2)
        hidden_4 = Dense(encoding_dim, activation=activations[4])(hidden_3)
        decoded = Dense(self.dimensions_count, 
                        activation=activations[3])(hidden_4)
        
        autoencoder = Model(inputs=input, outputs=decoded)
        
        encoder = Model(inputs=input, outputs=encoded)
        
        encoded_input = Input(shape=(encoding_dim,))
        hidden_layer_1 = autoencoder.layers[-5]
        hidden_layer_2 = autoencoder.layers[-4]
        hidden_layer_3 = autoencoder.layers[-3]
        hidden_layer_4 = autoencoder.layers[-2]
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(inputs=encoded_input, outputs=decoder_layer(
                hidden_layer_4(hidden_layer_3(
                        hidden_layer_2(hidden_layer_1(encoded_input))))))

        autoencoder.compile(optimizer=optimizer, loss=loss)
        return autoencoder, encoder, decoder
    
    def compute_distance(self):
        """
        Compute the distance of input data objects and output reconstructions 
        according to specified distance function.
        
        Returns
        -------
        1D numpy.array of float32
            The distance for all data objects.
        """
        encoded = self.encoder.predict(self.data_norm.values)
        decoded = self.decoder.predict(encoded)
        dist = np.zeros(len(self.data_norm.values))
        if self.error == "euclid":
            for i, x in enumerate(self.data_norm.values):
                dist[i] = spdist.euclidean(x, decoded[i])
        elif self.error == "max":
            for i, x in enumerate(self.data_norm.values):
                dist[i] = np.amax(x-decoded[i])
        elif self.error == "manhattan":
            for i, x in enumerate(self.data_norm.values):
                dist[i] = spdist.cityblock(x, decoded[i])
        elif self.error == "cosine":
            for i, x in enumerate(self.data_norm.values):
                dist[i] = spdist.cosine(x, decoded[i])
        elif self.error == "correlation":
            for i, x in enumerate(self.data_norm.values):
                dist[i] = spdist.correlation(x, decoded[i])
        elif self.error == "angle":
            for i, x in enumerate(self.data_norm.values):
                dist[i] = np.arccos(np.clip(np.dot((x / np.linalg.norm(x)), 
                    (decoded[i] / np.linalg.norm(decoded[i]))), -1.0, 1.0))
        return dist
    
    def compute_error_per_dim(self, point):
        """
        Compute the reconstruction error per dimension for the given data 
        object and the specified distance function.
        
        Parameters
        ----------
        point : 1D numpy.array of float32
            The data object.
        
        Returns
        -------
        1D numpy.array of float32
            The distance per dimension for the given data object.
        """
        p = np.array(self.data_norm.iloc[point,:]).reshape(1,
                    self.dimensions_count)
        encoded = self.encoder.predict(p)
        decoded = self.decoder.predict(encoded)
        dist = np.array(np.abs((p - decoded)[0]))
        return np.array(dist)
    
    def compute_roc_auc(self):
        """
        Compute the ROC area under the curve, the FPR and the TPR.
        
        Returns
        -------
        float32, float32, float32
            ROC AUC, FPR, TPR.
        """
        fpr, tpr, thresholds = roc_curve(self.labels, self.data['dist'])
        roc_auc = auc(fpr, tpr)
        return roc_auc, fpr, tpr
    
    def compute_pr_auc(self):
        """
        Compute the precision recall (PR) area under the curve, the precision 
        and the recall.
        
        Returns
        -------
        float32, float32, float32
            PR AUC, precision, recall.
       """
        precision, recall, threshold_pr = precision_recall_curve(
                self.labels, 
                self.data['dist'])
        pr_auc = auc(recall, precision)
        return pr_auc, precision, recall

    def plot_roc(self):
        """
        Plot the ROC curve and save as PDF file.
        
        Returns
        -------
        float32
            ROC AUC.
        """
        plt.figure(figsize=(10,6))
        plt.plot(self.fpr, 
                 self.tpr, 
                 color='red', 
                 label='AUC = %0.2f)' % self.roc_auc)
        plt.xlim((0,1))
        plt.ylim((0,1))
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive rate')
        plt.ylabel('True Positive rate')
        plt.title('ROC Autoencoder {}-{}-{} ReLU/Sigmoid {}'.format(
                self.dimensions_count, 
                self.compressed_dimensions, 
                self.dimensions_count, 
                self.filename), fontsize=16)
        plt.legend(loc="lower right")
        #plt.show()
        plt.savefig("ae_outlier_training_roc_{}.pdf".format(self.filename), 
                    format="pdf")
        return self.roc_auc

    def plot_outlier_score(self):
        """
        Plot the outlier scores (reconstruction errors) for all data objects 
        and save as PDF file.
        """
        plt.figure(figsize=(6,3))
        levels = [0,1,2]
        colors = ['#348ABD', '#A60628']
        cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors)
        plt.scatter(self.data.index, 
                    self.data['dist'], 
                    c=self.labels, 
                    cmap=cmap, 
                    edgecolor='black', 
                    s=15)
        plt.xlabel('Index')
        plt.ylabel('Reconstruction error')
        plt.xlim((0,self.objects_count))
        plt.title("Reconstruction error for each data object")
        plt.tight_layout()
        plt.savefig("ae_outlier_scores_{}.pdf".format(self.filename), 
                    format="pdf")
    
    def plot_outlier_score_before_after(self):
        """
        Plot the outlier scores (reconstruction errors) of all data objects 
        before and after training the autoencoder and save as PDF file.
        """
        gs = gridspec.GridSpec(1, 2)
        
        plt.figure(figsize=(10,4))
        ax = plt.subplot(gs[0, 0]) # row 0, col 0
        plt.scatter(self.data.index, 
                    self.data['naivedist'], 
                    c=self.labels, 
                    edgecolor='black', 
                    s=15)
        plt.xlabel("Index")
        plt.ylabel('Score')
        plt.xlim((0,self.objects_count))
        plt.title("Before Learning", fontsize="20")
        
        ax = plt.subplot(gs[0, 1]) # row 0, col 1
        plt.scatter(self.data.index, 
                    self.data['dist'], 
                    c=self.labels, 
                    edgecolor='black', 
                    s=15)
        plt.xlabel('Index')
        plt.ylabel('Score')
        plt.xlim((0,self.objects_count))
        plt.title("After Learning", fontsize="20")
        
        plt.tight_layout()
        plt.savefig("ae_outlier_scores_before_after_{}.pdf".format(
                self.filename), format="pdf")

    def plot_reconstruction_error_for_point(self, point):
        """
        Plot the reconstruction error separately for each dimension of a data 
        object and save as PDF file.
        
        Parameters
        ----------
        point : int
            The data object.
        """
        plt.figure(figsize=(6,3))
        plt.plot(self.compute_error_per_dim(point))
        plt.xlim((0,self.dimensions_count))
        plt.xlabel('Dimension')
        plt.ylabel('Reconstruction error')
        plt.title("Reconstruction error in each dimension for \
                  data object {}".format(point))
        plt.tight_layout()
        plt.savefig("ae_outlier_reconstruction_error_point_\
                    {}_{}.pdf".format(point, self.filename), format="pdf")

    def plot_point_re_per_dimension(self, 
                                    point_index, 
                                    save_file=True, 
                                    plot_re=True, 
                                    plot_data=False):
        """
        Plot the reconstruction error separately for each dimension of a data 
        object and save as PDF file.
        
        Parameters
        ----------
        point : int
            The data object.
        
        Returns
        -------
        1D array of float32
            The reconstruction errors for each dimension for the specified 
            data object.
        """
        i = point_index
        res_per_dim = self.compute_error_per_dim(point_index)
        if save_file:
            with open('ae_res_per_dim_{}_compr_{}epochs_{}_point{}.txt'.format(
                    self.encoding_factor,
                    self.epochs, 
                    self.filename, 
                    i), 'w') as somfile:
                for line in res_per_dim:
                    somfile.write(str(line) + '\n')
        if plot_re:
            plt.figure(figsize=(6,3))
            plt.plot(res_per_dim)
            if plot_data:
                plt.plot(self.data_norm[i], c='g')
            plt.xlim((0,self.dimensions_count))
            plt.xlabel('Dimension')
            plt.ylabel('Reconstruction error')
            plt.title("Reconstruction error in each dimension for \
                      data object {}".format(i))
            plt.savefig("ae_re_per_dim_point_{}_{}.pdg".format(i, 
                        self.filename), format="pdg")
        return res_per_dim
    
    def plot_dataset_3d(self, x, y, z):
        """
        Plot the data set in the three specified dimensions as a 3-dimensional 
        scatter plot and save as PDF file.
        
        Parameters
        ----------
        x : int
            First dimension.
        y : int
            Second dimension.
        z : int
            Third dimension.
        """
        self.data.loc[self.data.index.isin([24, 52, 64]),'labels'] = 1
        
        threedee = plt.figure().gca(projection='3d')
        threedee.scatter(self.data[x], self.data[y], zs=self.data[z], 
                         c=self.labels, edgecolor='black')
        threedee.set_xlabel('x - dim {}'.format(x), labelpad=10)
        threedee.set_ylabel('y - dim {}'.format(y), labelpad=10)
        threedee.set_zlabel('z - dim {}'.format(z), labelpad=10)
        plt.tight_layout()
        plt.savefig("ae_3d_plot_outlier_dim_{}-{}-{}_{}.pdf".format(
                x,y,z,self.filename), format="pdf")

    def plot_dataset_2d_for_3_dimensions(self, x, y, z):
        """
        Plot the data set in three 2-dimensional projections of the three 
        specified dimensions
        three 2-dimensional scatter subplots and save as PDF file.
        
        Parameters
        ----------
        x : int
            First dimension.
        y : int
            Second dimension.
        z : int
            Third dimension.
        """
        gs = gridspec.GridSpec(1, 3, wspace=0.4)
        
        plt.figure(figsize=(6,2.5))
        
        ax = plt.subplot(gs[0, 0]) # row 0, col 0
        plt.scatter(self.data[x], self.data[y], c=self.labels, 
                    edgecolor='black', s=15)
        plt.xlabel('x - dim {}'.format(x))
        plt.ylabel('y - dim {}'.format(y))
        
        ax = plt.subplot(gs[0, 1]) # row 0, col 1
        plt.scatter(self.data[y], self.data[z], c=self.labels,
                   edgecolor='black', s=15)
        plt.xlabel('y - dim {}'.format(y))
        plt.ylabel('z - dim {}'.format(z))
        
        ax = plt.subplot(gs[0, 2]) # row 0, col 3
        plt.scatter(self.data[x], self.data[z], c=self.labels,
                   edgecolor='black', s=15)
        plt.xlabel('x - dim {}'.format(x))
        plt.ylabel('z - dim {}'.format(z))
        
        plt.savefig("ae_2d_plot_outlier_dim{}-{}-{}_{}.pdf".format(
                x,y,z,self.filename), format="pdf", bbox_inches='tight')
