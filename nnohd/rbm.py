import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, log_loss

from utils.preprocessing import Preprocessor
import utils.rbm_utils as utils

class RBM(object):
    """ 
    Restricted Boltzmann Machine implementation for outlier detection using 
    TensorFlow.
    
    The implementation generates the RBM and performs training on the 
    specified data set within initialization to facilitate batch runs for 
    different data sets and SOM parameter sets.
    
    Parameters
    ----------
    visible : int, optional 
        The number of visible units. If 0, the number of dimensions of the 
        data set is used.
        Default: 0
    hidden : int or float32
        Optional parameter for the number of hidden units. If between 
        0 < hidden < 1, the number of hidden neurons is hidden*dimensions. 
        If hidden>=1, the number of hidden neurons equal this value. 
        If hidden=0, the number of hidden neurons is 0.5*dimensions.
        Default: 0
    visible_unit_type : {'bin', 'gauss'}, optional 
        The type of the visible neurons
            * 'bin': Binary neurons (default)
            * 'gauss': Gaussian neurons
    main_dir : str, optional 
        The main directory to put the models, data and summary directories.
        Default: 'rbm/unsup_model'
    model_name : str, optional 
        The name of the model, used to save data.
        Default: 'rbm_model_unsup'
    gibbs_sampling_steps : int, optional 
        The number of Gibbs sampling stepts for contrastive divergence.
        Default: 4
    learning_rate : float32, optional 
        The learning rate.
        Default: 0.01
    momentum : float32, optional 
        Momentum for gradient descent.
        Default: 0.9
    l2 : float32, optional 
        The l2 weight decay.
        Default: 0.001
    batch_size : int, optional 
        The training batch size.
        Default: 100
    epochs : int, optional 
        The number of training epochs.
        Default: 10
    test_ratio : float32, optional 
        Specifies the percentage of the data set that is used for training.
        Default: 0.5
    stddev : float32, optional 
        The standard deviation for Gaussian neurons. Ignored if 
        visible_unit_type is not 'gauss'.
        Default: 0.1
    verbose : int, optional 
        The level of verbosity.
        Default: 0
    plot_training_loss : bool, optional 
        Specifies whether or not to plot training loss.
        Default False
    delimiter : str, optional 
        The column delimiter for data sets loaded from CSV files.
        Default: ','.
    """

    def __init__(self, 
                 filename, 
                 filepath="datasets", 
                 visible=0, 
                 hidden=0, 
                 visible_unit_type='gauss', 
                 main_dir='rbm/unsup_model', 
                 model_name='rbm_model_unsup', 
                 gibbs_sampling_steps=4, 
                 learning_rate=0.01, 
                 momentum = 0.9, 
                 l2 = 0.001, 
                 batch_size=100, 
                 epochs=10, 
                 test_ratio=0.5, 
                 stddev=0.1, 
                 verbose=0, 
                 plot_training_loss=False, 
                 delimiter=',', 
                 with_header=False):
        """
        Constructor for the RBM class.
        """
        self.filename = filename
        self.filepath = filepath
        file = "{}/{}".format(filepath,filename)
        input_data = Preprocessor(file, 
                                  delimiter=delimiter, 
                                  with_header=with_header)
        self.data = input_data.data
        self.data_norm = input_data.data_normalized
        self.data_v = input_data.data.values
        self.data_norm_v = input_data.data_normalized.values
        self.labels = input_data.labels.values
        self.objects_count = input_data.objects_count
        self.inlier_count = input_data.inlier_count
        self.outlier_count = input_data.outlier_count
        self.dimensions_count = input_data.dimensions_count
        
        self.data_gauss = self.data_norm_v
        
        TEST_RATIO = test_ratio
        TRA_INDEX = int((1-TEST_RATIO) * self.data_norm.shape[0])
        train_x = self.data_norm.iloc[:TRA_INDEX, :].values
        self.train_y = self.labels[:TRA_INDEX]
        test_x = self.data_norm.iloc[TRA_INDEX:, :].values
        self.test_y = self.labels[TRA_INDEX:]
        
        cols_mean = []
        cols_std = []
        for c in range(train_x.shape[1]):
            cols_mean.append(train_x[:,c].mean())
            cols_std.append(train_x[:,c].std())
            train_x[:, c] = (train_x[:, c] - cols_mean[-1]) / cols_std[-1]
            test_x[:, c] =  (test_x[:, c] - cols_mean[-1]) / cols_std[-1]
            self.data_gauss[:, c] = (
                    self.data_gauss[:, c] - cols_mean[-1] / cols_std[-1])
        
        if visible == 0:
            self.visible = self.dimensions_count
        else:
            self.visible = visible
        if hidden == 0:
            self.hidden = int(self.dimensions_count * 0.5)
        elif hidden < 1:
            self.hidden = int(hidden * visible)
        else:
            self.hidden = hidden
        self.visible_unit_type = visible_unit_type
        self.main_dir = main_dir
        self.model_name = model_name
        self.gibbs_sampling_steps = gibbs_sampling_steps
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.l2 = l2
        self.batch_size = batch_size
        self.epochs = epochs
        self.stddev = stddev
        self.verbose = verbose

        self._create_model_directory()
        self.model_path = os.path.join(self.main_dir, self.model_name)
        self.plot_training_loss = plot_training_loss

        self.W = None
        self.bh_ = None
        self.bv_ = None
        self.dw = None
        self.dbh_ = None 
        self.dbv_ = None

        self.w_upd8 = None
        self.bh_upd8 = None
        self.bv_upd8 = None

        self.encode = None
        self.recontruct = None

        self.loss_function = None
        self.batch_cost = None
        self.batch_free_energy = None

        self.training_losses = []

        self.input_data = None
        self.hrand = None
        self.validation_size = None

        self.tf_session = None
        self.tf_saver = None

        self.fit(train_x, validation_set=test_x)
        
        self.fe = self.getFreeEnergy(self.data_gauss).reshape(-1)
        self.re = self.getReconstructError(self.data_gauss).reshape(-1)
        self.fpr, self.tpr, self.thresholds = roc_curve(self.labels, self.fe)
        self.roc_auc = auc(self.fpr, self.tpr)
        self.precision, self.recall, self.threshold_pr = \
            precision_recall_curve(self.labels, self.fe)
        self.pr_auc = auc(self.recall, self.precision)
        self.f1_scores = 2 * (self.precision * self.recall) \
            / (self.precision + self.recall)
        self.f1_score = np.max(np.nan_to_num(self.f1_scores))
        #self.logloss = log_loss(self.labels, self.fe)


    def fit(self, 
            train_set, 
            validation_set=None, 
            restore_previous_model=False):
        """
        Fit the model to the training data.
        
        Parameters
        ----------
        train_set : 2D numpy.array of float32
            The training set to fit the model to.
        validation_set : 2D numpy.array of float32
            Optional parameter for the validation set.
            Default: None
        restore_previous_model : bool, optional 
            Specifies whether or not to load a previously trained model with the 
            same name of this model is restored from disk to continue training.
            Default: False
        """
        if validation_set is not None:
            self.validation_size = validation_set.shape[0]

        tf.reset_default_graph()

        self._build_model()

        with tf.Session() as self.tf_session:

            self._initialize_tf_utilities_and_ops(restore_previous_model)
            self._train_model(train_set, validation_set)
            self.tf_saver.save(self.tf_session, self.model_path)

            if self.plot_training_loss:
                plt.plot(self.training_losses)
                plt.title("Training batch losses v.s. iteractions")
                plt.xlabel("Num of training iteractions")
                plt.ylabel("Reconstruction error")
                plt.show()

    def _initialize_tf_utilities_and_ops(self, restore_previous_model):
        """
        Internal function to initialize TensorFlow operations: summaries, 
        init operations, saver, summary_writer.
        
        Parameters
        ----------
        restore_previous_model : bool
            If True, restore a previously trained model.
        """
        init_op = tf.global_variables_initializer()
        self.tf_saver = tf.train.Saver()

        self.tf_session.run(init_op)

        if restore_previous_model:
            self.tf_saver.restore(self.tf_session, self.model_path)

    def _train_model(self, train_set, validation_set):
        """
        Train the model on training data set and optionally use validation set.
        
        Parameters
        ----------
        train_set : 2D numpy.array of float32
            The training data set.
        validation_set : 2D numpy.array of float32, optional
            The validation set.
            Default: None
        """
        for i in range(self.epochs):
            self._run_train_step(train_set)

            if validation_set is not None:
                self._run_validation_error(i, validation_set)

    def _run_train_step(self, train_set):
        """
        Internal function to run a training step. A training step is made by 
        randomly shuffling the training set, dividing it into batches and 
        running the variable update nodes for each batch. If 
        self.plot_training_loss is true, training loss is shown after each 
        batch. 
        
        Parameters
        ----------
        train_set : 2D numpy.array of float32
            The training data set.
        """
        np.random.shuffle(train_set)

        batches = [_ for _ in utils.gen_batches(train_set, self.batch_size)]
        updates = [self.w_upd8, self.bh_upd8, self.bv_upd8]


        for batch in batches:
            if self.plot_training_loss:
                _, loss = self.tf_session.run(
                        [updates, self.loss_function], 
                        feed_dict=self._create_feed_dict(batch))
                self.training_losses.append(loss)
            else:
                self.tf_session.run(updates, 
                                    feed_dict=self._create_feed_dict(batch))

    def _run_validation_error(self, epoch, validation_set):
        """
        Internal function to run the error computation on the validation set 
        and print it out for each epoch. 
        
        Parameters
        ----------
        epoch : int
            The number of the current epoch.
        validation_set : 2D numpy.array of float32, optional
            The validation set.
        """
        loss = self.tf_session.run(
                self.loss_function, 
                feed_dict=self._create_feed_dict(validation_set))

        if self.verbose == 1:
            print("Validation cost at step %s: %s" % (epoch, loss))

    def _create_feed_dict(self, data):
        """
        Internal function to create the dictionary of data to feed to 
        TensorFlow's session during training.
        
        Parameters
        ----------
        data : 2D numpy.array of float32
            The training or validation data set batch.
        
        Returns
        -------
        {1D numpy.array of float32: 1D numpy.array of float32}
            dictionary(self.input_data: data, self.hrand: random_uniform)
        """
        return {
            self.input_data: data,
            self.hrand: np.random.rand(data.shape[0], self.hidden),
        }

    def _build_model(self):
        """
        Build the Restricted Boltzmann Machine model in TensorFlow.
        """
        self.input_data, self.hrand = self._create_placeholders()
        self.W, self.bh_, self.bv_, self.dw, self.dbh_, self.dbv_ = \
            self._create_variables()

        hprobs0, hstates0, vprobs, hprobs1, hstates1 = \
            self.gibbs_sampling_step(self.input_data)
        positive = self.compute_positive_association(self.input_data, 
                                                     hprobs0, 
                                                     hstates0)

        nn_input = vprobs

        for step in range(self.gibbs_sampling_steps - 1):
            hprobs, hstates, vprobs, hprobs1, hstates1 = \
                self.gibbs_sampling_step(nn_input)
            nn_input = vprobs

        self.recontruct = vprobs

        negative = tf.matmul(tf.transpose(vprobs), hprobs1)

        self.encode = hprobs1  # encoded data, used by the transform method

        dw = positive - negative
        self.dw = self.momentum*self.dw + (1-self.momentum)*dw
        self.w_upd8 = self.W.assign_add(self.learning_rate*self.dw \
                                        - self.learning_rate*self.l2*self.W)

        dbh_ = tf.reduce_mean(hprobs0 - hprobs1, 0)
        self.dbh_ = self.momentum*self.dbh_ + self.learning_rate*dbh_
        self.bh_upd8 = self.bh_.assign_add(self.dbh_)

        dbv_ = tf.reduce_mean(self.input_data - vprobs, 0)
        self.dbv_ = self.momentum*self.dbv_ + self.learning_rate*dbv_
        self.bv_upd8 = self.bv_.assign_add(self.dbv_)

        self.loss_function = tf.sqrt(tf.reduce_mean(tf.square(
                self.input_data - vprobs)))

        self.batch_cost = tf.sqrt(tf.reduce_mean(tf.square(
                self.input_data - vprobs), 1))
   
        self._create_free_energy_for_batch()
        
    def _create_free_energy_for_batch(self):
        """
        Create free energy TensorFlow operations to batch input data.
        """
        if self.visible_unit_type == 'bin':
            self._create_free_energy_for_bin()    
        elif self.visible_unit_type == 'gauss':
            self._create_free_energy_for_gauss()
        else:
            self.batch_free_energy = None

    def _create_free_energy_for_bin(self):
        """
        Create free energy for model with binary visible layer.
        """
        self.batch_free_energy = - (tf.matmul(
                self.input_data, 
                tf.reshape(self.bv_, [-1, 1])) + tf.reshape(
                        tf.reduce_sum(
                                tf.log(
                                        tf.exp(
                                                tf.matmul(self.input_data, 
                                                          self.W) + self.bh_) \
                                                + 1), 1), [-1, 1]))

    def _create_free_energy_for_gauss(self):
        """
        Create free energy for model with Gaussian visible layer .
        """
        self.batch_free_energy = - (tf.matmul(
                self.input_data, 
                tf.reshape(self.bv_, [-1, 1])) - tf.reshape(
                        tf.reduce_sum(
                                0.5 * self.input_data * self.input_data, 1), 
                                [-1, 1]) + tf.reshape(
                                        tf.reduce_sum(tf.log(tf.exp(
                                                tf.matmul(self.input_data, 
                                                          self.W) + self.bh_) \
                                + 1), 1), [-1, 1]))

    def _create_placeholders(self):
        """
        Create the TensorFlow placeholders for the model.
        
        Returns
        -------
        (2D array of float, 2D array of float)
            Tuple of visible neurons (shape(None, visible)), 
            hidden neurons (shape(None, hidden)))
        """
        x = tf.placeholder('float', [None, self.visible], name='x-input')
        hrand = tf.placeholder('float', [None, self.hidden], name='hrand')

        return x, hrand

    def _create_variables(self):
        """
        Create the TensorFlow variables for the model.
        
        Returns
        -------
        (2D array of float, 1D array of float, 1D array of float)
            Tuple of weights (shape(visible, hidden)),
                           hidden bias (shape(hidden)),
                           visible bias (shape(visible)))
        """
        W = tf.Variable(tf.random_normal((self.visible, self.hidden), 
                                         mean=0.0, 
                                         stddev=0.01), name='weights')
        dw = tf.Variable(tf.zeros([self.visible, self.hidden]), 
                         name = 'derivative-weights')

        bh_ = tf.Variable(tf.zeros([self.hidden]), name='hidden-bias')
        dbh_ = tf.Variable(tf.zeros([self.hidden]), 
                           name='derivative-hidden-bias')

        bv_ = tf.Variable(tf.zeros([self.visible]), name='visible-bias')
        dbv_ = tf.Variable(tf.zeros([self.visible]), 
                           name='derivative-visible-bias')

        return W, bh_, bv_, dw, dbh_, dbv_

    def gibbs_sampling_step(self, visible):
        """
        Performs one step of gibbs sampling.
        
        Parameters
        ----------
        visible : 2D array of float32
            Activations of the visible neurons.
        
        Returns
        -------
        (2D array of float, 2D array of float, 2D array of float,
         2D array of float, 2D array of float)
            Tuple of hidden probabilities, hidden states, visible 
            probabilities, new hidden probabilities and new hidden states.
        """
        hprobs, hstates = self.sample_hidden_from_visible(visible)
        vprobs = self.sample_visible_from_hidden(hprobs)
        hprobs1, hstates1 = self.sample_hidden_from_visible(vprobs)
        #print("Hprobs: {} - Vprobs: {}".format(hprobs,vprobs))

        return hprobs, hstates, vprobs, hprobs1, hstates1

    def sample_hidden_from_visible(self, visible):
        """
        Sample the hidden units from the visible units.
        This is the Positive phase of the Contrastive Divergence algorithm.
        
        Parameters
        ----------
        visible : 2D array of float32
            Activations of the visible neurons.
        
        Returns
        -------
        (tensor, tensor)
            Tuple of hidden probabilities and hidden binary states
        """
        hprobs = tf.nn.sigmoid(tf.matmul(visible, self.W) + self.bh_)
        hstates = utils.sample_prob(hprobs, self.hrand)

        return hprobs, hstates

    def sample_visible_from_hidden(self, hidden):
        """
        Sample the visible units from the hidden units.
        This is the Negative phase of the Contrastive Divergence algorithm.
        
        Parameters
        ----------
        hidden : 2D array of float32
            Activations of the hidden neurons.
        
        Returns
        -------
        tensor
            Visible probabilities
        """
        visible_activation = tf.matmul(hidden, tf.transpose(self.W)) + self.bv_

        if self.visible_unit_type == 'bin':
            vprobs = tf.nn.sigmoid(visible_activation)

        elif self.visible_unit_type == 'gauss':
            vprobs = tf.truncated_normal((1, self.visible), 
                                         mean=visible_activation, 
                                         stddev=self.stddev)
        else:
            vprobs = None

        return vprobs

    def compute_positive_association(self, 
                                     visible, 
                                     hidden_probs, 
                                     hidden_states):
        """ 
        Compute positive associations between visible and hidden units.
        
        Parameters
        ----------
        visible : 1D array of float32
            The visible units.
        hidden_probs : 1D array of float32
            The hidden units probabilities.
        hidden_states : 1D array of float32
            The hidden units states.
        
        Returns
        -------
        tensor
            positive association = dot(visible.T, hidden)
        """
        if self.visible_unit_type == 'bin':
            positive = tf.matmul(tf.transpose(visible), hidden_states)

        elif self.visible_unit_type == 'gauss':
            positive = tf.matmul(tf.transpose(visible), hidden_probs)

        else:
            positive = None

        return positive

    def _create_model_directory(self):
        """
        Create the directory for storing the model.
        """
        if not os.path.isdir(self.main_dir):
            print("Created dir: ", self.main_dir)
            os.mkdir(self.main_dir)

    def getReconstructError(self, data):
        """
        Compute the reconstruction error or loss from data objects in batch.
        
        Parameters
        ----------
        data : 2D numpy.array. of float32
            Input data of shape num_samples x visible_size
        
        Returns
        -------
        1D array of float32.
            Reconstruction error for each data object in the batch
        """
        with tf.Session() as self.tf_session:

            self.tf_saver.restore(self.tf_session, self.model_path)

            batch_loss = self.tf_session.run(
                    self.batch_cost, 
                    feed_dict=self._create_feed_dict(data))
            return batch_loss

    def getFreeEnergy(self, data):
        """
        Compute the free energy for the data objects in batch.
        
        Parameters
        ----------
        data : 2D numpy.array of float32
            Data set batch to compute the free energy on.
        
        Returns
        -------
        1D array of float32.
            Free energy for each data object x in the batch as p(x)
        """
        with tf.Session() as self.tf_session:

            self.tf_saver.restore(self.tf_session, self.model_path)

            batch_FE = self.tf_session.run(
                    self.batch_free_energy, 
                    feed_dict=self._create_feed_dict(data))

            return batch_FE

    def getRecontruction(self, data):
        """
        Compute the reconstruction of the data objects by the RBM.
        
        Parameters
        ----------
        data : 2D numpy.array of float32
            Data set to compute the reconstruction on.
        
        Returns
        -------
        1D array of float32.
            Reconstruction for each data object.
        """
        with tf.Session() as self.tf_session:
            
            self.tf_saver.restore(self.tf_session, self.model_path)

            batch_reconstruct = self.tf_session.run(
                    self.recontruct, 
                    feed_dict=self._create_feed_dict(data))

            return batch_reconstruct

    def load_model(self, shape, gibbs_sampling_steps, model_path):
        """
        Load a trained model from disk. The shape of the model 
        (visible, hidden) and the number of gibbs sampling steps must be known 
        in order to restore the model.
        
        Parameters
        ----------
        shape : (int, int)
            The shape of the RBM as a tuple of the number of visible and 
            hidden neurons.
        gibbs_sampling_steps : int
            The number of Gibbs samling steps.
        model_path : str
            The path for the saved model.
        """
        self.visible, self.hidden = shape[0], shape[1]
        self.gibbs_sampling_steps = gibbs_sampling_steps
        
        tf.reset_default_graph()

        self._build_model()

        init_op = tf.global_variables_initializer()
        self.tf_saver = tf.train.Saver()

        with tf.Session() as self.tf_session:

            self.tf_session.run(init_op)
            self.tf_saver.restore(self.tf_session, model_path)

    def get_model_parameters(self):
        """
        Return the model parameters in the form of numpy arrays.
        
        Returns
        -------
        model parameters
        """
        with tf.Session() as self.tf_session:

            self.tf_saver.restore(self.tf_session, self.model_path)

            return {
                'W': self.W.eval(),
                'bh_': self.bh_.eval(),
                'bv_': self.bv_.eval()
            }

