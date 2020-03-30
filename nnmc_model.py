
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import random
from utils import plt_outer_loop
from utils import plt_inner_loop

class nnmc_model(object):
    
    def __init__(self, reg = 0.1, hidden = 5):
        
        # System configuration
        self.outer_iteration = 50 # Number of iterations for outer loop
        self.outer_learning_rate = 0.1 # Learning rate for outer loop
        
        self.min_iter_nn_train = 400 # Number of epochs for nn train at first outer iteration
        self.second_pass = 50 # Number of epochs for NN train after first outer iteration
        self.step_size = 0.1 # NN train SGD step size
        self.neural_net_store_on_off = 1 # Enable weight store and transfer mechanism
        self.shuffle_before_iterate = 1 # Shuffle before every NN train
        
        # Verbose options
        self.plot_inner_loop_on_off = 0
        self.plot_inner_loop_feature_no = 0
        self.plot_outer_loop_on_off = 0	
        self.disp = 5
        
        # User-controlled config
        self.reg = reg	# Regularization strength
        self.no_of_hidden_layers = hidden	# Number of neurons in hidden layer

    class weight:
        def __init__(self, n_inp, h, n_out):
            self.W = 0.01 * np.random.randn(n_inp, h)
            self.b = np.zeros((1,h))
            self.W2 = 0.01 * np.random.randn(h,n_out)
            self.b2 =np.zeros((1,n_out))

    def feature_model_initialize(self, D, h):
        
        init_weight_matrix = []
        for i in range(0, D):
            init_weight = self.weight(D-1, h, 1)
            init_weight_matrix.append(init_weight)
        return init_weight_matrix;

    def mc_complete(self, X_train, # Incomplete matrix (missing values filled zeros)
                          X_mask, # Binary matrix size of X_train (0 indicates missing value)
                          X_miss): # Ground truth matrix with missing entries (Used for verbose only))
        
        h = self.no_of_hidden_layers
        D = X_train.shape[1]

        updated_weight_matrix = []
        MSE_function = []
            
        X_est_final = X_train.copy()
            
        for j in range(0, self.outer_iteration):
                
            X_temp = X_est_final.copy()
        
            if ((j % self.disp) == 0):
                print('PROCESS: {:d}. outer iteration started' .format(j+1) )
                
            if j  == 0:
                input_weight_matrix = self.feature_model_initialize (D, h)
            else:
                if self.neural_net_store_on_off == 1:
                    input_weight_matrix = updated_weight_matrix.copy()
                else:
                    input_weight_matrix = self.feature_model_initialize (D, h)
            
            updated_weight_matrix, updated_predictions = self.mc_train(X_train = X_temp,
                                                                       input_weight_matrix = input_weight_matrix,
                                                                       iteration_no = j,
                                                                       D = D)
            X_est = updated_predictions.copy()
            
            Miss_ind = np.where(X_mask == 0)
            
            Residual = X_est[Miss_ind] - X_est_final[Miss_ind]
            
            X_est_final[Miss_ind] = X_est_final[Miss_ind] + self.outer_learning_rate * Residual

            MSE = np.sum(np.power((X_est_final[Miss_ind] - X_miss[Miss_ind]), 2)) / Miss_ind[0].size
            MSE_function = np.append(MSE_function, MSE);
    
        if self.plot_outer_loop_on_off == 1:
            plt_outer_loop(MSE_function)
    
        return X_est_final;

        
    def mc_train(self, X_train, input_weight_matrix, iteration_no, D):
            
        X_temp = X_train.copy()
        
        #Shuffling
        shfl = list(range(0, X_train.shape[0]))
        random.shuffle(shfl)
        if self.shuffle_before_iterate == 1:
            X_temp_shfl = X_temp[shfl]
        else:
            X_temp_shfl = X_temp.copy()
        
        updated_weight_matrix = input_weight_matrix.copy()
        updated_predictions = X_train.copy()
        
        if iteration_no  == 0:
            inner_iteration = self.min_iter_nn_train
        else:
            if self.neural_net_store_on_off == 1:
                inner_iteration = self.second_pass;
        
        for i in range(0, D):
    
            inp = X_temp_shfl.copy()
            inp = np.delete(inp, i, 1)
            inp = np.reshape(inp, (inp.shape[0], inp.shape[1]))
                
            target = X_temp_shfl[:, i][None].T
            
            inp_test = X_temp.copy()
            inp_test = np.delete(inp_test, i, 1)
            inp_test = np.reshape(inp_test, (inp_test.shape[0], inp_test.shape[1]))
            
            train_weight, loss_function, val_error_function, pred = self.mc_nn_regression(inp,
                                                                                       target,
                                                                                       inp_test,
                                                                                       input_weight_matrix[i],
                                                                                       inner_iteration,
                                                                                       self.reg,
                                                                                       self.step_size)
            
            updated_weight_matrix[i] = train_weight
            updated_predictions[:, i] = pred[:,0]
            
            if self.plot_inner_loop_on_off == 1:
                if i == self.plot_inner_loop_feature_no:
                    plt_inner_loop(val_error_function, loss_function, i)
            
        return updated_weight_matrix, updated_predictions;

    def mc_nn_regression(self, inp, target, inp_test, init_weight, inner_iteration, reg, step_size):
        
            n_inp = inp.shape[1]
            n_out = target.shape[1]
            W_init = init_weight.W.copy()
            b_init = init_weight.b.copy()
            W2_init = init_weight.W2.copy()
            b2_init = init_weight.b2.copy()
            h = b2_init.shape[1]
    
            loss_function = []
            val_error_function = []
    
            graph = tf.Graph()
            with graph.as_default():
                x = tf.placeholder(tf.float64, [None, n_inp])
                y_ = tf.placeholder(tf.float64, [None, n_out])
                
                W = tf.Variable(W_init)
                b = tf.Variable(b_init)
                W2 = tf.Variable(W2_init)
                b2 = tf.Variable(b2_init)
    
                y1 = tf.matmul(x, W) + b
                relu_layer= tf.nn.relu(y1)
                y = tf.matmul(relu_layer, W2) + b2
    
                loss = tf.reduce_mean(tf.square(tf.subtract(y_, y)))
                regularizer = tf.nn.l2_loss(W) + tf.nn.l2_loss(W2)
                loss = tf.reduce_mean(loss + reg * regularizer)
    
                train_step = tf.train.GradientDescentOptimizer(step_size).minimize(loss)
    
            with tf.Session(graph = graph, config=tf.ConfigProto(log_device_placement=False)) as sess:
                sess.run(tf.global_variables_initializer())
                for i in range(inner_iteration):
                    
                      feed_dict={x: inp, y_: target}
                      _, loss_value = sess.run([train_step, loss], feed_dict=feed_dict)
                      loss_function = np.append(loss_function, loss_value);
    
                train_weight = self.weight(n_inp, h, n_out)
                train_weight.W = sess.run(W);
                train_weight.b = sess.run(b);
                train_weight.W2 = sess.run(W2);
                train_weight.b2 = sess.run(b2);
                
                pred = sess.run(y, feed_dict={x: inp_test})
            
            return train_weight, loss_function, val_error_function, pred;   
    
