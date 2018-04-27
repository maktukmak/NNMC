
import numpy as np
import tensorflow as tf
import random
from visualize import plt_outer_loop
from visualize import plt_inner_loop

class weight:
    def __init__(self, n_inp, h, n_out):
        self.W = 0.01 * np.random. randn(n_inp, h)
        self.b = np.zeros((1,h))
        self.W2 = 0.01 * np.random.randn(h,n_out)
        self.b2 =np.zeros((1,n_out))

def feature_model_initialize (Mn, h):
    
    init_weight_matrix = []
    for i in range(0, len(Mn)):
        init_weight = weight(np.sum(Mn)-Mn[i], h, Mn[i])
        init_weight_matrix.append(init_weight)
    return init_weight_matrix;

def multimodal_alg (X_train, Y, X_miss, Mn, opt_params,fixed_params):
    
    h = opt_params.no_of_hidden_layers
    iteration_alg = fixed_params.iteration_outer_loop
    
    model_parameters = []
    
    XNN_out = np.zeros(X_train.shape)
    
    update_value_vec = []
    updated_weight_matrix = []
    MSE_function = []
    CE_function = []
        
    X_est_final = X_train.copy()
        
    for j in range(0, iteration_alg):
            
        XNN_temp = X_est_final.copy()
    
        if ((j % 5) == 0):
            print('PROCESS: {:d}. outer iteration started' .format(j+1) )
            
        
            
        if j  == 0:
            input_weight_matrix = feature_model_initialize (Mn, h)
        else:
            if fixed_params.neural_net_store_on_off == 1:
                input_weight_matrix = updated_weight_matrix.copy()
            else:
                input_weight_matrix = feature_model_initialize (Mn, h)
        
        updated_weight_matrix, updated_predictions = multimodal_model_train (XNN_temp, Y, input_weight_matrix, j, Mn, opt_params, fixed_params)
        X_est = updated_predictions.copy()
        
        miss = np.where(Y == 0)
        
        update_value = X_est[miss] - X_est_final[miss]
        
        X_est_final[miss] = X_est_final[miss] + fixed_params.outer_learning_rate * update_value
        
        Dg = len(np.transpose(np.where(Mn == 1)))
        miss_g = np.where(Y[:, 0:Dg] == 0)
        MSE = np.sum(np.power((X_est_final[miss_g] - X_miss[miss_g]), 2)) / miss_g[0].size
        MSE_function = np.append(MSE_function, MSE);

        pred = []
        gt = []
        for i in range(0, len(Mn)):
            if Mn[i] > 1:
                miss_cat = np.where(Y[:,sum(Mn[0:i])] == 0)
                pred = np.append(pred, X_est_final[np.ix_(miss_cat[0], range(sum(Mn[0:i]),sum(Mn[0:i+1])))])
                gt = np.append(gt, X_miss[np.ix_(miss_cat[0], range(sum(Mn[0:i]),sum(Mn[0:i+1])))])
                
                CE_function = np.append(CE_function,  log_loss(gt, pred))
        
        
        update_value_vec = np.append(update_value_vec, np.mean(abs(update_value)))
        update_value_vec = np.convolve(update_value_vec, np.ones((10,))/10, mode='valid')
        
    
    model_parameters = np.append(model_parameters, updated_weight_matrix)
    
    
    for i in range(0, len(Mn)):
        if Mn[i] > 1:
            tmp2 = np.zeros((len(X_est_final), Mn[i]))
            if Mn[i] > 1:
                tmp = np.argmax(X_est_final[:,range(sum(Mn[0:i]),sum(Mn[0:i+1]))], axis = 1)
                tupl = (np.arange(len(X_est_final)), tmp)
                tmp2[tupl] = 1
                X_est_final[:,range(sum(Mn[0:i]),sum(Mn[0:i+1]))] = tmp2
            else:
                tmp =np.where(X_est_final[:, range(sum(Mn[0:i]),sum(Mn[0:i+1]))] >= 0.5)
                X_est_final[:, range(sum(Mn[0:i]),sum(Mn[0:i+1]))] = 0
                X_est_final[tmp[0],range(sum(Mn[0:i]),sum(Mn[0:i+1]))] = 1
                
    if fixed_params.plot_outer_loop_on_off == 1:
        plt_outer_loop (MSE_function, CE_function)

    XNN_out = np.copy(X_est_final)
    
    return XNN_out;

        
def multimodal_model_train (X_train, Y, input_weight_matrix, iteration_no, Mn, opt_params, fixed_params):
    

    XNN_temp = X_train.copy()
    
    #Shuffling
    shfl = list(range(0, X_train.shape[0]))
    random.shuffle(shfl)
    
    if fixed_params.shuffle_before_iterate == 1:
        Y_shfl = Y[shfl]
        XNN_temp_shfl = XNN_temp[shfl]
    else:
        Y_shfl = Y.copy()
        XNN_temp_shfl = XNN_temp.copy()
    
    updated_weight_matrix = input_weight_matrix.copy()
    updated_predictions = X_train.copy()
    
    if iteration_no  == 0:
        tf_inner_iteration = fixed_params.min_iter_nn_train
    else:
        if fixed_params.neural_net_store_on_off == 1:
            tf_inner_iteration = fixed_params.tf_second_pass;
    
    for i in range(0, len(Mn)):

        z1 = XNN_temp_shfl.copy()
        z1 = np.delete(z1, range(sum(Mn[0:i]),sum(Mn[0:i+1])), 1)
        z1 = np.reshape(z1, (z1.shape[0], z1.shape[1]))
            
        z2 = XNN_temp_shfl[:, range(sum(Mn[0:i]),sum(Mn[0:i+1]))]
        z2 = np.reshape(z2, (z2.shape[0], z2.shape[1]))
        
        z1_test = XNN_temp.copy()
        z1_test = np.delete(z1_test, range(sum(Mn[0:i]),sum(Mn[0:i+1])), 1)
        z1_test = np.reshape(z1_test, (z1_test.shape[0], z1_test.shape[1]))
        
        if Mn[i] == 1:
            train_weight, loss_function, val_error_function, y = mc_nn_regression(z1, z2, z1_test, input_weight_matrix[i], tf_inner_iteration, opt_params, fixed_params)
        else:
            train_weight, loss_function, val_error_function, y = mc_nn_classification(z1, z2, z1_test, input_weight_matrix[i], tf_inner_iteration, opt_params, fixed_params)
        
        updated_weight_matrix[i] = train_weight
        updated_predictions[:, range(sum(Mn[0:i]),sum(Mn[0:i+1]))] = y
        
        if fixed_params.plt_inner_loop_on_off == 1:
            if i == fixed_params.plot_inner_loop_feature_no:
                plt_inner_loop(val_error_function, loss_function, i)
        
    return updated_weight_matrix, updated_predictions;
        
def feature_model_exploit (X_inp, weight_matrix, iteration_no, Mn):
      
    X_est = np.zeros(X_inp.shape)
    
    for i in range(0, len(Mn)):
        
        XNN_temp = X_inp.copy()
        XNN_temp = np.delete(XNN_temp, range(sum(Mn[0:i]),sum(Mn[0:i+1])), 1)
        XNN_temp = np.reshape(XNN_temp, (XNN_temp.shape[0], XNN_temp.shape[1]))
        train_weight = weight_matrix[i]

        hidden_layer = np.maximum(0, np.dot(XNN_temp, train_weight.W) + train_weight.b) # note, ReLU activation
        est = np.dot(hidden_layer, train_weight.W2) + train_weight.b2         
        
        est = np.reshape(est, (est.shape[0]))
        
        X_est[:, i] = est
        
    return X_est;

def mc_nn_regression(z1, z2, z1_test, init_weight, tf_inner_iteration, opt_params, fixed_params):
    
        n_inp = z1.shape[1]
        n_out = z2.shape[1]
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
            loss = tf.reduce_mean(loss + opt_params.reg * regularizer)

            train_step = tf.train.GradientDescentOptimizer(fixed_params.step_size).minimize(loss)

        with tf.Session(graph = graph, config=tf.ConfigProto(log_device_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(tf_inner_iteration):
                
                  feed_dict={x: z1, y_: z2}
                  _, loss_value = sess.run([train_step, loss], feed_dict=feed_dict)
                  loss_function = np.append(loss_function, loss_value);

            train_weight = weight(n_inp, h, n_out)
            train_weight.W = sess.run(W);
            train_weight.b = sess.run(b);
            train_weight.W2 = sess.run(W2);
            train_weight.b2 = sess.run(b2);
            
            y = sess.run(y, feed_dict={x: z1_test})
        
        return train_weight, loss_function, val_error_function, y;   
    
def mc_nn_classification(z1, z2, z1_test, init_weight, tf_inner_iteration, opt_params, fixed_params):
    
        n_inp = z1.shape[1]
        n_out = z2.shape[1]
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
            loss = tf.reduce_mean(loss + opt_params.reg * regularizer)
            
            cross_entropy = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels = y_, logits = y))
            
            #train_prediction = tf.round(tf.sigmoid(y))
            train_prediction = tf.sigmoid(y)
            
            train_step = tf.train.GradientDescentOptimizer(fixed_params.step_size).minimize(cross_entropy)

        with tf.Session(graph = graph, config=tf.ConfigProto(log_device_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(tf_inner_iteration):
                
                  feed_dict={x: z1, y_: z2}
                  _, loss_value = sess.run([train_step, cross_entropy], feed_dict=feed_dict)
                  loss_function = np.append(loss_function, loss_value);

            train_weight = weight(n_inp, h, n_out)
            train_weight.W = sess.run(W);
            train_weight.b = sess.run(b);
            train_weight.W2 = sess.run(W2);
            train_weight.b2 = sess.run(b2);
            
            predictions = sess.run(train_prediction, feed_dict={x: z1_test})
        
        return train_weight, loss_function, val_error_function, predictions; 