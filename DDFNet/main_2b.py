
#%%
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score,ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score
from sklearn.manifold import TSNE
from tensorflow.keras.layers import Concatenate

import models
from preprocess_2b import (get_data, data_get)


#%%
def draw_learning_curves(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.show()
    plt.close()

# def draw_confusion_matrix(cf_matrix, sub, results_path):
#     # Generate confusion matrix plot
#     display_labels = ['Left hand', 'Right hand']
#     disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix,
#                                   display_labels=display_labels)
#     disp.plot()
#     disp.ax_.set_xticklabels(display_labels, rotation=12)
#     plt.title('Confusion Matrix of Subject: ' + sub )
#     plt.savefig(results_path + '/subject_' + sub + '.png')
#     plt.show()

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def draw_confusion_matrix(cf_matrix, sub, results_path, classes_labels):
    # Set global font to Times New Roman
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    plt.rcParams['font.family'] = 'Times New Roman'

    # Initialize confusion matrix display object
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix,
                                  display_labels=classes_labels)
    
    # Create figure and axis; set figure size
    fig, ax = plt.subplots(figsize=(10, 8))
    # Plot with yellow colormap
    disp.plot(ax=ax, cmap='YlOrBr')
    
    # Rotate and right-align x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Adjust bottom margin to give x-axis labels space
    plt.subplots_adjust(bottom=0.25)
    
    # Set uniform font size
    font_size = 18
    # Set tick label font size
    ax.tick_params(axis='both', labelsize=font_size)
    # Set axis title font size
    ax.set_xlabel('Predicted label', fontsize=font_size)
    ax.set_ylabel('True label', fontsize=font_size)
    
    # Adjust cell text font size
    for text in ax.texts:
        text.set_fontsize(font_size)
    
    # Set title font size
    plt.title(f'Confusion Matrix of Subject: {sub}', fontsize=font_size + 2)
    
    # Adjust colorbar tick label size
    # Get colorbar object
    cbar = disp.im_.colorbar
    # Set colorbar tick font size
    cbar.ax.tick_params(labelsize=font_size)
    
    # Save image; ensure labels fully visible
    plt.savefig(f'{results_path}/subject_{sub}.png', bbox_inches='tight')
    # Display image
    plt.show()


def draw_performance_barChart(num_sub, metric, label):
    fig, ax = plt.subplots()
    x = list(range(1, num_sub+1))
    ax.bar(x, metric, 0.5, label=label)
    ax.set_ylabel(label)
    ax.set_xlabel("Subject")
    ax.set_xticks(x)
    ax.set_title('Model '+ label + ' per subject')
    ax.set_ylim([0,1])
    
    
#%% Training 
def train(dataset_conf, train_conf, results_path):
    # Get the current 'IN' time to calculate the overall training time
    in_exp = time.time()
    # Create a file to store the path of the best model among several runs
    best_models = open(results_path + "/best models.txt", "w")
    # Create a file to store performance during training
    log_write = open(results_path + "/log.txt", "w")
    # Create a .npz file (zipped archive) to store the accuracy and kappa metrics 
    # for all runs (to calculate average accuracy/kappa over all runs)
    perf_allRuns = open(results_path + "/perf_allRuns.npz", 'wb')
    
    # Get data paramters
    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    isStandard = dataset_conf.get('isStandard')
    LOSO = dataset_conf.get('LOSO')
    # Get training hyperparamters
    batch_size = train_conf.get('batch_size')
    epochs = train_conf.get('epochs')
    patience = train_conf.get('patience')
    lr = train_conf.get('lr')
    LearnCurves = train_conf.get('LearnCurves') # Plot Learning Curves?
    n_train = train_conf.get('n_train')
    model_name = train_conf.get('model')

    # Initialize variables
    acc = np.zeros((n_sub, n_train))
    kappa = np.zeros((n_sub, n_train))

    # Iteration over subjects 
    for sub in range(n_sub): # (num_sub): for all subjects, (i-1,i): for the ith subject.
        # Get the current 'IN' time to calculate the subject training time
        in_sub = time.time()
        print('\nTraining on subject ', sub+1)
        log_write.write( '\nTraining on subject '+ str(sub+1) +'\n')
        # Initiating variables to save the best subject accuracy among multiple runs.
        BestSubjAcc = 0 
        bestTrainingHistory = [] 
        # Get training and test data
        X_train, _, y_train_onehot, X_test, _, y_test_onehot = get_data(
            data_path, sub, LOSO, isStandard)
        # X_train1=np.load("./dataset/cwt_e"+str(sub+1)+".npz")
        # X_test1 = np.load("./dataset/cwt_t" + str(sub + 1)+".npz")
        # X_train, _, y_train_onehot, X_test, _, y_test_onehot = data_get(
        #      sub)
        
        # Iteration over multiple runs 
        for train in range(n_train): # How many repetitions of training for subject i.
            # Get the current 'IN' time to calculate the 'run' training time
            in_run = time.time()
            # Create folders and files to save trained models for all runs
            filepath = results_path + '/saved models/run-{}'.format(train+1)
            if not os.path.exists(filepath):
                os.makedirs(filepath)        
            filepath = filepath + '/subject-{}.h5'.format(sub+1)
            
            # Create the model
            model = getModel(model_name)
            # Compile and train the model
            in_train = time.time()
            model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
            callbacks = [
                ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, 
                                save_best_only=True, save_weights_only=True, mode='max'),
                EarlyStopping(monitor='val_accuracy', verbose=1, mode='max', patience=patience)
            ]
            history = model.fit(X_train,y_train_onehot, validation_data=(X_test, y_test_onehot),
                                epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)
            out_train = time.time()
            # Evaluate the performance of the trained model. 
            # Here we load the Trained weights from the file saved in the hard 
            # disk, which should be the same as the weights of the current model.
            model.load_weights(filepath)
            in_test = time.time()
            y_pred = model.predict(X_test).argmax(axis=-1)
            out_test = time.time()
            labels = y_test_onehot.argmax(axis=-1)
            acc[sub, train]  = accuracy_score(labels, y_pred)
            kappa[sub, train] = cohen_kappa_score(labels, y_pred)
              
            # Get the current 'OUT' time to calculate the 'run' training time
            out_run = time.time()
            # Print & write performance measures for each run
            info = 'Subject: {}   Train no. {}   Time: {:.1f} m Train_time: {:.4f}'.format(sub + 1, train + 1,
                                                                                           ((out_run - in_run) / 60),
                                                                                           out_train - in_train)
            info = info + 'Test_acc: {:.4f}   Test_kappa: {:.4f} Test_time: {:.4f}'.format(acc[sub, train],
                                                                                           kappa[sub, train],
                                                                                           out_test - in_test)
            print(info)
            log_write.write(info +'\n')
            # If current training run is better than previous runs, save the history.
            if(BestSubjAcc < acc[sub, train]):
                 BestSubjAcc = acc[sub, train]
                 bestTrainingHistory = history
        
        # Store the path of the best model among several runs
        best_run = np.argmax(acc[sub,:])
        filepath = '/saved models/run-{}/subject-{}.h5'.format(best_run+1, sub+1)+'\n'
        best_models.write(filepath)
        # Get the current 'OUT' time to calculate the subject training time
        out_sub = time.time()
        # Print & write the best subject performance among multiple runs
        info = '----------\n'
        info = info + 'Subject: {}   best_run: {}   Time: {:.1f} m   '.format(sub+1, best_run+1, ((out_sub-in_sub)/60))
        info = info + 'acc: {:.4f}   avg_acc: {:.4f} +- {:.4f}   '.format(acc[sub, best_run], np.average(acc[sub, :]), acc[sub,:].std() )
        info = info + 'kappa: {:.4f}   avg_kappa: {:.4f} +- {:.4f}'.format(kappa[sub, best_run], np.average(kappa[sub, :]), kappa[sub,:].std())
        info = info + '\n----------'
        print(info)
        log_write.write(info+'\n')
        # Plot Learning curves 
        if (LearnCurves == True):
            print('Plot Learning Curves ....... ')
            draw_learning_curves(bestTrainingHistory)
          
    # Get the current 'OUT' time to calculate the overall training time
    out_exp = time.time()
    info = '\nTime: {:.1f} h   '.format( (out_exp-in_exp)/(60*60) )
    print(info)
    log_write.write(info+'\n')
    
    # Store the accuracy and kappa metrics as arrays for all runs into a .npz 
    # file format, which is an uncompressed zipped archive, to calculate average
    # accuracy/kappa over all runs.
    np.savez(perf_allRuns, acc = acc, kappa = kappa)
    
    # Close open files 
    best_models.close()   
    log_write.close() 
    perf_allRuns.close() 


#%% Evaluation 
def test(model, dataset_conf, results_path, allRuns = True):
    # Open the  "Log" file to write the evaluation results 
    log_write = open(results_path + "/log.txt", "a")
    # Open the file that stores the path of the best models among several random runs.
    best_models = open(results_path + "/best models.txt", "r")   
    
    # Get data paramters
    n_classes = dataset_conf.get('n_classes')
    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    isStandard = dataset_conf.get('isStandard')
    LOSO = dataset_conf.get('LOSO')
    
    # Initialize variables
    acc_bestRun = np.zeros(n_sub)
    kappa_bestRun = np.zeros(n_sub)  
    cf_matrix = np.zeros([n_sub, n_classes, n_classes])
    classes_labels = dataset_conf.get('cl_labels')
    # Calculate the average performance (average accuracy and K-score) for 
    # all runs (experiments) for each subject.
    if(allRuns): 
        # Load the test accuracy and kappa metrics as arrays for all runs from a .npz 
        # file format, which is an uncompressed zipped archive, to calculate average
        # accuracy/kappa over all runs.
        perf_allRuns = open(results_path + "/perf_allRuns.npz", 'rb')
        perf_arrays = np.load(perf_allRuns)
        acc_allRuns = perf_arrays['acc']
        kappa_allRuns = perf_arrays['kappa']
    y_all = []
    labels_all = []
    # Iteration over subjects 
    for sub in range(n_sub): # (num_sub): for all subjects, (i-1,i): for the ith subject.
        # Load data
        _, _, _, X_test, _, y_test_onehot = get_data(data_path, sub, LOSO, isStandard)
        # Load the best model out of multiple random runs (experiments).
        filepath = best_models.readline()
        model.load_weights(results_path + filepath[:-1])
        # Predict MI task
        y_p = model.predict(X_test)
        y_pred = y_p.argmax(axis=-1)
        # Calculate accuracy and K-score
        labels = y_test_onehot.argmax(axis=-1)
        if sub==0:
            y_all=y_p
            labels_all=labels
        else:
            y_all=Concatenate(axis=0)([y_all, y_p])
            labels_all=Concatenate()([labels_all, labels])



        # tsne_2D = TSNE(random_state=0).fit_transform(y_p)
        # color_map = ['r', 'y', 'k', 'g', 'b', 'm', 'c']
        # x_min, x_max = np.min(tsne_2D, 0), np.max(tsne_2D, 0)
        # tsne_2D = (tsne_2D - x_min) / (x_max - x_min)
        # fig = plt.figure()
        # for i in range(tsne_2D.shape[0]):
        #     # plt.plot(y_all[i, 0], y_all[i, 1], marker='o', markersize=1, color=color_map[labels_all[i]])
        #     plt.plot(tsne_2D[i, 0], tsne_2D[i, 1], marker='o', markersize=1, color=color_map[labels_all[i]])


        plt.xticks([])
        plt.yticks([])


        # plt.title('t-sne')
        # fig.show()


        acc_bestRun[sub] = accuracy_score(labels, y_pred)
        kappa_bestRun[sub] = cohen_kappa_score(labels, y_pred)
        # Calculate and draw confusion matrix
        cf_matrix[sub, :, :] = confusion_matrix(labels, y_pred, normalize='pred')
        # draw_confusion_matrix(cf_matrix[sub, :, :], str(sub+1), results_path)
        
        # Print & write performance measures for each subject
        info = 'Subject: {}   best_run: {:2}  '.format(sub+1, (filepath[filepath.find('run-')+4:filepath.find('/sub')]) )
        info = info + 'acc: {:.4f}   kappa: {:.4f}   '.format(acc_bestRun[sub], kappa_bestRun[sub] )
        if(allRuns): 
            info = info + 'avg_acc: {:.4f} +- {:.4f}   avg_kappa: {:.4f} +- {:.4f}'.format(
                np.average(acc_allRuns[sub, :]), acc_allRuns[sub,:].std(),
                np.average(kappa_allRuns[sub, :]), kappa_allRuns[sub,:].std() )
        print(info)
        log_write.write('\n'+info)
      
    # Print & write the average performance measures for all subjects     
    info = '\nAverage of {} subjects - best runs:\nAccuracy = {:.4f}   Kappa = {:.4f}\n'.format(
        n_sub, np.average(acc_bestRun), np.average(kappa_bestRun)) 
    # if(allRuns):
    #     info = info + '\nAverage of {} subjects x {} runs (average of {} experiments):\nAccuracy = {:.4f}   Kappa = {:.4f}'.format(
    #         n_sub, acc_allRuns.shape[1], (n_sub * acc_allRuns.shape[1]),
    #         np.average(acc_allRuns), np.average(kappa_allRuns))
    print(info)
    tsne_2D = TSNE(random_state=0).fit_transform(y_all)
    color_map = ['r', 'y', 'k', 'g', 'b', 'm', 'c']
    x_min, x_max = np.min(tsne_2D, 0), np.max(tsne_2D, 0)
    tsne_2D = (tsne_2D - x_min) / (x_max - x_min)
    fig = plt.figure()


    # for i in range(tsne_2D.shape[0]):
    #     # plt.plot(y_all[i, 0], y_all[i, 1], marker='o', markersize=1, color=color_map[labels_all[i]])
    #     plt.plot(tsne_2D[i, 0], tsne_2D[i, 1], marker='o', markersize=1, color=color_map[labels_all[i]])




    plt.xticks([])
    plt.yticks([])


    # plt.title('t-sne')


    fig.show()
    log_write.write(info)
    draw_confusion_matrix(cf_matrix.mean(0), 'All', results_path, classes_labels)
    # # Draw a performance bar chart for all subjects
    # draw_performance_barChart(n_sub, acc_bestRun, 'Accuracy')
    # draw_performance_barChart(n_sub, kappa_bestRun, 'K-score')
    # # Draw confusion matrix for all subjects (average)
    # draw_confusion_matrix(cf_matrix.mean(0), 'All', results_path)
    # # Close open files
    # log_write.close()
    
    
#%%
def getModel(model_name):
    # Select the model
    if(model_name == 'DDFFNet'):
        # Train using the proposed model (ATCNet): https://doi.org/10.1109/TII.2022.3197419
        model = models.DDFNet_(
            # Dataset parameters
            n_classes = 2,
            in_chans = 3,
            in_samples = 1125,
            # Sliding window (SW) parameter
            n_windows = 5, 
            # Attention (AT) block parameters
            attention1='mha',  # Options: None, 'mha','mhla', 'cbam', 'se'
            attention2='custom',  # Options: None, 'mha','mhla', 'cbam', 'se'
            # Convolutional (CV) block parameters
            eegn_F1 = 16,
            eegn_D = 2, 
            eegn_kernelSize = 64,
            eegn_poolSize = 7,
            eegn_dropout = 0.3,
            # Temporal convolutional (TC) block parameters
            tcn_depth = 2, 
            tcn_kernelSize = 4,
            tcn_filters = 32,
            tcn_dropout = 0.3, 
            tcn_activation='elu'
            )     
    elif(model_name == 'TCNet_Fusion'):
        # Train using TCNet_Fusion: https://doi.org/10.1016/j.bspc.2021.102826
        model = models.TCNet_Fusion(n_classes = 2)
    elif(model_name == 'EEGTCNet'):
        # Train using EEGTCNet: https://arxiv.org/abs/2006.00622
        model = models.EEGTCNet(n_classes = 2)
    elif(model_name == 'EEGNet'):
        # Train using EEGNet: https://arxiv.org/abs/1611.08024
        model = models.EEGNet_classifier(n_classes = 2)
    elif(model_name == 'EEGNeX'):
        # Train using EEGNeX: https://arxiv.org/abs/2207.12369
        model = models.EEGNeX_8_32(n_timesteps = 1125 , n_features = 3, n_outputs = 2)
    elif(model_name == 'DeepConvNet'):
        # Train using DeepConvNet: https://doi.org/10.1002/hbm.23730
        model = models.DeepConvNet(nb_classes = 2 , Chans = 3, Samples = 1125)
    elif(model_name == 'ShallowConvNet'):
        # Train using ShallowConvNet: https://doi.org/10.1002/hbm.23730
        model = models.ShallowConvNet(nb_classes = 2 , Chans = 3, Samples = 1125)
    elif(model_name == 'MYNET'):
        model= models.MYNET(n_classes = 2)
    elif (model_name == 'ATCNetNS'):
        # Train using the proposed model (ATCNet): https://doi.org/10.1109/TII.2022.3197419
        model = models.ATCNetNS(
            # Dataset parameters
            n_classes=2,
            in_chans=3,
            in_samples=1125,
            # Sliding window (SW) parameter
            n_windows=5,
            # Attention (AT) block parameter
            attention='mhla',  # Options: None, 'mha','mhla', 'cbam', 'se'
            # Convolutional (CV) block parameters
            eegn_F1=16,
            eegn_D=2,
            eegn_kernelSize=64,
            eegn_poolSize=7,
            eegn_dropout=0.3,
            # Temporal convolutional (TC) block parameters
            tcn_depth=2,
            tcn_kernelSize=4,
            tcn_filters=32,
            tcn_dropout=0.3,
            tcn_activation='elu'
        )
    else:
        raise Exception("'{}' model is not supported yet!".format(model_name))

    return model
    
    
#%%
def run():
    # Get data path
    data_path = './BCI-2b'
    
    # Create a folder to store the results of the experiment
    results_path = os.getcwd() + "/2b results"
    if not  os.path.exists(results_path):
      os.makedirs(results_path)   # Create a new directory if it does not exist 
      
    # Set data paramters
    dataset_conf = { 'n_classes': 2, 'n_sub': 9, 'n_channels': 3, 'data_path': data_path,
                'isStandard': True, 'LOSO': False}
    # Set training hyperparamters
    train_conf = { 'batch_size': 64, 'epochs': 1000, 'patience': 300, 'lr': 0.0009,
                  'LearnCurves': False, 'n_train': 1, 'model':'DDFNet'}
           
    # Train the model
    train(dataset_conf, train_conf, results_path)

    # Evaluate the model based on the weights saved in the '/results' folder
    model = getModel(train_conf.get('model'))
    test(model, dataset_conf, results_path)    
    
#%%
if __name__ == "__main__":
    run()
    