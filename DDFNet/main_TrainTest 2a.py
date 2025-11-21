import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.manifold import TSNE
import seaborn as sns
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score
import models
from preprocess import get_data
from tensorflow.keras import backend as K

#%%
def draw_learning_curves(history):
    # Plot learning curves
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy') # Model accuracy
    plt.ylabel('Accuracy') # Accuracy
    plt.xlabel('Epoch') # Epoch
    plt.legend(['Train', 'val'], loc='upper left') # Train and validation
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss') # Model loss
    plt.ylabel('Loss') # Loss
    plt.xlabel('Epoch') # Epoch
    plt.legend(['Train', 'val'], loc='upper left') # Train and validation
    plt.show()
    plt.close()


from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def draw_confusion_matrix(cf_matrix, sub, results_path, classes_labels):
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
    # Plot performance bar chart
    fig, ax = plt.subplots()
    x = list(range(1, num_sub+1))
    ax.bar(x, metric, 0.5, label=label)
    ax.set_ylabel(label)
    ax.set_xlabel("Subject") # Subject
    ax.set_xticks(x)
    ax.set_title('Model '+ label + ' per subject') # Model metric per subject
    ax.set_ylim([0,1])



#%% Training
def train(dataset_conf, train_conf, results_path):
    # Get the current "IN" time to compute overall training time
    in_exp = time.time()
    # Create a file to store paths of best models across runs
    best_models = open(results_path + "/best models.txt", "w")
    # Create a file to store performance during training
    log_write = open(results_path + "/log.txt", "w")
    # Create a .npz archive to store accuracy and kappa for all runs
    # (to compute average accuracy/kappa across runs)
    perf_allRuns = open(results_path + "/perf_allRuns.npz", 'wb')

    # Get dataset parameters
    dataset = dataset_conf.get('name')
    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    isStandard = dataset_conf.get('isStandard')
    LOSO = dataset_conf.get('LOSO')
    # Get training hyperparameters
    batch_size = train_conf.get('batch_size')
    epochs = train_conf.get('epochs')
    patience = train_conf.get('patience')
    lr = train_conf.get('lr')
    LearnCurves = train_conf.get('LearnCurves') # Plot learning curves?
    n_train = train_conf.get('n_train')
    model_name = train_conf.get('model')

    # Initialize variables
    acc = np.zeros((n_sub, n_train))
    kappa = np.zeros((n_sub, n_train))

    pre_trained_weights = {
        0: 'D:/code/MI/5/EEG-ATCNet-main/results diff/saved models/run-1/subject-1.h5',
        1: 'D:/code/MI/5/EEG-ATCNet-main/results diff/saved models/run-10/subject-2.h5',
        2: 'D:/code/MI/5/EEG-ATCNet-main/results diff/saved models/run-5/subject-3.h5',
        3: 'D:/code/MI/5/EEG-ATCNet-main/results diff/saved models/run-5/subject-4.h5',
        4: 'D:/code/MI/5/EEG-ATCNet-main/results diff/saved models/run-7/subject-5.h5',
        5: 'D:/code/MI/5/EEG-ATCNet-main/results diff/saved models/run-4/subject-6.h5',
        6: 'D:/code/MI/5/EEG-ATCNet-main/results diff/saved models/rn-3/subject-7.h5',
        7: 'D:/code/MI/5/EEG-ATCNet-main/results diff/saved models/run-10/subject-8.h5',
        8: 'D:/code/MI/5/EEG-ATCNet-main/results diff/saved models/run-8/subject-9.h5'
    }
    # Iterate over subjects
    for sub in range(n_sub): # Loop over all subjects
        # Get the current "IN" time to compute subject training time
        in_sub = time.time()
        print('\nTraining on subject ', sub+1) # Training subject
        log_write.write( '\nTraining on subject '+ str(sub+1) +'\n')
        # Initialize trackers to save best subject accuracy across runs
        BestSubjAcc = 0
        bestTrainingHistory = []
        # Get training and testing data
        X_train, _, y_train_onehot, X_test, _, y_test_onehot = get_data(
            data_path, sub, dataset, LOSO = LOSO, isStandard = isStandard)

        # Iterations across multiple runs
        for train in range(n_train): # Number of training repetitions per subject
            # Get the current "IN" time to compute the run training time
            tf.random.set_seed(train+1)
            np.random.seed(train+1)

            in_run = time.time()
            # Create folders/files to save trained models for all runs
            filepath = results_path + '/saved models/run-{}'.format(train+1)
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            filepath = filepath + '/subject-{}.h5'.format(sub+1)

            # Create model
            model = getModel(model_name, dataset_conf)

            # 预训练权重加载
            # if sub in pre_trained_weights and os.path.exists(pre_trained_weights[sub]):
            #    pre_trained_weights_path = pre_trained_weights[sub]
            #    model.load_weights(pre_trained_weights_path)
            #    print(f"Pre-trained weights for subject {sub + 1} loaded successfully from {pre_trained_weights_path}")
            # else:
            #    print(f"No pre-trained weights for subject {sub + 1}, training from scratch.")



            # Compile and train model
            model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
            # model.summary()
            # plot_model(model, to_file='plot_model.png', show_shapes=True, show_layer_names=True)

            callbacks = [
                ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0,
                                save_best_only=True, save_weights_only=True, mode='max'),

                ReduceLROnPlateau(monitor="val_loss", factor=0.90, patience=20, verbose=1, min_lr=0.0001),

                EarlyStopping(monitor='val_accuracy', verbose=1, mode='max', patience=patience)
            ]
            history = model.fit(X_train, y_train_onehot, validation_data=(X_test, y_test_onehot),
                                epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)

            # Evaluate trained model performance
            # Load trained weights from disk; should match current model weights
            model.load_weights(filepath)
            y_pred = model.predict(X_test).argmax(axis=-1)
            labels = y_test_onehot.argmax(axis=-1)
            acc[sub, train]  = accuracy_score(labels, y_pred)
            kappa[sub, train] = cohen_kappa_score(labels, y_pred)

            # Get the current "OUT" time to compute run training time
            out_run = time.time()
            # Print & write performance metrics for each run
            info = 'Subject: {}   Train no. {}   Time: {:.1f} m   '.format(sub+1, train+1, ((out_run-in_run)/60)) # Subject, run number, time
            info = info + 'Test_acc: {:.4f}   Test_kappa: {:.4f}'.format(acc[sub, train], kappa[sub, train]) # Test accuracy, test kappa
            print(info)
            log_write.write(info +'\n')
            # If current run outperforms previous runs, save history
            if(BestSubjAcc < acc[sub, train]):
                 BestSubjAcc = acc[sub, train]
                 bestTrainingHistory = history

        # Store path of the best model across runs
        best_run = np.argmax(acc[sub,:])
        filepath = '/saved models/run-{}/subject-{}.h5'.format(best_run+1, sub+1)+'\n'
        best_models.write(filepath)
        # Get the current "OUT" time to compute subject training time
        out_sub = time.time()
        # Print & write best subject performance among multiple runs
        info = '----------\n'
        info = info + 'Subject: {}   best_run: {}   Time: {:.1f} m   '.format(sub+1, best_run+1, ((out_sub-in_sub)/60)) # Subject, best run, time
        info = info + 'acc: {:.4f}   avg_acc: {:.4f} +- {:.4f}   '.format(acc[sub, best_run], np.average(acc[sub, :]), acc[sub,:].std() ) # Accuracy, average accuracy
        info = info + 'kappa: {:.4f}   avg_kappa: {:.4f} +- {:.4f}'.format(kappa[sub, best_run], np.average(kappa[sub, :]), kappa[sub,:].std()) # Kappa, average kappa
        info = info + '\n----------'
        print(info)
        log_write.write(info+'\n')
        # Plot learning curves
        if (LearnCurves == True):
            print('Plot Learning Curves ....... ') # Plot learning curves
            draw_learning_curves(bestTrainingHistory)

    # Get the current "OUT" time to compute overall training time
    out_exp = time.time()
    info = '\nTime: {:.1f} h   '.format( (out_exp-in_exp)/(60*60) ) # Time
    print(info)
    log_write.write(info+'\n')
    
    # Store accuracy and kappa arrays for all runs into .npz (uncompressed archive)
    # Used to compute average accuracy/kappa across runs
    np.savez(perf_allRuns, acc = acc, kappa = kappa)

    # Close open files
    best_models.close()
    log_write.close()
    perf_allRuns.close()


#%% Evaluation
def test(model, dataset_conf, results_path, allRuns = True):
    # Open log file to write evaluation results
    log_write = open(results_path + "/log.txt", "a")
    # Open file storing paths of best models across random runs
    best_models = open(results_path + "/best models.txt", "r")

    # Get dataset parameters
    dataset = dataset_conf.get('name')
    n_classes = dataset_conf.get('n_classes')
    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    isStandard = dataset_conf.get('isStandard')
    LOSO = dataset_conf.get('LOSO')
    classes_labels = dataset_conf.get('cl_labels')

    # Initialize variables
    acc_bestRun = np.zeros(n_sub)
    kappa_bestRun = np.zeros(n_sub)
    cf_matrix = np.zeros([n_sub, n_classes, n_classes])

    # Compute average performance across runs (accuracy and kappa)
    if(allRuns):
        # Load test accuracy and kappa arrays for all runs from .npz (uncompressed archive)
        # Used to compute average accuracy/kappa across runs
        perf_allRuns = open(results_path + "/perf_allRuns.npz", 'rb')
        perf_arrays = np.load(perf_allRuns)
        acc_allRuns = perf_arrays['acc']
        kappa_allRuns = perf_arrays['kappa']

    # Iterate over subjects
    for sub in range(n_sub): # Loop over all subjects
        # Load data
        _, _, _, X_test, _, y_test_onehot = get_data(
            data_path, sub, dataset, LOSO = LOSO, isStandard = isStandard)
        # Load weights of the best run among multiple random runs
        filepath = best_models.readline()
        model.load_weights(results_path + filepath[:-1])



        # Final outputs
        features = get_layer_output(model, X_test, index=371)
        # features = get_layer_output(model, X_test, index=365)
        # features = get_layer_output(model, X_test, index=30)
        # features = get_layer_output(model, X_test, index=100)
        # features = get_layer_output(model, X_test, index=180)
        # features = get_layer_output(model, X_test, index=0)


        features = features.reshape(features.shape[0], -1)  # Flatten features


        # Prediction
        y_pred = model.predict(X_test).argmax(axis=-1)
        # Compute accuracy and kappa
        labels = y_test_onehot.argmax(axis=-1)

        # plot_tsne_single(features, labels,
        #                  subject_id=sub + 1,
        #                  save_path=results_path,
        #                  class_labels=classes_labels)

        acc_bestRun[sub] = accuracy_score(labels, y_pred)
        kappa_bestRun[sub] = cohen_kappa_score(labels, y_pred)





        # Compute and plot confusion matrix
        cf_matrix[sub, :, :] = confusion_matrix(labels, y_pred, normalize='true')
        draw_confusion_matrix(cf_matrix[sub, :, :], str(sub+1), results_path, classes_labels)
        # Print & write performance metrics for each subject
        info = 'Subject: {}   best_run: {:2}  '.format(sub+1, (filepath[filepath.find('run-')+4:filepath.find('/sub')]) ) # Subject, best run
        info = info + 'acc: {:.4f}   kappa: {:.4f}   '.format(acc_bestRun[sub], kappa_bestRun[sub] ) # Accuracy, kappa
        if(allRuns):
            info = info + 'avg_acc: {:.4f} +- {:.4f}   avg_kappa: {:.4f} +- {:.4f}'.format(
                np.average(acc_allRuns[sub, :]), acc_allRuns[sub,:].std(),
                np.average(kappa_allRuns[sub, :]), kappa_allRuns[sub,:].std() ) # Average accuracy, average kappa
        print(info)
        log_write.write('\n'+info)

    # Print & write average performance metrics across all subjects
    info = '\nAverage of {} subjects - best runs:\nAccuracy = {:.4f}   Kappa = {:.4f}\n'.format(
        n_sub, np.average(acc_bestRun), np.average(kappa_bestRun)) # Average across subjects, best runs, accuracy, kappa
    if(allRuns):
        info = info + '\nAverage of {} subjects x {} runs (average of {} experiments):\nAccuracy = {:.4f}   Kappa = {:.4f}'.format(
            n_sub, acc_allRuns.shape[1], (n_sub * acc_allRuns.shape[1]),
            np.average(acc_allRuns), np.average(kappa_allRuns)) # Average over subjects x runs; average accuracy, average kappa
    print(info)
    log_write.write(info)

    # Plot performance bar charts for all subjects
    draw_performance_barChart(n_sub, acc_bestRun, 'Accuracy') # Accuracy
    draw_performance_barChart(n_sub, kappa_bestRun, 'K-score') # Kappa score
    # Draw confusion matrix averaged across all subjects
    draw_confusion_matrix(cf_matrix.mean(0), 'All', results_path, classes_labels) # All subjects confusion matrix
    # Close open files
    log_write.close()

# t-SNE plot without axis labels

def plot_tsne_single(features, labels, subject_id, save_path, class_labels):
    """
    :param features: feature matrix (n_samples, n_features)
    :param labels: ground-truth labels (n_samples,)
    :param subject_id: subject ID
    :param save_path: output directory
    :param class_labels: class label list
    """
    plt.rcParams['font.family'] = 'Times New Roman'
    tsne = TSNE(n_components=2, random_state=0)
    tsne_2d = tsne.fit_transform(features)

    x_min, x_max = np.min(tsne_2d, 0), np.max(tsne_2d, 0)
    tsne_2d = (tsne_2d - x_min) / (x_max - x_min)

    # High-contrast palette (red, green, yellow, black)
    colors = ['#e6194b', '#3cb44b', '#ffe119', '#000000']

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.tick_params(labelsize=20)
    for i, label in enumerate(class_labels):
        mask = labels == i
        plt.scatter(tsne_2d[mask, 0], tsne_2d[mask, 1],
                    color=colors[i],
                    edgecolor='k',
                    label=label,
                    alpha=0.5,
                    s=100,
                    linewidth=0.6)

    # Remove axis labels
    plt.xlabel('')
    plt.ylabel('')

    # Remove top and right spines (matplotlib defaults to four)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Emphasize bottom and left axis lines
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)

    # Hide axis ticks
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])

    plt.title(f'Subject {subject_id}', fontsize=12)

    # Legend handling
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          label=class_labels[i],
                          markerfacecolor=colors[i],
                          markeredgecolor='k',
                          markersize=10) for i in range(len(class_labels))]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=18)

    os.makedirs(save_path, exist_ok=True)
    save_file = f"{save_path}/tsne_subject_{subject_id}.png"
    plt.savefig(save_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved t-SNE plot to {save_file}")


def get_layer_output(model, x, index=-1):
    """
    get the computing result output of any layer you want, default the last layer.
    :param model: primary model
    :param x: input of primary model( x of model.predict([x])[0])
    :param index: index of target layer, i.e., layer[23]
    :return: result
    """
    layer = K.function([model.input], [model.layers[index].output])
    return layer([x])[0]
#%%
def getModel(model_name, dataset_conf):
    n_classes = dataset_conf.get('n_classes')
    n_channels = dataset_conf.get('n_channels')
    in_samples = dataset_conf.get('in_samples')

    # Select model
    if (model_name == 'DDFNet'):
        model = models.DDFNet_(
            # Dataset parameters
            n_classes=n_classes,
            in_chans=n_channels,
            in_samples=in_samples,
            # Sliding window (SW) parameters
            n_windows=5,
            # Attention (AT) block parameters
            attention1='mha',  # Options: None, 'mha','mhla', 'cbam', 'se'
            attention2='custom',  # Options: None, 'mha','mhla', 'cbam', 'se'
            # Convolution (CV) block parameters
            eegn_F1=16,
            eegn_D=2,
            eegn_kernelSize=64,
            eegn_poolSize=7,
            eegn_dropout=0.3,
            # Temporal convolution (TC) block parameters
            tcn_depth=2,
            tcn_kernelSize=4,
            tcn_filters=32,
            tcn_dropout=0.3,
            tcn_activation='elu'
        )
    elif (model_name == 'TCNet_Fusion'):
        # Train using TCNet_Fusion: https://doi.org/10.1016/j.bspc.2021.102826
        model = models.TCNet_Fusion(n_classes=n_classes, Chans=n_channels, Samples=in_samples)
    else:
        raise Exception("'{}' model is not supported yet!".format(model_name))

    return model

#%%
def run():
    # Define dataset parameters
    dataset = 'BCI2a'  # Options: 'BCI2a','HGD', 'CS2R'

    if dataset == 'BCI2a':
        in_samples = 1125
        n_channels = 22
        n_sub = 9
        n_classes = 4
        classes_labels = ['Left hand', 'Right hand', 'Foot', 'Tongue'] # Left hand, Right hand, Foot, Tongue
        data_path = './BCI-2a'

    elif dataset == 'HGD':
        in_samples = 1125
        n_channels = 44
        n_sub = 14
        n_classes = 4
        classes_labels = ['Right Hand', 'Left Hand', 'Rest', 'Feet'] # Right hand, Left hand, Rest, Feet
        data_path = os.path.expanduser(
            '~') + '/mne_data/MNE-schirrmeister2017-data/robintibor/high-gamma-dataset/raw/master/data/'
    elif dataset == 'CS2R':
        in_samples = 1125
        n_channels = 32
        n_sub = 18
        n_classes = 3
        classes_labels = ['Fingers', 'Wrist', 'Elbow'] # Fingers, Wrist, Elbow
        data_path = os.path.expanduser(
            '~') + '/CS2R MI EEG dataset/all/EDF - Cleaned - phase one (remove extra runs)/two sessions/'
    else:
        raise Exception("'{}'dataset is not supported yet!".format(dataset))

    # Create folder to store experiment results
    results_path = os.getcwd() + "/results"
    if not os.path.exists(results_path):
        os.makedirs(results_path)  # Create directory if it does not exist

    # Set dataset parameters
    dataset_conf = {'name': dataset, 'n_classes': n_classes, 'cl_labels': classes_labels,
                    'n_sub': n_sub, 'n_channels': n_channels, 'in_samples': in_samples,
                    'data_path': data_path, 'isStandard': True, 'LOSO': False}
    # Set training hyperparameters
    train_conf = {'batch_size': 64, 'epochs': 1000, 'patience': 300, 'lr': 0.001,
                  'LearnCurves': True, 'n_train': 10, 'model': 'DDFNet'}

    # Train model
    train(dataset_conf, train_conf, results_path)

    # Evaluate model based on weights saved in the '/results' folder
    model = getModel(train_conf.get('model'), dataset_conf)
    test(model, dataset_conf, results_path)


#%%
if __name__ == "__main__":
    run()
