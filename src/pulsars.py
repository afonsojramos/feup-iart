from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras.callbacks import TensorBoard

from imblearn.datasets import make_imbalance
from imblearn.under_sampling import RandomUnderSampler, NeighbourhoodCleaningRule, AllKNN, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, ClusterCentroids
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

import datetime
import numpy as np
import matplotlib.pyplot as plt
import sys

# split training dataset
def split_train_dataset(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(test_ratio * len(data))
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data[train_indices,:], data[test_indices,:]

# oversampling auxiliary plot function
def plot_resampling(ax, X, y, title):
    c0 = ax.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0", alpha=0.5)
    c1 = ax.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1", alpha=0.5)
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([-6, 8])
    ax.set_ylim([-6, 6])

    return c0, c1

def sampling(algorithm, x_train, y_train):

    if (algorithm == 'standard'):

        print('\nUsing Standard Scaler.\n')

        scaler = StandardScaler().fit(x_train)
        X_resampled = scaler.transform(x_train)
        y_resampled = y_train

    elif(algorithm == 'undersampling'):
        # Instanciate a PCA object for the sake of easy visualisation
        pca = PCA(n_components=2)
        # Fit and transform x to visualise inside a 2D feature space
        X_vis = pca.fit_transform(x_train)

        print('\nUsing Random Under Sampling.\n')

        rus = RandomUnderSampler(return_indices=True)
        X_resampled, y_resampled, idx_resampled = rus.fit_sample(x_train, y_train)
        X_res_vis = pca.transform(X_resampled)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        idx_samples_removed = np.setdiff1d(np.arange(X_vis.shape[0]),
                                        idx_resampled)

        idx_class_0 = y_resampled == 0
        plt.scatter(X_res_vis[idx_class_0, 0], X_res_vis[idx_class_0, 1],
                    alpha=.8, label='Class #0')
        plt.scatter(X_res_vis[~idx_class_0, 0], X_res_vis[~idx_class_0, 1],
                    alpha=.8, label='Class #1')
        plt.scatter(X_vis[idx_samples_removed, 0], X_vis[idx_samples_removed, 1],
                    alpha=.8, label='Removed samples')
        
        # make nice plotting
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))
        ax.set_xlim([-6, 6])
        ax.set_ylim([-6, 6])

        plt.title('Under-sampling using random under-sampling')
        plt.legend()
        plt.tight_layout()
        plt.show()

    elif(algorithm == 'smote'):

        print('\nUsing SMOTE.\n')

        # Instanciate a PCA object for the sake of easy visualisation
        pca = PCA(n_components=2)
        # Fit and transform x to visualise inside a 2D feature space
        X_vis = pca.fit_transform(x_train)

        kinds = ['regular', 'borderline1', 'borderline2', 'svm']
        kind = [kinds[int(sys.argv[2] if len(sys.argv) >= 3 else 'regular')]]
        print(kind)
        sm = [SMOTE(kind=k) for k in kind]
        X_resampled = []
        y_resampled = []
        X_res_vis = []
        for method in sm:
            X_res, y_res = method.fit_sample(x_train, y_train)
            X_resampled.append(X_res)
            y_resampled.append(y_res)
            X_res_vis.append(pca.transform(X_res))

        f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
        ax2.axis('off')
        ax_res = [ax3, ax4, ax5, ax6]

        c0, c1 = plot_resampling(ax1, X_vis, y_train, 'Original set')
        for i in range(len(kind)):
            plot_resampling(ax_res[i], X_res_vis[i], y_resampled[i],
                            'SMOTE {}'.format(kind[i]))

        ax2.legend((c0, c1), ('Class #0', 'Class #1'), loc='center',
                ncol=1, labelspacing=0.)
        plt.tight_layout()
        plt.show()

    elif(algorithm=='neighbourhood'):

        print('\nUsing Neighbourhood Cleaning Rule.\n')

        pca = PCA(n_components=2)
        X_vis = pca.fit_transform(x_train)

        ncl = NeighbourhoodCleaningRule(return_indices=True)
        X_resampled, y_resampled, idx_resampled = ncl.fit_sample(x_train, y_train)
        X_res_vis = pca.transform(X_resampled)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        idx_samples_removed = np.setdiff1d(np.arange(X_vis.shape[0]),
                                        idx_resampled)

        idx_class_0 = y_resampled == 0
        plt.scatter(X_res_vis[idx_class_0, 0], X_res_vis[idx_class_0, 1],
                    alpha=.8, label='Class #0')
        plt.scatter(X_res_vis[~idx_class_0, 0], X_res_vis[~idx_class_0, 1],
                    alpha=.8, label='Class #1')
        plt.scatter(X_vis[idx_samples_removed, 0], X_vis[idx_samples_removed, 1],
                    alpha=.8, label='Removed samples')

        # make nice plotting
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))
        ax.set_xlim([-6, 6])
        ax.set_ylim([-6, 6])

        plt.title('Under-sampling using neighbourhood cleaning rule')
        plt.legend()
        plt.tight_layout()
        plt.show()

    elif(algorithm == 'ENN'):

        print('\nUsing ENN.\n')

        enn = EditedNearestNeighbours(return_indices=True)
        X_resampled, y_resampled, idx_resampled = enn.fit_sample(x_train, y_train)
        reduction_str = ('Reduced {:.2f}%'.format(100 * (1 - float(len(X_resampled)) /
                                                        len(x_train))))
        print(reduction_str)

    elif(algorithm == 'RENN'):

        print('\nUsing RENN.\n')

        renn = RepeatedEditedNearestNeighbours(return_indices=True)
        X_resampled, y_resampled, idx_resampled = renn.fit_sample(x_train, y_train)
        reduction_str = ('Reduced {:.2f}%'.format(100 * (1 - float(len(X_resampled)) /
                                                        len(x_train))))
        print(reduction_str)

    elif(algorithm == 'AllKNN'):

        print('\nUsing AllKNN.\n')

        allknn = AllKNN(return_indices=True)
        X_resampled, y_resampled, idx_resampled = allknn.fit_sample(x_train, y_train)
        reduction_str = ('Reduced {:.2f}%'.format(100 * (1 - float(len(X_resampled)) /
                                                        len(x_train))))
        print(reduction_str)

    elif(algorithm == 'centroids'):

        print('\nUsing Cluster Centroids.\n')

        # Apply Cluster Centroids
        cc = ClusterCentroids()
        X_resampled, y_resampled = cc.fit_sample(x_train, y_train)

    elif(algorithm == 'centroidshard'):

        print('\nUsing Cluster Centroids with Hard Voting.\n')

        pca = PCA(n_components=2)
        X_vis = pca.fit_transform(x_train)
        # Apply Cluster Centroids
        cc = ClusterCentroids()
        X_resampled, y_resampled = cc.fit_sample(x_train, y_train)
        X_res_vis_soft = pca.transform(X_resampled)

        # Use hard voting instead of soft voting
        cc = ClusterCentroids(voting='hard')
        X_resampled, y_resampled = cc.fit_sample(x_train, y_train)
        X_res_vis_hard = pca.transform(X_resampled)

        # Two subplots, unpack the axes array immediately
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        c0 = ax1.scatter(X_vis[y_train == 0, 0], X_vis[y_train == 0, 1], label="Class #0",
                        alpha=0.5)
        c1 = ax1.scatter(X_vis[y_train == 1, 0], X_vis[y_train == 1, 1], label="Class #1",
                        alpha=0.5)
        ax1.set_title('Original set')

        ax2.scatter(X_res_vis_soft[y_resampled == 0, 0],
                    X_res_vis_soft[y_resampled == 0, 1],
                    label="Class #0", alpha=.5)
        ax2.scatter(X_res_vis_soft[y_resampled == 1, 0],
                    X_res_vis_soft[y_resampled == 1, 1],
                    label="Class #1", alpha=.5)
        ax2.scatter(X_vis[y_train == 1, 0],
                        X_vis[y_train == 1, 1], label="Original #1",
                        alpha=0.2)
        ax2.set_title('Cluster centroids with soft voting')

        ax3.scatter(X_res_vis_hard[y_resampled == 0, 0],
                    X_res_vis_hard[y_resampled == 0, 1],
                    label="Class #0", alpha=.5)
        ax3.scatter(X_res_vis_hard[y_resampled == 1, 0],
                    X_res_vis_hard[y_resampled == 1, 1],
                    label="Class #1", alpha=.5)
        ax3.scatter(X_vis[y_train == 1, 0],
                    X_vis[y_train == 1, 1],
                    alpha=0.2)
        ax3.set_title('Cluster centroids with hard voting')

        # make nice plotting
        for ax in (ax1, ax2, ax3):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            ax.spines['left'].set_position(('outward', 10))
            ax.spines['bottom'].set_position(('outward', 10))
            ax.set_xlim([-6, 8])
            ax.set_ylim([-6, 6])

        plt.figlegend((c0, c1), ('Class #0', 'Class #1', 'Original Class #1'),
                    loc='lower center',
                    ncol=3, labelspacing=0.)
        plt.tight_layout(pad=3)
        plt.show()

    else:
        
        # Instanciate a PCA object for the sake of easy visualisation
        pca = PCA(n_components=2)
        # Fit and transform x to visualise inside a 2D feature space
        X_vis = pca.fit_transform(x_train)

        return x_train, y_train

    return X_resampled, y_resampled

# removing some randomness to get cleaner results
np.random.seed(42)

dataframe = np.loadtxt('../dataset/HTRU_2.csv', delimiter=',', dtype=np.float64)

train, test = split_train_dataset(dataframe, 0.2)

# generate training data
x_train, y_train = train[:,:8], train[:,8]

# generate test data
x_test, y_test = test[:,:8], test[:,8]

data_dim = 8
batch_size = 128
optimizer = 'rmsprop'

# model creation and configuration
model = Sequential()
 
model.add(Dense(9, input_dim=data_dim, kernel_initializer='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(32, kernel_initializer='uniform'))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(1))
model.add(Activation('sigmoid'))
 
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,  
              metrics=["accuracy"])

x_train, y_train = sampling(sys.argv[1] if len(sys.argv) >= 2 else 'base', x_train, y_train)

tensorboard = TensorBoard(log_dir='./../Graph', histogram_freq=0, write_graph=True, write_images=True)

# model fitting
model.fit(x_train, y_train,
        epochs=5,
        batch_size=batch_size,
        callbacks=[tensorboard])

# precision calculation
pred = model.predict(x_test, batch_size=batch_size)

y_pred = [i[0] for i in pred]

def precision_calculation(a):
    if a > 0.5:
        return 1.0
    return 0.0

vfunc = np.vectorize(precision_calculation)
y = vfunc(y_pred)

# reporting and printing summary, scores and some stats
report = classification_report(y_test, y)

print(report)

score = model.evaluate(x_test, y_test, batch_size=batch_size)
print(model.summary()) 
print('\nTest loss:', score[0])
print('Test accuracy:', score[1], 'with', sys.argv[1] if len(sys.argv) >= 2 else 'base', 'sampling.')

uniq_filename = '../logs/' + str("%.7f" % score[1]) + '_' + optimizer + '_' + (sys.argv[1] if len(sys.argv) >= 2 else 'base') + '_' + str(datetime.datetime.now().date()) + '.h5'
model.save(uniq_filename)
