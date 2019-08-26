import os
from glob import glob
from scipy import signal
from scipy.signal import resample
import pywt
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import pickle
import time
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
import collections
from sklearn import svm
import numpy as np
cpu_threads = 7
def display_signal(beat):
    plt.plot(beat)
    plt.ylabel('Signal')
    plt.show()

# Class for RR intervals features
class RR_intervals:
    def __init__(self):
        # Instance atributes
        self.pre_R = np.array([])
        self.post_R = np.array([])
        self.local_R = np.array([])
        self.global_R = np.array([])

def pre_pro(MLII):
    baseline = medfilt(MLII, 71)
    baseline = medfilt(baseline, 215)
    # Remove Baseline
    for i in range(0, len(MLII)):
        MLII[i] = MLII[i] - baseline[i]
    return MLII
def get_image_files(sig_dir):
  fs = glob("{}/*.dat".format(sig_dir))
  fs = [os.path.basename(filename) for filename in fs]
  return sorted(fs)
def draw_ecg(x):
    plt.plot(x)
    plt.show()
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    绘制混淆矩阵图，来源：
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'#, without normalization

    cm = confusion_matrix(y_true, y_pred)
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def print_results(y_true, y_pred, target_names,filename):
    """
    打印相关结果
    :param y_true: 期望输出，1-d array
    :param y_pred: 实际输出，1-d array
    :param target_names: 各类别名称
    :return: 打印结果
    """
    overall_accuracy = accuracy_score(y_true, y_pred)
    print('\n----- overall_accuracy: {0:f} -----'.format(overall_accuracy))
    cm = confusion_matrix(y_true, y_pred)
    f = open(filename, "w")
    f.write("Confusion Matrix:" + "\n\n")
    f.write(str(cm)+ "\n\n")
    f.write('----- overall_accuracy: {0:f} -----\n'.format(overall_accuracy))
    recall=recall(y_true, y_pred)
    precision=precision(y_true, y_pred)
    f1=f1(y_true, y_pred)
    print(' All Se = ' + str(recall))
    print(' All S+ = ' + str(precision))
    print(' All f1 = ' + str(f1))
    f.write(' All Se = ' + str(recall)+ "\n")
    f.write(' All S+ = ' + str(precision)+ "\n")
    f.write(' All f1 = ' + str(f1)+ "\n")
    for i in range(len(target_names)):
        print(target_names[i] + ':')
        Se = cm[i][i]/np.sum(cm[i])
        Pp = cm[i][i]/np.sum(cm[:, i])
        print('  Se = ' + str(Se))
        print('  P+ = ' + str(Pp))
        f.write(target_names[i] + ':'+ "\n")
        f.write('  Se = ' + str(Se) + "\n")
        f.write('  P+ = ' + str(Pp) + "\n")
    print('--------------------------------------')
    f.close()

# 对描述符数据执行过采样方法
def perform_oversampling(oversamp_method, tr_features, tr_labels,model_class):
    start = time.time()
    if True:
        print(model_class+" oversampling method:\t" + oversamp_method + " ...")
        # 1 SMOTE
        if oversamp_method == 'SMOTE':
            # kind={'borderline1', 'borderline2', 'svm'}
            svm_model = svm.SVC(C=0.001, kernel='rbf', degree=3, gamma='auto', decision_function_shape='ovo')
            oversamp = SMOTE(ratio='auto', random_state=None, k_neighbors=5, m_neighbors=10, out_step=0.5, kind='svm',
                             svm_estimator=svm_model, n_jobs=1)

            # PROBAR SMOTE CON OTRO KIND

        elif oversamp_method == 'SMOTE_regular_min':
            oversamp = SMOTE(ratio='minority', random_state=None, k_neighbors=5, m_neighbors=10, out_step=0.5,
                             kind='regular', svm_estimator=None, n_jobs=1)

        elif oversamp_method == 'SMOTE_regular':
            oversamp = SMOTE(ratio='auto', random_state=None, k_neighbors=5, m_neighbors=10, out_step=0.5,
                             kind='regular', svm_estimator=None, n_jobs=1)

        elif oversamp_method == 'SMOTE_border':
            oversamp = SMOTE(ratio='auto', random_state=None, k_neighbors=5, m_neighbors=10, out_step=0.5,
                             kind='borderline1', svm_estimator=None, n_jobs=1)

        # 2 SMOTEENN
        elif oversamp_method == 'SMOTEENN':
            oversamp = SMOTEENN()

        # 3 SMOTE TOMEK
        # NOTE: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.65.3904&rep=rep1&type=pdf
        elif oversamp_method == 'SMOTETomek':
            oversamp = SMOTETomek()

        # 4 ADASYN
        elif oversamp_method == 'ADASYN':
            oversamp = ADASYN(ratio='auto', random_state=None, k=None, n_neighbors=5, n_jobs=cpu_threads)

        tr_features_balanced, tr_labels_balanced = oversamp.fit_sample(tr_features, tr_labels)

    end = time.time()

    count = collections.Counter(tr_labels_balanced)
    print("Oversampling balance")
    print(count)
    print("Time required: " + str(format(end - start, '.2f')) + " sec")

    return tr_features_balanced, tr_labels_balanced
def ovo_class_combinations(n_classes):
    class_pos = []
    class_neg = []
    for c1 in range(n_classes-1):
        for c2 in range(c1+1,n_classes):
            class_pos.append(c1)
            class_neg.append(c2)

    return class_pos, class_neg
def ovo_voting(decision_ovo, n_classes):
    predictions = np.zeros(len(decision_ovo))
    class_pos, class_neg = ovo_class_combinations(n_classes)

    counter = np.zeros([len(decision_ovo), n_classes])

    for p in range(len(decision_ovo)):
        for i in range(len(decision_ovo[p])):
            if decision_ovo[p,i] > 0:
                counter[p, class_pos[i]] += 1
            else:
                counter[p, class_neg[i]] += 1

        predictions[p] = np.argmax(counter[p])

    return predictions, counter
def ovo_voting_exp(decision_ovo, n_classes):
    predictions = np.zeros(len(decision_ovo))
    class_pos, class_neg = ovo_class_combinations(n_classes)

    counter = np.zeros([len(decision_ovo), n_classes])

    for p in range(len(decision_ovo)):
        for i in range(len(decision_ovo[p])):
            counter[p, class_pos[i]] += 1 / (1 + np.exp(-decision_ovo[p,i]) )
            counter[p, class_neg[i]] += 1 / (1 + np.exp( decision_ovo[p,i]) )

        predictions[p] = np.argmax(counter[p])

    return predictions, counter

def basic_rules(probs_ensemble, rule_index):
    n_ensembles, n_instances, n_classes = probs_ensemble.shape
    predictions_rule = np.zeros(n_instances)

    # Product rule
    if rule_index == 0:
        probs_rule = np.ones([n_instances, n_classes])

        for p in range(n_instances):
            for e in range(n_ensembles):
                probs_rule[p] = probs_rule[p] * probs_ensemble[e, p]
            predictions_rule[p] = np.argmax(probs_rule[p])

    # Sum rule
    elif rule_index == 1:
        probs_rule = np.zeros([n_instances, n_classes])

        for p in range(n_instances):
            for e in range(n_ensembles):
                probs_rule[p] = probs_rule[p] + probs_ensemble[e, p]
            predictions_rule[p] = np.argmax(probs_rule[p])

    # Minimum rule
    elif rule_index == 2:
        probs_rule = np.ones([n_instances, n_classes])

        for p in range(n_instances):
            for e in range(n_ensembles):
                probs_rule[p] = np.minimum(probs_rule[p], probs_ensemble[e, p])
            predictions_rule[p] = np.argmax(probs_rule[p])

    # Maximum rule
    elif rule_index == 3:
        probs_rule = np.zeros([n_instances, n_classes])

        for p in range(n_instances):
            for e in range(n_ensembles):
                probs_rule[p] = np.maximum(probs_rule[p], probs_ensemble[e, p])
            predictions_rule[p] = np.argmax(probs_rule[p])

    # Majority rule
    elif rule_index == 4:
        rank_rule = np.zeros([n_instances, n_classes])
        # Just simply adds the position of the ranking
        for p in range(n_instances):

            for e in range(n_ensembles):
                rank = np.argsort(probs_ensemble[e, p])
                for j in range(n_classes):
                    rank_rule[p, rank[j]] = rank_rule[p, rank[j]] + j
            predictions_rule[p] = np.argmax(rank_rule[p])

    return predictions_rule