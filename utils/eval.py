from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def cal_precision_and_recall(gt_label, pred_label):
    precision = defaultdict(int)
    recall = defaultdict(int)
    total = defaultdict(int)
    for t_lab, p_lab in zip(gt_label, pred_label):
        total[t_lab] += 1  # TP + FN
        recall[p_lab] += 1  # TP + FP
        if t_lab == p_lab:
            precision[t_lab] += 1

    result = dict()
    for key in sorted(precision):
        pre = float(precision[key]) / float(recall[key])
        rec = float(precision[key]) / float(total[key])
        F1 = (2 * pre * rec) / (pre + rec)
        result[key] = [pre, rec, F1]
    gt_label = np.array(gt_label)
    pred_label = np.array(pred_label)
    acc_avg = float(sum(gt_label == pred_label)) / float(len(gt_label))
    return result, acc_avg


def plot_confusion_matrix(
        cm,
        target_names,
        title='Confusion matrix',
        filepath='ConfusionMatrix.jpg',
        cmap=None,
        normalize=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i,j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i,j]),
                     color="white" if cm[i,j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(filepath, format='jpg')
    # plt.show()


def cal_confusion_matrix(gt_label, pred_label):
    # sns.set()
    cm = confusion_matrix(gt_label, pred_label, labels=None, sample_weight=None)
    # sns.headmap(cm, annot=True)
    return cm


if __name__ == "__main__":
    gt_label = [1, 0, 1, 0, 1, 1]
    pred_label = [1, 1, 1, 0, 0, 0]
    result, acc = cal_precision_and_recall(gt_label, pred_label)
    print(result, acc)
