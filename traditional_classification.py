#nltk.download('averaged_perceptron_tagger')
from functools import reduce

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
# import pylab as pl
from sklearn.model_selection import train_test_split
# from data import load_corpus
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interp

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer, label_binarize

from data import load_corpus

def print_and_plot_confusion_matrix(y_true, y_pred, classes,
                                    normalize=False,
                                    title=None,
                                    cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    elif title and normalize:
        title = "Normalized " + title

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]

    print('Confusion matrix, without normalization')
    print(cm)
    plot_confusion_matrix(cm, cmap, classes, title, False)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # cm = cm.round(decimals=2)
        print("Normalized confusion matrix")
        print(cm)
        plot_confusion_matrix(cm, cmap, classes, title, normalize)


def plot_confusion_matrix(cm, cmap, classes, title, normalize):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    return ax


def class_report(y_true, y_pred, y_score=None, average='micro'):
    if y_true.shape != y_pred.shape:
        print("Error! y_true %s is not the same shape as y_pred %s" % (
              y_true.shape,
              y_pred.shape)
        )
        return

    lb = LabelBinarizer()

    if len(y_true.shape) == 1:
        lb.fit(y_true)

    #Value counts of predictions
    labels, cnt = np.unique(
        y_pred,
        return_counts=True)
    n_classes = len(labels)
    pred_cnt = pd.Series(cnt, index=labels)

    metrics_summary = precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels)

    avg = list(precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index,
        columns=labels)

    support = class_report_df.loc['support']
    total = support.sum()
    class_report_df['avg / total'] = avg[:-1] + [total]

    class_report_df = class_report_df.T
    class_report_df['pred'] = pred_cnt
    class_report_df['pred'].iloc[-1] = total

    if not (y_score is None):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for label_it, label in enumerate(labels):
            fpr[label], tpr[label], _ = roc_curve(
                (y_true == label).astype(int),
                y_score[:, label_it])

            roc_auc[label] = auc(fpr[label], tpr[label])

        if average == 'micro':
            if n_classes <= 2:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                    lb.transform(y_true).ravel(),
                    y_score[:, 1].ravel())
            else:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                        lb.transform(y_true).ravel(),
                        y_score.ravel())

            roc_auc["avg / total"] = auc(
                fpr["avg / total"],
                tpr["avg / total"])

        elif average == 'macro':
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([
                fpr[i] for i in labels]
            ))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in labels:
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr

            roc_auc["avg / total"] = auc(fpr["macro"], tpr["macro"])

        class_report_df['AUC'] = pd.Series(roc_auc)

    return class_report_df


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)


def report_average(*args):
    report_list = list()
    cr = {'caution_and_advice': [0.0 for i in range(4)], 'displaced_people_and_evacuations': [0.0 for i in range(4)], 'donation_needs_or_offers_or_volunteering_services': [0.0 for i in range(4)], 'infrastructure_and_utilities_damage': [0.0 for i in range(4)],
          'injured_or_dead_people': [0.0 for i in range(4)], 'missing_trapped_or_found_people': [0.0 for i in range(4)], 'not_related_or_irrelevant': [0.0 for i in range(4)], 'other_useful_information': [0.0 for i in range(4)], 'sympathy_and_emotional_support': [0.0 for i in range(4)]}
    for report in args[0]:
        splited = [' '.join(x.split()) for x in report.split('\n\n')]
        class_split = splited[1].split(' ')
        accavg_split = splited[2].split(' ')

        # prec, rec and f1 per class
        for i in range(0, len(class_split), 5):
            if class_split[i] in cr:
                cr[class_split[i]] = [x + y for x, y in zip(cr[class_split[i]], list(map(float, class_split[i+1:i+5])))]
            else:
                cr[class_split[i]] = list(map(float, class_split[i+1:i+5]))

        # acc and support
        if accavg_split[0] in cr:
            cr[accavg_split[0]] = [x + y for x, y in zip(cr[accavg_split[0]], list(map(float, accavg_split[1:3])))]
        else:
            cr[accavg_split[0]] = list(map(float, accavg_split[1:3]))

        # macro and weighted avg prec, rec, f1
        avg_split = accavg_split[3:]
        for i in range(0, len(avg_split), 6):
            avgtype = avg_split[i] + "_" + avg_split[i + 1]
            if avgtype in cr:
                cr[avgtype] = [x + y for x, y in zip(cr[avgtype], list(map(float, avg_split[i+2:i+6])))]
            else:
                cr[avgtype] = list(map(float, avg_split[i+2:i+6]))

    # import collections
    # cr = collections.OrderedDict(sorted(cr.items()))
    df = pd.DataFrame(columns=['class', 'precision', 'recall', 'f1-score ', 'support'])
    for i, c in enumerate(cr):
        if c == "accuracy":
            cr[c][0] = '%.3f' % np.divide(cr[c][0], len(args[0]))
            #support
            cr[c][1] = int(cr[c][1])

            #add to df
            df.loc[i] = [c, ".", "."] + list(cr[c])
        else:
            cr[c][:3] = ['%.3f' % (val / len(args[0])) for val in cr[c][:3]]# np.divide(cr[c][:3], len(args[0]))
            #support
            cr[c][3] = int(cr[c][3])

            # add to df
            df.loc[i] = [c] + list(cr[c])

    print(df.to_string(index=False))

    return cr


def run_model_per_dataset(datasets):

    for dataset in datasets:
        print(list(dataset.keys())[0])
        trainset = list(dataset.values())[0]

        word_vector = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), binary=False, max_features=1000)
        char_vector = TfidfVectorizer(ngram_range=(2, 3), analyzer="char", binary=False, min_df=9, max_features=1000)

        # combine word-char ngrams features
        vectorizer = FeatureUnion([("chars", char_vector), ("words", word_vector)])

        # corpus for storing tweet texts and classes for labels
        corpus = []
        classes = []

        for i, item in enumerate(trainset):
            corpus.append(item['text'])
            classes.append(item['label'])

        unique_classes = sorted(list(set(classes)))
        n_classes = len(unique_classes)

        print("num of training instances: ", len(classes))
        print("num of training classes: ", len(set(classes)))

        # vectorise the corpus to tf-idf matrix
        matrix = vectorizer.fit_transform(corpus)

        print("num of features: ", len(vectorizer.get_feature_names()))

        print("training model")
        X = matrix.toarray()
        y = np.asarray(classes)

        print("Support: ", len(X)/10)
        kf = KFold(n_splits=10, random_state=212, shuffle=True)  # Define the split - into 10 folds
        kf.get_n_splits(X)  # returns the number of splitting iterations in the cross-validator
        print(kf)
        KFold(n_splits=10, random_state=None, shuffle=False)

        for model in [LinearSVC(loss='hinge', dual=True)]: # RandomForestClassifier(), MultinomialNB()

            prfs = [np.zeros(9), np.zeros(9), np.zeros(9), np.zeros(9).astype(int)]
            f1_scores_macro = []
            precisions_macro = []
            recalls_macro = []

            f1_scores_weighted = []
            precisions_weighted = []
            recalls_weighted = []

            class_reports = []
            accs = []
            model_name = type(model).__name__
            print(model_name)
            i = 1
            for train_index, test_index in kf.split(X):
                print(i)
                #model = GaussianNB() #LinearSVC(loss='hinge', dual=True) #OneVsRestClassifier(SVC(kernel='linear', probability=True)) #SVC(kernel='linear', probability=True) #KNeighborsClassifier() #RandomForestClassifier(max_depth=2, random_state=0)
                #print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                y_fit = model.fit(X_train, y_train)
                y_pred = y_fit.predict(X_test)

                class_rep = classification_report(y_test, y_pred, digits=10)
                class_reports.append(class_rep)
                print(class_rep)
                # print()
                acc = accuracy_score(y_test, y_pred)
                accs.append(acc)
                # print("Accuracy:", acc)
                # print()

                prf = precision_recall_fscore_support(y_test, y_pred, labels=unique_classes)
                prfs = [[i1+j1 for i1, j1 in zip(i,j)] for i, j in zip(prfs, prf)]

                #prfs = [x + y for i, (x, y) in enumerate(zip(prfs[i], list(map(float, prf[i]))))]
                f1_scores_macro.append(metrics.f1_score(y_test, y_pred, average='macro'))
                precisions_macro.append(metrics.precision_score(y_test, y_pred, average='macro'))
                recalls_macro.append(metrics.recall_score(y_test, y_pred, average='macro'))

                f1_scores_weighted.append(metrics.f1_score(y_test, y_pred, average='weighted'))
                precisions_weighted.append(metrics.precision_score(y_test, y_pred, average='weighted'))
                recalls_weighted.append(metrics.recall_score(y_test, y_pred, average='weighted'))

                i += 1

            print("avg from string")
            report_avg = report_average(class_reports)

            print()
            print("avg from arrays")
            prfs[:3] = [[item / 10 for item in prf] for prf in prfs[:3]]
            for l in range(n_classes):
                prf = "\t\t\t\t\t\t\t\t\t\t\t\t\t"
                for m in range(len(prfs)-1):
                    try:
                        prf += "\t" + str('%.3f' % prfs[m][l])
                    except:
                        print("An exception occurred")
                prf += "\t" + str(prfs[len(prfs)-1][l])
                print(prf)

            print()
            print("\t\t\t\t\t\t\t\t\t\tAccuracy:\t\t\t\t\t", "%.3f" % np.mean(accs))
            print("\t\t\t\t\t\t\t\t\t\tmacro avg:\t\t", "%.3f" % np.mean(precisions_macro), "%.3f" % np.mean(recalls_macro), "%.3f" % np.mean(f1_scores_macro))
            print("\t\t\t\t\t\t\t\t\t\tmacro avg:\t\t", "%.3f" % np.mean(precisions_weighted),
                  "%.3f" % np.mean(recalls_weighted), "%.3f" % np.mean(f1_scores_weighted))
            print("\t\t\t\t\t\t\t\t\t\tstd weighted f1:\t\t", "%.3f" % np.std(f1_scores_weighted))
            print("\n\n\n")

        print(classification_report(y_test, y_pred))
        print()
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print()

        print("\n=======================================================\n")

corpus = load_corpus(True, True, True, True, True, True)
run_model_per_dataset(corpus)
