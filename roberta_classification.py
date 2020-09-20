import numpy as np
import os

import pandas as pd
from pandas import DataFrame
from simpletransformers.model import TransformerModel
import torch
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

from sklearn import metrics

from data import load_corpus

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "4";

torch.multiprocessing.freeze_support()

nr_folds = 2
nr_epochs = 3

def f1_micro_multiclass(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, average='micro')


def f1_macro_multiclass(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, average='macro')


def f1_weighted_multiclass(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, average='weighted')


def precision_macro_multiclass(y_true, y_pred):
    return metrics.precision_score(y_true, y_pred, average='macro')


def precision_weighted_multiclass(y_true, y_pred):
    return metrics.precision_score(y_true, y_pred, average='weighted')


def recall_macro_multiclass(y_true, y_pred):
    return metrics.recall_score(y_true, y_pred, average='macro')


def recall_weighted_multiclass(y_true, y_pred):
    return metrics.recall_score(y_true, y_pred, average='weighted')


coded_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]


def prf_milticlass(y_true, y_pred):
    return precision_recall_fscore_support(y_true, y_pred, labels=coded_classes)


def calss_report(y_true, y_pred):
    cr = classification_report(y_true, y_pred, digits=10)
    print(cr)
    return cr


def report_average(*args):
    report_list = list()
    cr = {'0': [0.0 for i in range(4)], '1': [0.0 for i in range(4)], '2': [0.0 for i in range(4)],
          '3': [0.0 for i in range(4)],
          '4': [0.0 for i in range(4)], '5': [0.0 for i in range(4)], '6': [0.0 for i in range(4)],
          '7': [0.0 for i in range(4)], '8': [0.0 for i in range(4)]}
    for report in args[0]:
        splited = [' '.join(x.split()) for x in report.split('\n\n')]
        class_split = splited[1].split(' ')
        accavg_split = splited[2].split(' ')

        # prec, rec and f1 per class
        for i in range(0, len(class_split), 5):
            if class_split[i] in cr:
                cr[class_split[i]] = [x + y for x, y in
                                      zip(cr[class_split[i]], list(map(float, class_split[i + 1:i + 5])))]
            else:
                cr[class_split[i]] = list(map(float, class_split[i + 1:i + 5]))

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
                cr[avgtype] = [x + y for x, y in zip(cr[avgtype], list(map(float, avg_split[i + 2:i + 6])))]
            else:
                cr[avgtype] = list(map(float, avg_split[i + 2:i + 6]))

    # import collections
    # cr = collections.OrderedDict(sorted(cr.items()))
    df = pd.DataFrame(columns=['class', 'precision', 'recall', 'f1-score ', 'support'])
    for i, c in enumerate(cr):
        if c == "accuracy":
            cr[c][0] = '%.3f' % np.divide(cr[c][0], len(args[0]))
            # support
            cr[c][1] = int(cr[c][1])

            # add to df
            df.loc[i] = [c, ".", "."] + list(cr[c])
        else:
            cr[c][:3] = ['%.3f' % (val / len(args[0])) for val in cr[c][:3]]  # np.divide(cr[c][:3], len(args[0]))
            # support
            cr[c][3] = int(cr[c][3])

            # add to df
            df.loc[i] = [c] + list(cr[c])

    print(df.to_string(index=False))

    return cr, df.to_string(index=False)


def run():

    # datasets = load_corpus(True, True, True, True, True, True)
    datasets = load_corpus(False, False, False, False, False, False)

    for dataset in datasets:
        datasetname = list(dataset.keys())[0]
        print(datasetname)
        trainset = list(dataset.values())[0]

        df = DataFrame(data=trainset)
        df = df[['text', 'label']]

        le = LabelEncoder()
        le.fit(df["label"])
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(le_name_mapping)
        df["label"] = le.transform(df["label"])
        # df["label"] = numpy.int64(le.fit_transform(df["label"]))

        kf = KFold(n_splits=nr_folds, random_state=212, shuffle=True)  # Define the split - into 10 folds
        kf.get_n_splits(df)  # returns the number of splitting iterations in the cross-validator
        print(kf)

        prfs = [np.zeros(9), np.zeros(9), np.zeros(9), np.zeros(9).astype(int)]
        f1_scores_macro = []
        precisions_macro = []
        recalls_macro = []

        f1_scores_weighted = []
        precisions_weighted = []
        recalls_weighted = []

        teller = 1
        accs = []
        class_reports = []
        f = open("results_bert.txt", "a")
        f.write(datasetname + "\n\n")
        f.close()
        for train_index, test_index in kf.split(df):
            model = TransformerModel('roberta', 'roberta-base', num_labels=9,
                                     args={'learning_rate': 1e-5, 'num_train_epochs': 1, 'reprocess_input_data': True,
                                           'overwrite_output_dir': True, 'fp16': False}, use_cuda=False)
            print("Fold: ", teller)
            print("Train indices: ", train_index, " Test indices", test_index)

            train_df, eval_df = df.iloc[train_index], df.iloc[test_index]
            model.train_model(train_df)

            result, model_outputs, wrong_predictions = model.eval_model(eval_df, f1_macro=f1_macro_multiclass,
                                                                        f1_weighted=f1_weighted_multiclass,
                                                                        prec_macro=precision_macro_multiclass,
                                                                        prec_weighted=precision_weighted_multiclass,
                                                                        rec_macro=recall_macro_multiclass,
                                                                        rec_weighted=recall_weighted_multiclass,
                                                                        prf=prf_milticlass, acc=accuracy_score,
                                                                        cr=calss_report)
            true_labels = eval_df["label"].values
            predictions, raw_outputs = model.predict(eval_df["text"].values)

            print(result)
            print(model_outputs)
            print(wrong_predictions)

            accs.append(result["acc"])
            class_reports.append(result["cr"])

            prf = result["prf"]
            prfs = [[i1 + j1 for i1, j1 in zip(i, j)] for i, j in zip(prfs, prf)]

            # prfs = [x + y for i, (x, y) in enumerate(zip(prfs[i], list(map(float, prf[i]))))]
            f1_scores_macro.append(result["f1_macro"])
            precisions_macro.append(result["prec_macro"])
            recalls_macro.append(result["rec_macro"])

            f1_scores_weighted.append(result["f1_weighted"])
            precisions_weighted.append(result["prec_weighted"])
            recalls_weighted.append(result["rec_weighted"])

            f = open("results_bert.txt", "a")
            f.write("Iter: " + str(teller) + "\n")
            f.write(result["cr"])
            f.write("\n\nAcc: " + str(result["acc"]) + "\n\n")
            f.close()

            teller += 1

        print(le_name_mapping)

        cr, report_avg = report_average(class_reports)
        report_avg = "\n\n- From classification reports:\n\n" + report_avg
        print("Avg. Accuracy:", np.mean(accs))

        print()
        print("avg from arrays")
        prfs[:3] = [[item / nr_folds for item in prf] for prf in prfs[:3]]
        prf_rep = "\n\n- From managed arrays using sklearn metrics:\n\n"
        for l in range(9):
            prf = "\t"
            for m in range(len(prfs) - 1):
                try:
                    prf += "\t" + str('%.3f' % prfs[m][l])
                except:
                    print("An exception occurred")
            prf += "\t" + str(prfs[len(prfs) - 1][l])
            print(prf)
            prf_rep += "\n" + prf

        print()
        mean_acc = "Avg. accuracy:\t\t\t" + ("%.3f" % np.mean(accs))
        prf_rep += "\n\n" + mean_acc
        print(mean_acc)
        macro_avg = "Macro avg:\t" + ("%.3f" % np.mean(precisions_macro)) + "\t" + (
                    "%.3f" % np.mean(recalls_macro)) + "\t" + ("%.3f" % np.mean(f1_scores_macro))
        prf_rep += "\n" + macro_avg
        print(macro_avg)
        weighted_avg = "Weighted avg:\t" + ("%.3f" % np.mean(precisions_weighted)) + "\t" + (
                    "%.3f" % np.mean(recalls_weighted)) + "\t" + ("%.3f" % np.mean(f1_scores_weighted))
        prf_rep += "\n" + macro_avg + "\n"
        print(weighted_avg)
        print("\n\n\n")

        f = open("results_bert.txt", "a")
        f.write("\n\n-------------------\n")
        f.write(report_avg)
        f.write(prf_rep)
        # f.write("\n\nAvg. Accuracy: " + str(np.mean(accs)))
        f.write("\n\n#######################################\n\n")
        f.close()


if __name__ == '__main__':
    run()