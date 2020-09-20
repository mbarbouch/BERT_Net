import os

import numpy as np
import csv

import pandas as pd
import torch
from pandas import DataFrame
from simpletransformers.model import TransformerModel
from sklearn import metrics

from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import label_binarize, LabelEncoder, FunctionTransformer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from data import load_corpus


def run():
    nr_folds = 10
    nr_epochs = 4

    def report_average_num(*args):
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

    def report_average(*args):
        report_list = list()
        cr = {'caution_and_advice': [0.0 for i in range(4)],
              'displaced_people_and_evacuations': [0.0 for i in range(4)],
              'donation_needs_or_offers_or_volunteering_services': [0.0 for i in range(4)],
              'infrastructure_and_utilities_damage': [0.0 for i in range(4)],
              'injured_or_dead_people': [0.0 for i in range(4)],
              'missing_trapped_or_found_people': [0.0 for i in range(4)],
              'not_related_or_irrelevant': [0.0 for i in range(4)], 'other_useful_information': [0.0 for i in range(4)],
              'sympathy_and_emotional_support': [0.0 for i in range(4)]}
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

        return cr

    datasets = load_corpus(True, True, True, True, True, True)

    for dataset in datasets:
        dsname = list(dataset.keys())[0].lower()
        print(dsname)

        dsnameparts = dsname.split("_")

        part_dsname = dsnameparts[0] + "_" + dsnameparts[1]

        trainset = list(dataset.values())[0]
        word_vector = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), binary=False, max_features=1000)
        char_vector = TfidfVectorizer(ngram_range=(2, 3), analyzer="char", binary=False, min_df=9, max_features=1000)

        rnkfile = None
        all_files = os.listdir(os.getcwd())
        for i, filename in enumerate(all_files):
            if os.path.isfile(os.getcwd() + "/" + filename) and filename.lower().startswith(part_dsname):
                rnkfile = filename
                break

        df = pd.read_csv(rnkfile, sep="\t", header=0, low_memory=False, quoting=csv.QUOTE_NONE).fillna(0)
        # Indegree
        # featuresToUsePre = ['ID', 'User', 'FollowersCount', 'Indegree', 'Label']

        featuresToUse = ['ScreenNameResponse', 'User', 'SVM', 'BERT', 'EntitiesMentions', 'EntitiesHashtags',
                         'NumCharacters', 'Indegree', 'Betweenness', 'Closeness', 'Eigenvector', 'Label']
        # df = df[featuresToUsePre]
        pd.set_option('display.max_columns', None)
        # , 'UserActiveDays', 'FavoriteCount', "NumCharacters", "NumWords"
        # print(df)
        # ur vectors are the feature union of word/char ngrams
        vectorizer = FeatureUnion([("chars", char_vector), ("words", word_vector)])

        ids = []

        # corpus is a list with the n-word chunks
        corpus = []
        # classes is the labels of each chunk
        classes = []

        test_indices = []

        for i, item in enumerate(trainset):
            ids.append(item['id'])
            corpus.append(item['text'])
            classes.append(item['label'])

        print("num of training instances: ", len(classes))
        print("num of training classes: ", len(set(classes)))

        # fit the model of tfidf vectors for the coprus
        matrix = vectorizer.fit_transform(corpus)
        print("num of features: ", len(vectorizer.get_feature_names()))

        print("training model")
        X = matrix.toarray()
        y = np.asarray(classes)

        n_classes = 9

        print("Support: ", len(X))
        kf = KFold(n_splits=nr_folds, random_state=212, shuffle=True)  # Define the split - into 10 folds
        kf.get_n_splits(X)  # returns the number of splitting iterations in the cross-validator

        model1 = LinearSVC(loss='hinge', dual=True)

        class_reports = []
        accs = []
        twt_pred = {}
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            test_ids = (np.array(ids))[test_index]

            y_fit = model1.fit(X_train, y_train)
            y_pred = y_fit.predict(X_test)

            for id, pred in zip(test_ids, y_pred):
                twt_pred[int(id)] = pred

            # print(y_pred)
            # print()
            # print(y_test)

            class_rep = classification_report(y_test, y_pred, digits=10)
            class_reports.append(class_rep)
            # print(class_rep)
            print()
            acc = accuracy_score(y_test, y_pred)
            accs.append(acc)
            # print("Accuracy:", acc)
            print()

        report_avg = report_average(class_reports)
        print("Accuracy:", np.mean(accs))
        print("\n\n\n")

        ###################################### RoBERTa

        torch.multiprocessing.freeze_support()

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

        def report_average_brt(*args):
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
                    cr[accavg_split[0]] = [x + y for x, y in
                                           zip(cr[accavg_split[0]], list(map(float, accavg_split[1:3])))]
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
                    cr[c][:3] = ['%.3f' % (val / len(args[0])) for val in
                                 cr[c][:3]]  # np.divide(cr[c][:3], len(args[0]))
                    # support
                    cr[c][3] = int(cr[c][3])

                    # add to df
                    df.loc[i] = [c] + list(cr[c])

            print(df.to_string(index=False))

            return cr, df.to_string(index=False)

        datasets2 = load_corpus(False, False, False, False, False, False, part_dsname)

        twt_pred_brt = {}
        dfb_ = None
        for dataset in datasets2:
            datasetname = list(dataset.keys())[0]
            print(datasetname)
            trainset = list(dataset.values())[0]

            dfb_ = DataFrame(data=trainset)
            dfb_['id'] = pd.to_numeric(dfb_['id'])
            # print(dfb_['id'].values.tolist())

            idtmp = df['ID'].values.tolist()
            dfcompt = dfb_[~dfb_['id'].isin(idtmp)]
            print(len(df))
            print(len(dfb_))
            print(len(dfcompt))

            ids_brt = dfb_[['id']].values
            dfb = dfb_[['text', 'label']]

            print("To train")
            print(len(dfb))

            le = LabelEncoder()
            le.fit(dfb["label"])
            le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            print(le_name_mapping)
            dfb["label"] = le.transform(dfb["label"])
            # df["label"] = numpy.int64(le.fit_transform(df["label"]))

            kf = KFold(n_splits=nr_folds, random_state=212, shuffle=True)
            kf.get_n_splits(df)
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
            f = open("results_bert_ensem.txt", "a")
            f.write(datasetname + "\n\n")
            f.close()
            for train_index, test_index in kf.split(dfb):
                model = TransformerModel('roberta', 'roberta-base', num_labels=9,
                                         args={'learning_rate': 1e-5, 'num_train_epochs': nr_epochs,
                                               'reprocess_input_data': True,
                                               'overwrite_output_dir': True, 'fp16': False})
                print("Fold: ", teller)
                print("Train indices: ", train_index, " Test indices", test_index)

                train_df, eval_df = dfb.iloc[train_index], dfb.iloc[test_index]
                test_ids = (np.array(ids_brt))[test_index]

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

                # print("First k predictions:")
                # for i in range(10):
                #     print(str(predictions[i]) + ": " + str(true_labels[i]))

                for id, pred in zip(test_ids, predictions):
                    twt_pred_brt[int(id)] = pred

                # print(result)
                # print(model_outputs)
                # print(wrong_predictions)

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

                # f = open("results_bert.txt", "a")
                # f.write("Iter: " + str(teller) + "\n")
                # f.write(result["cr"])
                # f.write("\n\nAcc: " + str(result["acc"]) + "\n\n")
                # f.close()

                teller += 1

            print(le_name_mapping)

            cr, report_avg = report_average_brt(class_reports)
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

            # f = open("results_bert.txt", "a")
            # f.write("\n\n-------------------\n")
            # f.write(report_avg)
            # f.write(prf_rep)
            # # f.write("\n\nAvg. Accuracy: " + str(np.mean(accs)))
            # f.write("\n\n#######################################\n\n")
            # f.close()

        ########################################### ldf

        df.insert(1, 'SVM', "")
        df.insert(2, 'BERT', -1)
        # df.insert(3, 'SVMBERT', -1)
        # df.insert(4, 'SVMENC', -1)
        # df = df.astype({'BERT': int})

        dfids = df["ID"].tolist()

        filtered_preds = {}
        for k, v in twt_pred.items():
            if k in dfids:
                filtered_preds[k] = v

        filtered_preds_b = {}
        for k, v in twt_pred_brt.items():
            if k in dfids:
                filtered_preds_b[k] = v

        pd.set_option('display.max_columns', None)
        for id, pred in filtered_preds.items():
            # pred.
            df.at[df['ID'] == id, "SVM"] = str(pred)
            # print(df.loc[df['ID'] == id])

        # print(len(df.loc[df['SVM'] != ""]))

        for id, pred in filtered_preds_b.items():
            # pred.
            df.at[df['ID'] == id, "BERT"] = pred

        df = df[featuresToUse]

        # check mapping label numbers and their coressponding classes
        le = LabelEncoder()
        le.fit(df["Label"].values.tolist())
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(le_name_mapping)

        """
        Zet strings om naar integers
        """
        for column in df.columns:
            if df[column].dtype == type(object):
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column].astype(str))

        # df = df[featuresToUse]
        X2 = df[df.columns[:-1]]
        # print(X2)
        y2 = df["Label"]

        # translate rows to dicts
        def row_to_dict(X, y=None):
            return X.apply(dict, axis=1)

        # define ensemble prediction model
        top_classifier = RandomForestClassifier(random_state=212)

        # define features to combine
        svm = pd.get_dummies(X2["SVM"], prefix="svm")
        ber = pd.get_dummies(X2["BERT"], prefix="ber")
        snp = pd.get_dummies(X2["ScreenNameResponse"], prefix="snp")
        usr = pd.get_dummies(X2["User"], prefix="usr")
        df["EntitiesMentions"] = df["EntitiesMentions"].astype(str)
        men = df["EntitiesMentions"].str.get_dummies(sep=',').add_prefix('men_')
        men = men.drop('men_Null', 1)
        X2 = pd.concat([ber, svm, X2["Betweenness"], X2["Closeness"], X2["Indegree"], X2["Eigenvector"]], axis=1)

        subfeatures = X2.columns

        # print(X2)

        kf2 = KFold(n_splits=nr_folds, random_state=212, shuffle=True)  # Define the split - into 10 folds

        kf2.get_n_splits(X2)

        class_reports2 = []
        accs2 = []
        impdicts = {"ScreenNameResponse": 0.0, "SVM": 0.0, "BERT": 0.0, "Indegree": 0.0, "Betweenness": 0.0,
                    "Closeness": 0.0, "Eigenvector": 0.0}
        for train_index, test_index in kf.split(X2):
            X_train, X_test = X2.iloc[train_index], X2.iloc[test_index]
            y_train, y_test = y2.iloc[train_index], y2.iloc[test_index]

            y_fit = top_classifier.fit(X_train, y_train)
            y_pred = y_fit.predict(X_test)

            class_rep = classification_report(y_test, y_pred, digits=10)
            class_reports2.append(class_rep)
            # print(class_rep)
            print()
            acc = accuracy_score(y_test, y_pred)
            accs2.append(acc)
            # print("Accuracy:", acc)
            print()

            impdict = {"ScreenNameResponse": 0.0, "BERT": 0.0, "SVM": 0.0, "Indegree": 0.0, "Betweenness": 0.0,
                       "Closeness": 0.0, "Eigenvector": 0.0}

            print("Feature importance")
            fi_sorted = sorted(zip(subfeatures, top_classifier.feature_importances_), key=lambda x: x[1], reverse=True)
            for fi in fi_sorted:
                print(fi[0] + "\t" + str(round(fi[1], 3)))

            fis = zip(subfeatures, top_classifier.feature_importances_)
            for fi in fis:
                if fi[0][:3] == "snp":
                    impdict["ScreenNameResponse"] += fi[1]
                elif fi[0][:3] == "svm":
                    impdict["SVM"] += fi[1]
                elif fi[0][:3] == "ber":
                    impdict["BERT"] += fi[1]
                elif fi[0] == "Indegree":
                    impdict["Indegree"] += fi[1]
                elif fi[0] == "Betweenness":
                    impdict["Betweenness"] += fi[1]
                elif fi[0] == "Closeness":
                    impdict["Closeness"] += fi[1]
                elif fi[0] == "Eigenvector":
                    impdict["Eigenvector"] += fi[1]

            impdicts["ScreenNameResponse"] += impdict["ScreenNameResponse"]
            impdicts["SVM"] += impdict["SVM"]
            impdicts["BERT"] += impdict["BERT"]
            impdicts["Indegree"] += impdict["Indegree"]
            impdicts["Betweenness"] += impdict["Betweenness"]
            impdicts["Closeness"] += impdict["Closeness"]
            impdicts["Eigenvector"] += impdict["Eigenvector"]

            # print(impdict)
            print("")

        impdicts = {k: v / nr_folds for k, v in impdicts.items()}
        print(impdicts)
        cr, crstr = report_average_num(class_reports2)
        print("Accuracy2:", np.mean(accs2))
        print("\n\n\n")

        f = open("results_bert_ensem.txt", "a")
        f.write("\n\n-------------------\n")
        f.write(crstr)
        f.write("")
        f.write("Feature importance")
        f.write(str(impdicts))
        # f.write(prf_rep)
        # f.write("\n\nAvg. Accuracy: " + str(np.mean(accs)))
        f.write("\n\n#######################################\n\n")
        f.close()


if __name__ == '__main__':
    run()
