import pandas as pd
import os
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

nlp = spacy.load("en_core_web_sm")


def load_data(filename, encoding='iso-8859-1'):
    if not os.path.exists(filename):
        print("Error, File not found.")
        return None

    df = pd.read_csv(filename, encoding=encoding)
    data_df = df[['v1', 'v2']]
    data_df.columns = ['Target', 'SMS']

    return data_df


def clean_msg(msg):
    tmp_msg = msg.lower()
    doc = nlp(tmp_msg)
    # if 'lem' in funcs:
    tokens = []
    for token in doc:
        if re.search(r"[0-9]{1,}", token.lemma_):
            tokens.append("aanumbers")
        else:
            word = "".join([ch for ch in token.lemma_ if ch not in string.punctuation])
            if len(word) > 1 and word not in STOP_WORDS:
                tokens.append(word)

    return " ".join(tokens)


def preprocess_data(df):
    new_df = pd.DataFrame()
    new_df['Target'] = df['Target']
    new_df["SMS"] = df["SMS"].apply(clean_msg)
    return new_df


class Bow:
    def __init__(self):
        self.vocab_list = []

    def bag_of_words(self, df, operation='train', index_train=0):
        """
        A function that returns a pd.Dataframe with words as columns
        and values as bag of words
        :param df: a pd.Dataframe
        :param operation: "train" or "test"
        :return: pd.Dataframe
        """

        if operation == 'train':
            vocab = set()
            for i, row in df[:index_train].iterrows():
                tokens = row['SMS'].split(" ")
                for t in tokens:
                    if len(t) > 1:
                        vocab.add(t)

            vocab_list = list(vocab)
            self.vocab_list = sorted(vocab_list)

        new_dict = {}

        for i, word in enumerate(self.vocab_list):
            new_dict[word] = [0.0 for i in range(df.shape[0])]

        df_op = pd.DataFrame()
        if operation == 'train':
            df_op = df[:index_train]
        else:
            df_op = df[index_train:]

        all_data = []
        iter_ = 0
        for i, row in df_op.iterrows():
            tokens = row['SMS'].split(" ")
            for t in tokens:
                if len(t) > 1 and t in new_dict.keys():
                    new_dict[t][iter_] += 1
            iter_ += 1

        result = df_op[['Target', 'SMS']].reset_index(drop=True)  # .reset_index(drop=True)
        return pd.concat([result,
                          pd.DataFrame(new_dict)],
                         axis=1)

    def bow_test_msg(self, msg):
        new_dict = {}
        for i, word in enumerate(self.vocab_list):
            new_dict[word] = [0.0]
            tokens = msg.split(" ")
            for t in tokens:
                if len(t) > 1 and t in new_dict.keys():
                    new_dict[t][0] += 1

        return pd.DataFrame(new_dict)


def get_words_sum(df):
    return df.drop(['SMS'], axis=1).groupby(['Target']).sum()


def get_cond_prob(df, laplace=1.0):
    sum_sep_df = df.drop(['SMS'], axis=1).groupby(['Target']).sum()
    sum_df = sum_sep_df.sum(axis=1)
    spam_total = sum_df.loc['spam']
    ham_total = sum_df.loc['ham']
    total = len(sum_sep_df.columns) * laplace

    for c in sum_sep_df.columns:
        sum_sep_df[c]['ham'] = \
            (sum_sep_df[c]['ham'] + laplace) / float(total + ham_total)

        sum_sep_df[c]['spam'] = \
            (sum_sep_df[c]['spam'] + laplace) / float(total + spam_total)  # sum_df.loc['spam'])

    sum_sep_df = sum_sep_df.T

    sum_sep_df.columns = ['Ham Probability', 'Spam Probability']
    return sum_sep_df[['Spam Probability', 'Ham Probability']]


def all_cond(df, operation='train', laplace=1):
    dataset = df.sample(df.shape[0], random_state=43)
    bow_c = Bow()
    df = pd.DataFrame()

    train_last_index = int(dataset.shape[0] * 0.8)
    if operation == 'train':
        bow_df = bow_c.bag_of_words(dataset[:train_last_index], operation='train',
                                    index_train=train_last_index)  # [0:train_last_index])
    else:
        bow_df = bow_c.bag_of_words(dataset[train_last_index:], operation='test',
                                    index_train=train_last_index)
    return get_cond_prob(bow_df, laplace=laplace)


def nb_predict(row, cond_prob_df):
    prob_ham = 1
    prob_spam = 1
    for t in row.split(" "):

        if len(t) > 1 and t in list(cond_prob_df.index.values.astype(str)):
            prob_ham *= cond_prob_df.loc[t]['Ham Probability']
            prob_spam *= cond_prob_df.loc[t]['Spam Probability']

    if prob_ham > prob_spam:
        return 'ham'
    elif prob_spam > prob_ham:
        return 'spam'
    return 'unknown'


def predict_test(df, laplace=1.0):
    dataset = df.sample(df.shape[0], random_state=43)

    df = pd.DataFrame()
    bow_c = Bow()

    train_last_index = int(dataset.shape[0] * 0.8)
    bow_train_df = bow_c.bag_of_words(dataset, operation='train',
                                      index_train=train_last_index)

    bow_test_df = bow_c.bag_of_words(dataset, operation='test',
                                     index_train=train_last_index)

    cond_prob_df = get_cond_prob(bow_test_df, laplace=laplace)

    result = dataset[train_last_index:]['SMS'].apply(lambda x: nb_predict(x, cond_prob_df))
    result_df = pd.DataFrame()
    result_df['Predicted'] = result
    result_df['Actual'] = dataset[train_last_index:]['Target']
    return result_df, cond_prob_df


def confusion_matrix(df):
    tn, tp, fn, fp = 0.0, 0.0, 0.0, 0.0

    for _, row in df.iterrows():
        if row['Predicted'] == row['Actual']:
            if row['Predicted'] == 'spam':
                tp += 1
            else:
                tn += 1
        elif row['Predicted'] == 'spam':
            fp += 1
        elif row['Predicted'] == 'ham':
            fn += 1

    return [[tn, fp],
            [fn, tp]]


def metrics_manual(df):
    (tn, fp), (fn, tp) = confusion_matrix(df)
    ms = {'Accuracy': (tp + tn) / (tp + tn + fp + fn),
          'Recall': tp / (tp + fn),
          'Precision': tp / (tp + fp)}

    ms['F1'] = 2 * (ms['Precision'] * ms['Recall']) / (ms['Precision'] + ms['Recall'])
    return ms


def metrics_sklearn(y, y_pred):
    ms = {'Accuracy': accuracy_score(y, y_pred),
          'Recall': recall_score(y, y_pred),
          'Precision': precision_score(y, y_pred),
          'F1': f1_score(y, y_pred)}
    return ms


def if_spam(s):
    return 1 if s == 'spam' else 0


def __main__():
    file = input("Please enter a filename : ")
    while not os.path.exists(file):
        file = input("Please enter a filename : ")

    sms_df = load_data(file)
    sms_df = preprocess_data(sms_df)
    dataset = sms_df.sample(sms_df.shape[0], random_state=43)

    bow_c = Bow()
    train_last_index = int(dataset.shape[0] * 0.8)
    bow_train = bow_c.bag_of_words(dataset, operation='train',
                                   index_train=train_last_index)
    bow_test = bow_c.bag_of_words(dataset, operation='test',
                                  index_train=train_last_index)
    while True:
        answer = input("Do you want to use sklearn or custom implementation? (y/n/exit) ").lower()
        if answer == "exit":
            break
        while answer.lower() in ('y', 'n'):
            if answer == 'y':
                bow_train['Target'] = bow_train['Target'].apply(if_spam)
                bow_test['Target'] = bow_test['Target'].apply(if_spam)
                clf = MultinomialNB()
                clf.fit(bow_train.drop(['SMS', 'Target'], axis=1), bow_train['Target'])
                X = bow_test.drop(['SMS', 'Target'], axis=1)
                result = clf.predict(X)
                y = bow_test['Target']
                print("Metrics for sklearn : ")
                print(metrics_sklearn(y, result))

                test_msg = input('Please enter a test msg : (msg/exit) : ')
                while test_msg != 'exit':
                    x_test = bow_c.bow_test_msg(test_msg)
                    result = clf.predict(x_test)
                    if result[0] == 1:
                        print("This msg could be a Spam.")
                    else:
                        print("This msg is probably Ham.")
                    test_msg = input('Please enter a test msg : (msg/exit) : ')

            else:
                preds, cond_prob_df = predict_test(sms_df, laplace=0.01)
                print("Metrics for custom implementation : ")
                print(metrics_manual(preds))
                test_msg = input('Please enter a test msg : (msg/exit) : ')
                while test_msg != 'exit':
                    # x_test = bow_c.bow_test_msg(test_msg)
                    preds = nb_predict(clean_msg(test_msg), cond_prob_df)
                    if preds == 'spam':
                        print("This msg could be a Spam.")
                    else:
                        print("This msg is probably Ham.")
                    test_msg = input('Please enter a test msg : (msg/exit) : ')

            answer = input("Do you want to continue? (y/n/exit) ").lower()
            if answer == "exit" or answer == 'n':
                return


__main__()
