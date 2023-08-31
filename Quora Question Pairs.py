# Import the required libraries
import numpy as np
import pandas as pd
import lightgbm as lgb

from nltk.corpus import stopwords
from collections import Counter
from sklearn.model_selection import train_test_split

# Retrieve data
df_train = pd.read_csv("./data/input/train.csv")
df_test = pd.read_csv("./data/input/test.csv")
# Combining data
df = pd.concat([df_train, df_test])

#df_train.shape (404290, 6)
#df_test.shape (2345796, 3)

# Preprocessing
train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
words = (" ".join(train_qs)).lower().split()

# Extract artificial features
## Features of Word-share
### Calculate weights for each word
def calc_weight(freq, eps=10000):
    if freq < 2:
        return 0
    return  1 / (freq + eps)
#  If a word appears only once, we ignore it completely (likely a typo)
#  Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller

### Calculate the idf of the word
def compute_word_weights(words):
    word_freq = Counter(words)
    return {word: calc_weight(freq) for word, freq in word_freq.items()}
weights = compute_word_weights(words)

#stopwords
stops = set(stopwords.words("english"))

# Features of Word-share
def word_shares(row):
    q1_list = str(row[3]).lower().split()
    q2_list = str(row[4]).lower().split()
    q1 = set(q1_list)
    q2 = set(q2_list)
    q1words = q1.difference(stops)
    q2words = q2.difference(stops)
    if len(q1words) == 0:
        return '0:0:0:0:0:0:0:0'
    if len(q2words) == 0:
        return '0:0:0:0:0:0:0:0'

    # stopwords
    q1stops = q1.intersection(stops)
    q2stops = q2.intersection(stops)

    # Total number of 2-grams of sentence 1 and 2-grams of sentence 2
    q1_2gram = set([i for i in zip(q1_list, q1_list[1:])])
    q2_2gram = set([i for i in zip(q2_list, q2_list[1:])])

    shared_2gram = q1_2gram.intersection(q2_2gram)
    shared_words = q1words.intersection(q2words)

    # idf weights for shared words
    shared_weights = [weights.get(w, 0) for w in shared_words]
    q1_weights = [weights.get(w, 0) for w in q1words]
    q2_weights = [weights.get(w, 0) for w in q2words]
    total_weights = q1_weights + q1_weights

    words_hamming = sum(1 for i in zip(q1_list, q2_list) if i[0] == i[1]) / max(len(q1_list), len(q2_list))

    denom1 = (len(q1_2gram) + len(q2_2gram))
    denom2 = (np.sqrt(np.dot(q1_weights,q1_weights))*np.sqrt(np.dot(q2_weights,q2_weights)))

    return '{}:{}:{}:{}:{}:{}:{}:{}'.format(
        np.sum(shared_weights) / np.sum(total_weights),
        len(shared_words) / (len(q1words) + len(q2words) - len(shared_words)),
        len(shared_words),
        len(q1stops) / len(q1words),
        len(q2stops) / len(q2words),
        len(shared_2gram) / denom1 if denom1 != 0 else 0,
        np.dot(shared_weights, shared_weights) / denom2,
        words_hamming
    )

# Execute the word-share function on each line
df['word_shares'] = df.apply(word_shares, axis=1, raw=True)

# Create a new dataframe and fill it with features
x = pd.DataFrame()
columns = ['word_match', 'tfidf_word_match', 'shared_count', 'stops1_ratio', 'stops2_ratio', 'shared_2gram', 'cosine',
           'words_hamming']
for idx, column in enumerate(columns):
    x[column] = df['word_shares'].apply(lambda x: float(x.split(':')[idx]))

# The difference in length between sentence 1 and sentence 2
def diff_len(df, x):
    x['len_q1'] = df['question1'].apply(lambda x: len(str(x)))
    x['len_q2'] = df['question2'].apply(lambda x: len(str(x)))
    x['diff_len'] = x['len_q1'] - x['len_q2']

# The difference in the number of capital letters
def diff_caps(df, x):
    x['caps_count_q1'] = df['question1'].apply(lambda x: sum(1 for i in str(x) if i.isupper()))
    x['caps_count_q2'] = df['question2'].apply(lambda x: sum(1 for i in str(x) if i.isupper()))
    x['diff_caps'] = x['caps_count_q1'] - x['caps_count_q2']

# The difference in the number of characters
def diff_len_char(df, x):
    x['len_char_q1'] = df['question1'].apply(lambda x: len(str(x).replace(' ', '')))
    x['len_char_q2'] = df['question2'].apply(lambda x: len(str(x).replace(' ', '')))
    x['diff_len_char'] = x['len_char_q1'] - x['len_char_q2']

# The difference in number of words
def diff_len_word(df, x):
    x['len_word_q1'] = df['question1'].apply(lambda x: len(str(x).split()))
    x['len_word_q2'] = df['question2'].apply(lambda x: len(str(x).split()))
    x['diff_len_word'] = x['len_word_q1'] - x['len_word_q2']

# The difference in average word lengths
def diff_avg_word(x):
    x['avg_world_len1'] = x['len_char_q1'] / x['len_word_q1']
    x['avg_world_len2'] = x['len_char_q2'] / x['len_word_q2']
    x['diff_avg_word'] = x['avg_world_len1'] - x['avg_world_len2']

# Other feature
def other_features(df, x):
    x['diff_stops_r'] = x['stops1_ratio'] - x['stops2_ratio']
    x['exactly_same'] = (df['question1'] == df['question2']).astype(int)
    x['duplicated'] = df.duplicated(['question1', 'question2']).astype(int)

# Frequency of common interrogative words
def interrogative_words(x, df, word):
    x['q1_' + word] = df['question1'].apply(lambda x: (word in str(x).lower()) * 1)
    x['q2_' + word] = df['question2'].apply(lambda x: (word in str(x).lower()) * 1)
    x[word + '_both'] = x['q1_' + word] * x['q2_' + word]

# Execute all functions
diff_len(df, x)
diff_caps(df, x)
diff_len_char(df, x)
diff_len_word(df, x)
diff_avg_word(x)
other_features(df, x)

for word in ['how', 'what', 'which', 'who', 'where', 'when', 'why']:
    interrogative_words(x, df, word)

# Split the training sets and test sets
x_train = x[:df_train.shape[0]]
x_test = x[df_train.shape[0]:]
y_train = df_train['is_duplicate'].values

# Create a function that returns the loss every 10 rounds
def print_evaluation(period):
    def callback(env):
        if period > 0 and env.iteration % period == 0:
            print("Loss after iteration %d: %f" % (env.iteration, env.evaluation_result_list[-1][2]))
    return callback

# Parameters of the LightGBM model
lgb_params = {
    "boosting": "gbdt",
    "objective":'binary',
    "metric": 'binary_logloss',
    "learning_rate": 0.14,
    "sub_feature":0.5,
    "num_leaves": 512,
    "min_data_in_leaf": 50,
    "min_hessian":1,
    "verbose": 10
}

if 1: # Now we oversample the negative class - on your own risk of overfitting!
    pos_train = x_train[y_train == 1]
    neg_train = x_train[y_train == 0]

    print("Oversampling started for proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))
    p = 0.165
    scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
    while scale > 1:
        neg_train = pd.concat([neg_train, neg_train])
        scale -=1
    neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
    print("Oversampling done, new proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))

    x_train = pd.concat([pos_train, neg_train])
    y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
    del pos_train, neg_train

X, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=12345)
x_train.fillna(0)
x_test.fillna(0)
xg_train = lgb.Dataset(X, label=y_train)

# Validation sets do an early stopping
xg_val = lgb.Dataset(X_val, label=y_val)
# Training Models
clr = lgb.train(lgb_params, xg_train, valid_sets=[xg_val], callbacks=[print_evaluation(period=10)])
y_pred = clr.predict(x_test)

# Output data
sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = y_pred
sub.to_csv("submission.csv", index=False)




