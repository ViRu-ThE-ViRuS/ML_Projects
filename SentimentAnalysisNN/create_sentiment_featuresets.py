import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import random
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
hm_lines = 100000


def create_lexicon(pos, neg):
    lexicon = []

    with open(pos, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            lexicon += list(word_tokenize(l))

    with open(neg, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            lexicon += list(word_tokenize(l))

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)

    filtered_lexicon = []
    for w in w_counts:
        if 1000 > w_counts[w] > 50:
            filtered_lexicon.append(w)

    print('Length of lexicon generated: ', len(filtered_lexicon))
    return filtered_lexicon


def sample_handling(sample, lexicon, classification):
    featureset = []

    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            l = l.lower()

            current_words = word_tokenize(l)
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))

            for word in current_words:
                if word in lexicon:
                    index_value = lexicon.index(word)
                    features[index_value] += 1

            features = list(features)
            featureset.append([features, classification])

    return featureset


def create_feature_sets_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg)

    features = []
    features += sample_handling('pos.txt', lexicon, [1,0])
    features += sample_handling('neg.txt', lexicon, [0,1])

    random.shuffle(features)
    features = np.array(features)

    testing_size = int(test_size * len(features))

    train_x = list(features[:, 0][:-testing_size])
    train_y = list(features[:, 1][:-testing_size])
    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')

    with open('sentiment_set.pickle','wb') as f:
        pickle.dump([train_x,train_y,test_x,test_y],f)
