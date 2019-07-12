import numpy as np
import re
import itertools
from collections import Counter

"""
Original taken from https://github.com/dennybritz/cnn-text-classification-tf
"""


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9ก-๙\|!]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    #string = re.sub(r",", " , ", string)
    #string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\...", " ",string)
    #string = re.sub(r"\!","",string)
    return string.strip().lower()


def load_data_and_labels(positive_file,negative_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_file ,encoding="utf8").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_file ,encoding="utf8").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split("|") for s in x_text]
    # Generate labels
    
    positive_labels = [[1, 0] for _ in positive_examples]
    negative_labels = [[0, 1] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def load_test_data_and_labels(vocabulary):
    positive_testdata = list(open("./data-clickbait/dataset-test-clickbait.txt", encoding="utf8").readlines())
    positive_testdata = [s.strip() for s in positive_testdata]
    negative_testdata = list(open("./data-clickbait/dataset-test-news.txt"  ,encoding="utf8").readlines())
    negative_testdata = [s.strip() for s in negative_testdata]
    x_text = positive_testdata + negative_testdata
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split("|") for s in x_text]
    positive_labels = [[1, 0] for _ in positive_testdata]
    negative_labels = [[0, 1] for _ in negative_testdata]
    y = np.concatenate([positive_labels, negative_labels], 0)
    sentence_padded = pad_sentences(x_text)
    y = np.array(y)
    x = build_input_data_for_sentences(sentence_padded, vocabulary) 
    return [x, y]

def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    #sequence_length = 169
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

def pad_sentence(sentence, sequence_length, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    padded_sentences = []
    num_padding = sequence_length - len(sentence)
    new_sentence = sentence + [padding_word] * num_padding
    padded_sentences.append(new_sentence)
    return padded_sentences

def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

def build_input_data_for_sentences(sentences, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary.get(word,0) for word in sentence] for sentence in sentences])
    return x

def load_data(positive_file,negative_file):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels(positive_file,negative_file)
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
def loaddict():
    loadedData=np.load('vocabulary-clickbait.out.npy').item()
    return loadedData	
def my_get_input_sentence():
    raw = input("input a news headline: ")
    raw_comment_cut = raw.split('|')
    raw_comment_cut = [clean_str(sent) for sent in raw_comment_cut]
    #print(raw_comment_cut)
    sentence_padded = pad_sentence(raw_comment_cut)
    #vocabulary, vocabulary_inv = build_vocab(sentence_padded)
    vocabulary = loaddict()
    x = build_input_data_for_sentences(sentence_padded, vocabulary)
    return x
def process_sentence(sentence):
    raw_comment_cut = sentence.split('|')
    raw_comment_cut = [clean_str(sent) for sent in raw_comment_cut]
    #print(raw_comment_cut)
    sentence_padded = pad_sentence(raw_comment_cut)
    #vocabulary, vocabulary_inv = build_vocab(sentence_padded)
    vocabulary = loaddict()
    x = build_input_data_for_sentences(sentence_padded, vocabulary)
    return x

def process_sentence_vocabulary(sentence,vocab , sequence_length):
    raw_comment_cut = sentence.split('|')
    raw_comment_cut = [clean_str(sent) for sent in raw_comment_cut]
    #print(raw_comment_cut)
    sentence_padded = pad_sentence(raw_comment_cut , sequence_length)
    #vocabulary, vocabulary_inv = build_vocab(sentence_padded)
    vocabulary = vocab
    x = build_input_data_for_sentences(sentence_padded, vocabulary)
    return x

