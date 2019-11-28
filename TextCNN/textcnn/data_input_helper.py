import numpy as np
import re
import word2vec


class w2v_wrapper:
     def __init__(self,file_path):
        self.model = word2vec.load(file_path)
        if 'unknown' not  in self.model.vocab_hash:
            unknown_vec = np.random.uniform(-0.1,0.1,size=128)
            self.model.vocab_hash['unknown'] = len(self.model.vocab)
            self.model.vectors = np.row_stack((self.model.vectors,unknown_vec))


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def read_file_lines(filename,from_size,line_num):
    i = 0
    text = []
    end_num = from_size + line_num
    for line in open(filename):
        if(i >= from_size):
            text.append(line.strip())

        i += 1
        if i >= end_num:
            return text

    return text



def load_data_and_labels(filepath,max_size = -1):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    train_datas = []

    with open(filepath, 'r', encoding='utf-8',errors='ignore') as f:
        train_datas = f.readlines()

    one_hot_labels = []
    x_datas = []
    for line in train_datas:
        parts = line.split('\t',1)
        if(len(parts[1].strip()) == 0):
            continue

        x_datas.append(parts[1])
        if parts[0].startswith('0') :
            one_hot_labels.append([0,1])
        else:
            one_hot_labels.append([1,0])

    print (' data size = ' ,len(train_datas))
    return [x_datas, np.array(one_hot_labels)]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def get_text_idx(text,vocab,max_document_length):
    text_array = np.zeros([len(text), max_document_length],dtype=np.int32)

    for i,x in  enumerate(text):
        words = x.split(" ")
        for j, w in enumerate(words):
            if w in vocab:
                text_array[i, j] = vocab[w]
            else :
                text_array[i, j] = vocab['unknown']

    return text_array


if __name__ == "__main__":
    x_text, y = load_data_and_labels('/Users/slade/Documents/GitHub/sentiment_analysis_textcnn/data/cutclean_mine.txt')