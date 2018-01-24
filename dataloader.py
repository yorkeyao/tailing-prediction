import numpy as np
import copy
import math
import csv
from sklearn.model_selection import train_test_split

class Dis_dataloader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])

    def load_train_data(self, file):
        # Load data
        examples = []
        labels = []
        total_length = 0
        with open(file)as fin:
            f_csv = csv.reader(fin)
            cnt = 0
            tmp_examples = []
            tem_labels = []
            for row in f_csv:
                if cnt == 0:
                    cnt = cnt + 1
                    continue
                if cnt - 1 >= 360 and (cnt - 1) % 90 == 0:  #and (cnt - 1) % 180 == 0:
                    examples.append (tmp_examples[0:180])
                    labels.append (tem_labels[259])
                    tmp_examples = tmp_examples[90:360]
                    tem_labels = tem_labels[90:360]
                # print (row[0])
                tmp_examples.append(row[1:22])
                tem_labels.append(row[23])
                cnt = cnt + 1
        self.sentences = np.array(examples)
        #print (np.shape (self.sentences))
        self.labels = np.array(labels)
        #print (np.shape (self.labels))
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]
        #print (len(self.labels))
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)
        self.pointer = 0


    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret


    def reset_pointer(self):
        self.pointer = 0

    def set_folder(self, folder):
        self.folder = folder

