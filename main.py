# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import dataloader
#import matplotlib.pyplot as plt

class Rnn:
    def __init__(self,num_classes, learning_rate, batch_size, decay_steps, decay_rate,time_length,
                 vocab_size,data_size,is_training,initializer=tf.random_normal_initializer(stddev=0.1)):
        # set hyperparamter
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.time_length=time_length
        self.vocab_size=vocab_size
        self.data_size=data_size
        self.hidden_size=25
        self.is_training=is_training
        self.learning_rate=learning_rate
        self.initializer=initializer
        self.num_sampled=20

        self.input_x = tf.placeholder(tf.float32, [None, self.time_length, self.data_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32,[None], name="input_y")
        self.dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step=tf.Variable(0,trainable=False,name="Epoch_Step")
        self.epoch_increment=tf.assign(self.epoch_step,tf.add(self.epoch_step,tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.logits = self.inference()
        if not is_training:
            return
        self.loss_val = self.loss()
        self.train_op = self.train()
    def instantiate_weights(self):
        with tf.name_scope("result_matrix"):

            self.W_projection = tf.Variable(tf.random_normal([self.hidden_size, 1]), name='W')
            self.b_projection = tf.Variable(tf.random_normal([1]), name='b')

    def inference(self):
        lstm_fw_cell=rnn.BasicLSTMCell(self.hidden_size)

        if self.dropout_keep_prob is not None:
            lstm_fw_cell=rnn.DropoutWrapper(lstm_fw_cell,output_keep_prob=self.dropout_keep_prob)

        self.outputs,output_state=tf.nn.dynamic_rnn(lstm_fw_cell,self.input_x,dtype=tf.float32)
        print("output", self.outputs)
        self.output_rnn_last = self.outputs[:, -1, :]
        with tf.name_scope("output"):
            logits = tf.matmul(self.output_rnn_last, self.W_projection) + self.b_projection
        return logits

    def loss(self,l2_lambda=0.0001):
        with tf.name_scope("loss"):
            self.losses = tf.square(self.input_y - self.logits)
            total_loss = tf.reduce_mean(self.losses)
            print("logits.losses:",total_loss) # shape=(?,)
            #loss=tf.reduce_mean(losses)#print("2.loss.loss:", loss) #shape=()
            #l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss=total_loss #+l2_losses
        return loss

    def train(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam")
        return train_op

train_file = "train_file_valid.csv"
test_file = "test_file_valid.csv"

def test():
    num_classes=2
    learning_rate=0.25
    batch_size=32
    decay_steps=1000
    decay_rate=0.9
    time_length=180  #148
    vocab_size=10000
    data_size=21
    is_training=True
    dropout_keep_prob=0.5#0.5
    epoch = 200

    rnn=Rnn(num_classes, learning_rate, batch_size, decay_steps, decay_rate,time_length,vocab_size,data_size,is_training)

    data_loader_train = dataloader.Dis_dataloader(batch_size)
    cnt = 0
    loss_total = 0
    result = []
    whole_loss = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        data_loader_train.load_train_data(train_file)
        print (data_loader_train.num_batch)

        for f in range (epoch):

            data_loader_train.set_folder (f)
            data_loader_train.reset_pointer()
            cnt = 0
            loss_total = 0
            for it in range(data_loader_train.num_batch):
                input_x, input_y = data_loader_train.next_batch()
                #print (np.shape(input_x))
                #print (np.shape(input_y))
                outputs, loss,losses, _=sess.run([rnn.output_rnn_last,rnn.loss_val,rnn.losses, rnn.train_op],feed_dict={rnn.input_x:input_x,rnn.input_y:input_y,rnn.dropout_keep_prob:dropout_keep_prob})
                print (loss)
                #print (losses)
                loss_total = loss_total + loss
                result.append(loss)
                cnt = cnt + 1
                # break

            #print(loss)
            #break
            print(loss_total / cnt)
            whole_loss = whole_loss + loss_total / cnt
        print (whole_loss / epoch)


    #print (loss_total/cnt)
    '''
    figure = lambda: plt.figure(figsize=(16, 5))
    figure()
    plt.axis('auto')
    plt.plot(result, marker='.', label='lossuracy')
    plt.legend(loc='best')
    plt.show()
    '''

test()
