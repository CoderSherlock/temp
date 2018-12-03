'''
Distributed Tensorflow 1.2.0 example of using data parallelism and share model parameters.
Trains a simple sigmoid neural network on mnist for 20 epochs on three machines using one parameter server. 

Change the hardcoded host urls below with your own hosts. 
Run like this: 

pc-01$ python example.py --job_name="ps" --task_index=0 
pc-02$ python example.py --job_name="worker" --task_index=0 
pc-03$ python example.py --job_name="worker" --task_index=1 
pc-04$ python example.py --job_name="worker" --task_index=2 

More details here: ischlag.github.io
'''

from __future__ import print_function
from model import LeNet
import model

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
import numpy as np
#  from sklearn.utils import shuffle
import sys
import time

# cluster specification
parameter_servers = ["127.0.0.1:2000"]
#  workers = [	        "127.0.0.1:3000"]
        #  "127.0.0.1:3001"]
workers = [	        "127.0.0.1:3000",
                        "127.0.0.1:3001",
                        "127.0.0.1:3002",
                        "127.0.0.1:3003"]
#  workers = [	        "127.0.0.1:3000",
#                          "127.0.0.1:3001",
#                          "127.0.0.1:3002",
#                          "127.0.0.1:3003",
#                          "127.0.0.1:3004",
#                          "127.0.0.1:3005",
#                          "127.0.0.1:3006",
#                          "127.0.0.1:3007"]
cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

# start a server for a specific task
server = tf.train.Server(
        cluster,
        job_name=FLAGS.job_name,
        task_index=FLAGS.task_index)

# config
batch_size = 128
learning_rate = 0.01
training_epochs = 30
logs_path = "/tmp/mnist/1"

# load mnist data set
from tensorflow.examples.tutorials.mnist import input_data

def pre_data():
    mnist = input_data.read_data_sets("MNIST_data", reshape=False)
    X_train, y_train           = mnist.train.images, mnist.train.labels
    X_validation, y_validation = mnist.validation.images, mnist.validation.labels
    X_test, y_test             = mnist.test.images, mnist.test.labels

    assert(len(X_train) == len(y_train))
    assert(len(X_validation) == len(y_validation))
    assert(len(X_test) == len(y_test))

    print("Image Shape: {}".format(X_train[0].shape))
    print("Training Set:   {} samples".format(len(X_train)))
    print("Validation Set: {} samples".format(len(X_validation)))
    print("Test Set:       {} samples".format(len(X_test)))

    # Pad images with 0s
    X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')

    return X_train,y_train,X_validation,y_validation,X_test,y_test

X_train,y_train,X_validation,y_validation,X_test,y_test = pre_data()
X_train = X_train[:1000]
y_train = y_train[:1000]
#  X_train, y_train = shuffle(X_train, y_train)

if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":

        X_train = X_train[int(FLAGS.task_index)*250:250*(int(FLAGS.task_index)+1)]
        y_train = y_train[int(FLAGS.task_index)*250:250*(int(FLAGS.task_index)+1)]
        print(len(X_train), len(y_train))
        # Between-graph replication
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):

            # count the number of updates
                global_step = tf.get_variable(
                        'global_step',
                        [],
                        initializer = tf.constant_initializer(0),
                        trainable = False)

                with tf.name_scope('input'):
                    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
                    y = tf.placeholder(tf.int32, (None))
                    one_hot_y = tf.one_hot(y, 10);

                    logits = LeNet(x);

                with tf.name_scope('cross_entropy'):
                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_y)

                loss_operation = tf.reduce_mean(cross_entropy)

                with tf.name_scope('train'):
                    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                    training_operation = optimizer.minimize(loss_operation)

                with tf.name_scope('Accuracy'):
                    # accuracy
                        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
                        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                # create a summary for our cost and accuracy
                #  tf.summary.scalar("cost", cross_entropy)
                #  tf.summary.scalar("accuracy", accuracy)

                # merge all summaries into a single "operation" which we can execute in a session 
                #  summary_op = tf.summary.merge_all()
                init_op = tf.global_variables_initializer()
                print("Variables initialized ...")

        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0), global_step=global_step, init_op=init_op)

        begin_time = time.time()
        frequency = 100
        with sv.prepare_or_wait_for_session(server.target) as sess:
            '''
            # is chief
            if FLAGS.task_index == 0:
                    sv.start_queue_runners(sess, [chief_queue_runner])
                    sess.run(init_token_op)
            '''
            # create log writer object (this will log on every machine)
            writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

            # perform training cycles
            start_time = time.time()
            for epoch in range(training_epochs):

                    # number of batches in one epoch
                    num_examples = len(X_train)
                    batch_count = int(num_examples/batch_size)

                    count = 0
                    for offset in range(0, num_examples, batch_size):
                        end = offset + batch_size
                        batch_x, batch_y = X_train[offset:end], y_train[offset:end]

                            # perform the operations we defined earlier on batch
                        _, cost, step = sess.run(
                                        [training_operation, cross_entropy, global_step], 
                                        feed_dict={x: batch_x, y: batch_y})

                        count += 1
                        if count % frequency == 0 or count+1 == batch_count:
                                elapsed_time = time.time() - start_time
                                start_time = time.time()
                                print("Test-Accuracy: %2.2f" % sess.run(accuracy, feed_dict={x: X_test, y: y_test}))
                                print("Step: %d," % (step+1), 
                                                        " Epoch: %2d," % (epoch+1), 
                                                        " Batch: %3d of %3d," % (count+1, batch_count), 
                                                        #  " Cost: %.4f," % cost,
                                                        " AvgTime: %3.2fms" % float(elapsed_time*1000/frequency))
                    count = 0


            print("Test-Accuracy: %2.2f" % sess.run(accuracy, feed_dict={x: X_test, y: y_test}))
            print("Total Time: %3.2fs" % float(time.time() - begin_time))
            #  print("Final Cost: %.4f" % cost)

        sv.stop()
        print("done")
