import tensorflow as tf
from tensorflow.contrib.layers import flatten

def init_weight(shape):
    w = tf.truncated_normal(shape=shape, mean = 0, stddev = 0.1)
    return tf.Variable(w)
    
def init_bias(shape):
    b = tf.zeros(shape)
    return tf.Variable(b)

def LeNet(x):
    # name:      conv5-6    
    # structure: Input = 32x32x1. Output = 28x28x6.
    # weights:   (5*5*1+1)*6
    # connections: (28*28*5*5+28*28)*6
    conv1_W = init_weight((5,5,1,6))
    conv1_b = init_bias(6)
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)
    #Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    #conv5-16
    #input 14x14x6 Output = 10x10x16.
    #weights: (5*5*6+1)*16 ---real Lenet-5 is (5*5*3+1)*6+(5*5*4+1)*9+(5*5*6+1)*1
    conv2_W = init_weight((5, 5, 6, 16))
    conv2_b = init_bias(16)
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)
    #Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    #Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    #Input = 400. Output = 120.
    fc1_W = init_weight((400,120))
    fc1_b = init_bias(120)
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    fc1    = tf.nn.relu(fc1)

    #Input = 120. Output = 84.
    fc2_W  = init_weight((120,84))
    fc2_b  = init_bias(84)
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    fc2 = tf.nn.relu(fc2)

    #Input = 84. Output = 10.
    fc3_W  = init_weight((84,10))
    fc3_b  = init_bias(10)
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits
