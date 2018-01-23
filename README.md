import sys
import numpy as np
import tensorflow as tf
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

device_name = sys.argv[1]  # Choose device from cmd line. Options: gpu or cpu
shape = (int(sys.argv[2]), int(sys.argv[2]))
if device_name == "gpu":
    device_name = "/gpu:0"
else:
    device_name = "/cpu:0"

with tf.device(device_name):
    random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
    dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
    sum_operation = tf.reduce_sum(dot_operation)


startTime = time.time()
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
        result = session.run(sum_operation)
        print(result)
        
endTime = time.time()
train_time = endTime - startTime
# It can be hard to see the results on the terminal with lots of output -- add some newlines to improve readability.
print("\n" * 2)
print("Shape:", shape, "Device:", device_name)
print("Time taken:", train_time)

print("\n" * 2)
