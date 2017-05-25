import tensorflow as tf
import pandas as pd

filename_queue = tf.train.string_input_producer(["../data/train_2016.csv", "../data/train_2016/properties_2016.csv"])
reader = tf.TextLineReader()
label, features = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[1], [1], [1]]
col1, col2, col3 = tf.decode_csv(label, record_defaults=record_defaults)
features = tf.stack([col1, col3])

with tf.Session() as sess:
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(1200):
        # Retrieve a single instance:
        example, label = sess.run([features, col2])

    coord.request_stop()
    coord.join(threads)
