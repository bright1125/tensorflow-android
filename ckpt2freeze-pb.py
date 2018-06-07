import tensorflow as tf
from argparse import ArgumentParser
import os
import shutil
from tensorflow.python.tools import freeze_graph
import tensorflow.contrib.slim as slim
import numpy as np


def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)

#os.environ['CUDA_VISIBLE_DEVICES']='2'  #GPU
#tf.app.flags.DEFINE_integer('charset_size', 3755, "Choose the first `charset_size` character to conduct our experiment.")


#def build_graph(top_k):
    # with tf.device('/cpu:0'):

 #   return {'images': images,
  #          'labels': labels,
    #       'keep_prob': keep_prob,
   #         'top_k': top_k,
    #        'global_step': global_step,
    #        'train_op': train_op,
    #        'loss': loss,
      #      'accuracy': accuracy,
    #        'accuracy_top_k': accuracy_in_top_k,
     #       'merged_summary_op': merged_summary_op,
   #         'predicted_distribution': probabilities,
    #        'predicted_index_top_k': predicted_index_top_k,
     #       'predicted_val_top_k': predicted_val_top_k}


def main():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        dest='checkpoint',
                        help='dir or .ckpt file to load checkpoint from',
                        metavar='CHECKPOINT', required=True)
    parser.add_argument('--out-path', type=str,
                        dest='out_path',
                        help='model output directory',
                        metavar='MODEL_OUT', required=True)
    opts = parser.parse_args()

    if not os.path.exists(opts.out_path):
        os.mkdir(opts.out_path) 

    tf.reset_default_graph()
###############################################################################
 #   graph = build_graph(top_k=3)
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
    images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1], name='input')
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')

    conv_1 = slim.conv2d(images, 64, [3, 3], 1, padding='SAME', scope='conv1')
    max_pool_1 = slim.max_pool2d(conv_1, [2, 2], [2, 2], padding='SAME')
    conv_2 = slim.conv2d(max_pool_1, 128, [3, 3], padding='SAME', scope='conv2')
    max_pool_2 = slim.max_pool2d(conv_2, [2, 2], [2, 2], padding='SAME')
    conv_3 = slim.conv2d(max_pool_2, 256, [3, 3], padding='SAME', scope='conv3')
    max_pool_3 = slim.max_pool2d(conv_3, [2, 2], [2, 2], padding='SAME')

    flatten = slim.flatten(max_pool_3)
    fc1 = slim.fully_connected(slim.dropout(flatten, keep_prob), 1024, activation_fn=tf.nn.tanh, scope='fc1')
    logits = slim.fully_connected(slim.dropout(fc1, keep_prob), 3755, activation_fn=None, scope='fc2')
#    flow = tf.cast(logits, tf.uint8, 'output')
        # logits = slim.fully_connected(flatten, FLAGS.charset_size, activation_fn=None, reuse=reuse, scope='fc')
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))

    global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
    rate = tf.train.exponential_decay(2e-4, global_step, decay_steps=2000, decay_rate=0.97, staircase=True)
    train_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss, global_step=global_step)
    probabilities = tf.nn.softmax(logits)

    predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=3)
    predicted_val_top_k = tf.cast(predicted_val_top_k, tf.float32, 'predicted_val_top_k')
    predicted_index_top_k = tf.cast(predicted_index_top_k, tf.float32, 'predicted_index_top_k')
    output = tf.concat([predicted_val_top_k, predicted_index_top_k],-1,name='output')

  #  output = tf.
    accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, 3), tf.float32))


#####################################################################################
    saver = tf.train.Saver()
    with tf.Session() as sess:

        saver.restore(sess, opts.checkpoint)
        predict_val, predict_index = sess.run([predicted_val_top_k, predicted_index_top_k],
                                              feed_dict={images: np.zeros([1,64,64,1]), keep_prob: 1.0})
        #save graph
        tf.train.write_graph(sess.graph_def, opts.out_path, 'model.pb')
        #put graph and parameters together
        freeze_graph.freeze_graph(opts.out_path+'/model.pb', '', False, opts.checkpoint, 'output','save/restore_all', 'save/Const:0', opts.out_path+'/frozen_model.pb', False, "")

    print("done")

if __name__ == '__main__':
    main()

