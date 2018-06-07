import tensorflow as tf
from argparse import ArgumentParser
import os
import shutil

def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)

#os.environ['CUDA_VISIBLE_DEVICES']='2'  #GPU



def main():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        dest='checkpoint',
                        help='dir or .ckpt file to load checkpoint from',
                        metavar='CHECKPOINT', required=True)
    parser.add_argument('--model', type=str,
                        dest='model',
                        help='.meta for your model',
                        metavar='MODEL', required=True)
    parser.add_argument('--out-path', type=str,
                        dest='out_path',
                        help='model output directory',
                        metavar='MODEL_OUT', required=True)
    opts = parser.parse_args()

    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(opts.model)

    if os.path.exists(opts.out_path):
        path = opts.out_path
        del_file(path)
        shutil.rmtree(path)

    builder = tf.saved_model.builder.SavedModelBuilder(opts.out_path)

    with tf.Session(graph=tf.Graph()) as sess:
        # Restore variables from disk.
        saver.restore(sess, opts.checkpoint)
        print("Model restored.")
        builder.add_meta_graph_and_variables(sess,
                                       ['tfckpt2pb'],
                                       signature_def_map = foo_signatures,
                                       assets_collection = foo_assets
                                       strip_default_attrs=False)


if __name__ == '__main__':
    main()
