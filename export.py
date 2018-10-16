import os
import tensorflow as tf
import pickle

from data import UNK

# TODO: Handle with args

# Load mapping from words (str) to IDs (int)
word_to_id = pickle.load(open('embeddings/glove.6B.300d.wti', 'rb'))
id_to_word = pickle.load(open('embeddings/glove.6B.300d.itw', 'rb'))
mapping = tf.constant([id_to_word[id] for id in range(len(id_to_word))])
table = tf.contrib.lookup.index_table_from_tensor(mapping,
                                                default_value=word_to_id[UNK])

# Read and split words
input_words = tf.placeholder(tf.string, name='input_words')
split_words = tf.string_split(input_words).values
# Convert input strings to word IDs in mini-batch of size 1
word_ids = tf.expand_dims(table.lookup(split_words), axis=0)

checkpoint = tf.train.latest_checkpoint('./runs/2018-10-16-201207/')
# Load graph mapping input tensor
saver = tf.train.import_meta_graph(checkpoint + '.meta', input_map={'data/words:0': word_ids})
prediction = tf.get_default_graph().get_tensor_by_name('accuracy/prediction:0')

with tf.Session() as sess:
    tf.tables_initializer().run()
    saver.restore(sess, checkpoint)

    # Export signatures
    words_tensor_info = tf.saved_model.utils.build_tensor_info(input_words)
    preds_tensor_info = tf.saved_model.utils.build_tensor_info(prediction)
    split_tensor_info = tf.saved_model.utils.build_tensor_info(split_words)
    ids_tensor_info = tf.saved_model.utils.build_tensor_info(word_ids)
    prediction_signature = (
       tf.saved_model.signature_def_utils.build_signature_def(
           inputs={'input_words': words_tensor_info},
           outputs={'prediction': preds_tensor_info},
           method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    # Export model for serving
    export_dir = os.path.join('servables', '1')
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    builder.add_meta_graph_and_variables(
       sess,
       [tf.saved_model.tag_constants.SERVING],
       signature_def_map={
           tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
       },
       main_op=tf.tables_initializer()
    )
    builder.save()
