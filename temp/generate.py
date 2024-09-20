import tensorflow as tf

data = ["This is an example sentence.",
        "Another example sentence.",
        "Yet another example sentence."]

output_filename = 'language_model_data.tfrecord'

def create_tf_example(text):
    feature = {
        'text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[text.encode('utf-8')]))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

with tf.io.TFRecordWriter(output_filename) as writer:
   for sentence in data:
        tf_example = create_tf_example(sentence)
        writer.write(tf_example.SerializeToString())


