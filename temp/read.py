import tensorflow as tf

# TFRecord 파일명
input_filename = 'language_model_data.tfrecord'

# TFRecord 파일 읽기 함수
def parse_tf_example(serialized_example):
    feature_description = {
        'text': tf.io.FixedLenFeature([], tf.string),
        # 추가적인 feature가 있다면 여기에 추가
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    return example['text']

# TFRecord 파일 읽기
dataset = tf.data.TFRecordDataset(input_filename)
# TFRecord 파일의 각 example을 파싱하여 데이터셋에 적용
parsed_dataset = dataset.map(parse_tf_example)

# 데이터셋에서 데이터 읽기
for example in parsed_dataset:
    print(example.numpy().decode('utf-8'))
