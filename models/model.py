"""Model definition and training script."""

import tensorflow as tf
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow_transform import TFTransformOutput

def _build_keras_model():
    """Builds and compiles a Keras model."""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(4,)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def run_fn(fn_args):
    """Train and save the model."""
    tf_transform_output = TFTransformOutput(fn_args.transform_output)
    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, batch_size=40)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, batch_size=40)
    
    model = _build_keras_model()
    model.fit(train_dataset, steps_per_epoch=fn_args.train_steps, validation_data=eval_dataset, validation_steps=fn_args.eval_steps)
    model.save(fn_args.serving_model_dir, save_format='tf')

def _input_fn(file_pattern, tf_transform_output, batch_size=32):
    """Generate dataset for training and evaluation."""
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=tf_transform_output.transformed_feature_spec(),
        reader=tf.data.TFRecordDataset
    )
    return dataset
