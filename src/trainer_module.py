import tensorflow as tf
from tensorflow.keras import layers

def _build_keras_model() -> tf.keras.Model:
    model = tf.keras.Sequential([
        layers.Dense(10, activation='relu', input_shape=(4,)),
        layers.Dense(3, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def run_fn(fn_args):
    model = _build_keras_model()
    
    train_dataset = tf.data.experimental.make_batched_features_dataset(
        fn_args.train_files,
        batch_size=32,
        features=fn_args.schema,
        label_key='species'
    )
    eval_dataset = tf.data.experimental.make_batched_features_dataset(
        fn_args.eval_files,
        batch_size=32,
        features=fn_args.schema,
        label_key='species'
    )

    model.fit(
        train_dataset,
        validation_data=eval_dataset,
        epochs=5
    )
    
    model.save(fn_args.serving_model_dir)
