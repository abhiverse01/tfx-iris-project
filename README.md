# TFX Iris Pipeline Project

This project demonstrates how to create a TensorFlow Extended (TFX) pipeline using the Iris flower dataset. The pipeline includes data ingestion, validation, transformation, training, evaluation, and model serving.

## Project Structure

```bash
tfx_iris_project/
│
├── data/
│ ├── train/
│ │ └── train.csv
│ └── eval/
│ └── eval.csv
│
├── models/
│ ├── init.py
│ └── model.py
│
├── notebooks/
│ └── tfx_iris_pipeline.ipynb
│
├── pipeline/
│ ├── init.py
│ └── pipeline.py
│
├── scripts/
│ └── run_pipeline.py
│
└── README.md
```

## Instructions

1. **Set Up Environment**: Install the necessary packages.
    ```bash
    pip install tfx tensorflow tensorflow-data-validation tensorflow-transform tensorflow-model-analysis tensorflow-serving
    ```

2. **Prepare the Dataset**: Add your training and evaluation data to the `data/train/train.csv` and `data/eval/eval.csv` files, respectively.

3. **Run the Pipeline**:
    - You can run the pipeline using the script in the `scripts` directory.
    ```bash
    python scripts/run_pipeline.py
    ```

4. **Explore the Notebook**: Open the Jupyter notebook in the `notebooks` directory to explore the pipeline in an interactive environment.
    ```bash
    jupyter notebook notebooks/tfx_iris_pipeline.ipynb
    ```

## Pipeline Components

- **ExampleGen**: Ingests data and splits it into training and evaluation sets.
- **StatisticsGen**: Generates statistics for data validation.
- **SchemaGen**: Infers the schema of the dataset.
- **ExampleValidator**: Validates the data against the schema.
- **Transform**: Preprocesses the data using TensorFlow Transform.
- **Trainer**: Trains a TensorFlow model.
- **Evaluator**: Evaluates the model's performance.
- **Pusher**: Deploys the trained model.

## Model

The model is defined in `models/model.py` and consists of a simple neural network with one hidden layer.

```python
import tensorflow as tf

def _build_keras_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(4,)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
```
