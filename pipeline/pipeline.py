import os
from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Trainer, Evaluator, Pusher
from tfx.orchestration import pipeline
from tfx.orchestration.local import local_dag_runner
from tfx.orchestration.metadata import sqlite_metadata_connection_config
from tfx.proto import example_gen_pb2, trainer_pb2, eval_config_pb2, pusher_pb2

def create_pipeline():
    data_root = os.path.join(os.getcwd(), 'data')

    input_config = example_gen_pb2.Input(splits=[
        example_gen_pb2.Input.Split(name='train', pattern='train/*'),
        example_gen_pb2.Input.Split(name='eval', pattern='eval/*')
    ])

    example_gen = CsvExampleGen(input_base=data_root, input_config=input_config)
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )

    def preprocessing_fn(inputs):
        import tensorflow_transform as tft
        outputs = {}
        outputs['sepal_length'] = tft.scale_to_z_score(inputs['sepal_length'])
        outputs['sepal_width'] = tft.scale_to_z_score(inputs['sepal_width'])
        outputs['petal_length'] = tft.scale_to_z_score(inputs['petal_length'])
        outputs['petal_width'] = tft.scale_to_z_score(inputs['petal_width'])
        outputs['species'] = inputs['species']
        return outputs

    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        preprocessing_fn=preprocessing_fn
    )

    trainer = Trainer(
        module_file=os.path.join(os.getcwd(), 'models', 'model.py'),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(num_steps=1000),
        eval_args=trainer_pb2.EvalArgs(num_steps=100)
    )

    eval_config = eval_config_pb2.EvalConfig(
        slicing_specs=[eval_config_pb2.SlicingSpec()],
        metrics_specs=[eval_config_pb2.MetricsSpec(metrics=[
            eval_config_pb2.MetricConfig(class_name='ExampleCount'),
            eval_config_pb2.MetricConfig(class_name='Accuracy')
        ])]
    )

    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        eval_config=eval_config
    )

    pusher = Pusher(
        model=trainer.outputs['model'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(base_directory='serving_model_dir')
        )
    )

    return pipeline.Pipeline(
        pipeline_name='iris_pipeline',
        pipeline_root='pipeline_output',
        components=[
            example_gen,
            statistics_gen,
            schema_gen,
            example_validator,
            transform,
            trainer,
            evaluator,
            pusher
        ],
        enable_cache=True,
        metadata_connection_config=sqlite_metadata_connection_config('metadata.db')
    )

if __name__ == '__main__':
    local_dag_runner.LocalDagRunner().run(create_pipeline())
