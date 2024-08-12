# src/pipeline.py
from tfx.components import CsvExampleGen, Trainer, Evaluator, Pusher
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from tfx.proto import trainer_pb2


# Create an interactive context to run the pipeline
context = InteractiveContext()


# Data ingestion: Read the CSV file and convert it to TFX examples
example_gen = CsvExampleGen(input_base='data/')
context.run(example_gen)

# Trainer: Define and train the model
trainer = Trainer(
    module_file='src/trainer_module.py',
    examples=example_gen.outputs['examples'],
    train_args=trainer_pb2.TrainArgs(num_steps=100),
    eval_args=trainer_pb2.EvalArgs(num_steps=50)
)
context.run(trainer)

# Evaluator: Evaluate the trained model
evaluator = Evaluator(
    examples=example_gen.outputs['examples'],
    model_exports=trainer.outputs['model']
)
context.run(evaluator)


# Pusher: Deploy the model if it's blessed by the Evaluator
pusher = Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    push_destination='serving_path'
)
context.run(pusher)
