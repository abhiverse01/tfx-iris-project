"""Script to run the TFX pipeline."""

# tfx orchestration manages tfx containere by using the DAG - Directed Acyclic Graph

from tfx.orchestration.local import LocalDagRunner
from pipeline.pipeline import create_pipeline

if __name__ == '__main__':
    LocalDagRunner().run(create_pipeline())
