"""Script to run the TFX pipeline."""

from tfx.orchestration.local import LocalDagRunner
from pipeline.pipeline import create_pipeline

if __name__ == '__main__':
    LocalDagRunner().run(create_pipeline())
