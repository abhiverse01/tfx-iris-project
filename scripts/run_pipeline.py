from pipeline.pipeline import create_pipeline

if __name__ == '__main__':
    from tfx.orchestration.local import local_dag_runner
    local_dag_runner.LocalDagRunner().run(create_pipeline())
