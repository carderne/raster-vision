from rastervision.new_version.pipeline.pipeline_config import PipelineConfig

BASE_PIPELINE = 'base_pipeline'


class Pipeline():
    config_class = PipelineConfig
    commands = ['test_cpu', 'test_gpu']
    split_commands = ['test_cpu']
    gpu_commands = ['test_gpu']

    def __init__(self, config, tmp_dir):
        self.config = config
        self.tmp_dir = tmp_dir

    def test_cpu(self, split_ind=0, num_splits=1):
        print(self.config)

    @staticmethod
    def test_gpu(self):
        print(self.config)
