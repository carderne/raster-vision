from os.path import join

from rastervision.v2.learner.classification_config import (
    ClassificationLearnerConfig, ClassificationDataConfig)
from rastervision.v2.learner.learner_config import (LearnerPipelineConfig,
                                                    SolverConfig, ModelConfig)


def get_config(runner, test=False):
    base_uri = ('s3://raster-vision-lf-dev/learner/classification' if
                runner == 'aws_batch' else '/opt/data/learner/classification')
    root_uri = join(base_uri, 'output')
    data_uri = join(base_uri, 'tiny-buildings.zip')

    model = ModelConfig(backbone='resnet50')
    solver = SolverConfig(lr=2e-4, num_epochs=3, batch_sz=8, one_cycle=True)
    data = ClassificationDataConfig(
        data_format='image_folder',
        uri=data_uri,
        img_sz=200,
        labels=['building', 'no_building'])
    learner = ClassificationLearnerConfig(
        model=model, solver=solver, data=data, test_mode=test)
    pipeline = LearnerPipelineConfig(root_uri=root_uri, learner=learner)
    return pipeline
