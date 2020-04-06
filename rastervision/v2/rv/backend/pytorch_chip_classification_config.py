from rastervision.v2.rv.backend import BackendConfig
from rastervision.v2.learner.classification_config import (
    ClassificationLearnerConfig)
from rastervision.v2.core.config import register_config
from rastervision.v2.rv.backend.pytorch_chip_classification import (
    PyTorchChipClassification)


@register_config('pytorch_chip_classification')
class PyTorchChipClassificationConfig(BackendConfig):
    learner: ClassificationLearnerConfig

    def update(self, task=None):
        super().update(task=task)

        if task is not None:
            self.learner.data.img_sz = task.train_chip_sz
            self.learner.test_mode = task.debug
            self.learner.data.class_colors = task.dataset.class_config.colors
            self.learner.data.class_names = task.dataset.class_config.names
            self.learner.output_uri = task.train_uri
            self.learner.data.uri = task.chip_uri

        self.learner.update()

    def build(self, task, tmp_dir):
        return PyTorchChipClassification(task, self.learner, tmp_dir)
