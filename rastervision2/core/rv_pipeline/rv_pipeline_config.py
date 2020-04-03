from os.path import join
from typing import List, TYPE_CHECKING, Optional, Union

from rastervision2.pipeline.pipeline_config import PipelineConfig
from rastervision2.core.data import (DatasetConfig, StatsTransformerConfig,
                                     LabelStoreConfig, SceneConfig)
from rastervision2.core.analyzer import StatsAnalyzerConfig
from rastervision2.core.backend import BackendConfig
from rastervision2.core.evaluation import EvaluatorConfig
from rastervision2.core.analyzer import AnalyzerConfig
from rastervision2.pipeline.config import register_config, Field

if TYPE_CHECKING:
    from rastervision2.core.backend.backend import Backend  # noqa


@register_config('rv_pipeline')
class RVPipelineConfig(PipelineConfig):
    """Config for RVPipeline."""
    dataset: DatasetConfig = Field(
        ...,
        description='dataset containing train, validation, and optional test scenes')
    backend: BackendConfig = Field(
        ..., description='backend to use for interfacing with ML library')
    evaluators: List[EvaluatorConfig] = Field(
        [], description='evaluators to run during analyzer command')
    analyzers: List[AnalyzerConfig] = Field(
        [], description='analyzers to run during analyzer command')

    debug: bool = Field(False, description='if True, use debug mode')
    train_chip_sz: int = Field(200, description='size of training chips in pixels')
    predict_chip_sz: int = Field(800, description='size of predictions chips in pixels')
    predict_batch_sz: int = Field(8, description='batch size to use during prediction')

    analyze_uri: Optional[str] = Field(None, description='URI for output of analyze')
    chip_uri: Optional[str] = Field(None, description='URI for output of chip')
    train_uri: Optional[str] = Field(None, description='URI for output of train')
    predict_uri: Optional[str] = Field(None, description='URI for output of predict')
    eval_uri: Optional[str] = Field(None, description='URI for output of eval')
    bundle_uri: Optional[str] = Field(None, description='URI for output of bundle')

    from rastervision2.core.data import ClassConfig
    class_config: Union[ClassConfig, str] = ClassConfig(names=['a'], colors=['a'])

    def update(self):
        super().update()

        if self.analyze_uri is None:
            self.analyze_uri = join(self.root_uri, 'analyze')
        if self.chip_uri is None:
            self.chip_uri = join(self.root_uri, 'chip')
        if self.train_uri is None:
            self.train_uri = join(self.root_uri, 'train')
        if self.predict_uri is None:
            self.predict_uri = join(self.root_uri, 'predict')
        if self.eval_uri is None:
            self.eval_uri = join(self.root_uri, 'eval')
        if self.bundle_uri is None:
            self.bundle_uri = join(self.root_uri, 'bundle')

        self.dataset.update(pipeline=self)
        self.backend.update(pipeline=self)
        if not self.evaluators:
            self.evaluators.append(self.get_default_evaluator())
        for evaluator in self.evaluators:
            evaluator.update(pipeline=self)

        self._insert_analyzers()
        for analyzer in self.analyzers:
            analyzer.update(pipeline=self)

    def _insert_analyzers(self):
        # Inserts StatsAnalyzer if it's needed because a RasterSource has a
        # StatsTransformer, but there isn't a StatsAnalyzer in the list of Analyzers.
        has_stats_transformer = False
        for s in self.dataset.get_all_scenes():
            for t in s.raster_source.transformers:
                if isinstance(t, StatsTransformerConfig):
                    has_stats_transformer = True

        has_stats_analyzer = False
        for a in self.analyzers:
            if isinstance(a, StatsAnalyzerConfig):
                has_stats_analyzer = True
                break

        if has_stats_transformer and not has_stats_analyzer:
            self.analyzers.append(StatsAnalyzerConfig())

    def get_default_label_store(self, scene: SceneConfig) -> LabelStoreConfig:
        """Returns a default LabelStoreConfig to fill in any missing ones."""
        raise NotImplementedError()

    def get_default_evaluator(self) -> EvaluatorConfig:
        """Returns a default EvaluatorConfig to use if one isn't set."""
        raise NotImplementedError()
