from typing import Optional, List
from os.path import join

from rastervision2.core.data.label_store import (
    LabelStoreConfig, SemanticSegmentationLabelStore)
from rastervision2.pipeline.config import register_config, Config, Field


@register_config('vector_output')
class VectorOutputConfig(Config):
    uri: Optional[str] = Field(None, description='URI of vector output')
    class_id: int = Field(
        ..., description='The prediction class that is to turned into vectors.')
    denoise: int = Field(0, description=(
        'Radius of the structural element used to remove high-frequency signals from '
        'the image.'))

    def get_mode(self):
        raise NotImplementedError()


@register_config('polygon_vector_output')
class PolygonVectorOutputConfig(VectorOutputConfig):
    def get_mode(self):
        return 'polygons'


@register_config('building_vector_output')
class BuildingVectorOutputConfig(VectorOutputConfig):
    """Options useful for vectorization of building predictions.

    Intended to break up clusters of buildings.
    """
    min_aspect_ratio: float = Field(1.618, description=(
        'Ratio between length and height (or height and length) of anything that can '
        'be considered to be a cluster of buildings. The goal is to distinguish between '
        'rows of buildings and (say) a single building.'))
    min_area: float = Field(0.0, description=(
        'Minimum area of anything that can be considered to be a cluster of buildings. '
        'The goal is to distinguish between buildings and artifacts.'))
    element_width_factor: float = Field(0.5, description=(
        'Width of the structural element used to break building clusters as a fraction '
        'of the width of the cluster.'))
    element_thickness: float = Field(0.001, description=(
        'Thickness of the structural element that is used to break building clusters.'))

    def get_mode(self):
        return 'buildings'


@register_config('semantic_segmentation_label_store')
class SemanticSegmentationLabelStoreConfig(LabelStoreConfig):
    uri: Optional[str] = Field(None, description='URI of predictions.')
    vector_output: List[VectorOutputConfig] = []
    rgb: bool = Field(False, description=(
        'If True, save prediction class_ids in RGB format using the colors in '
        'class_config.'))

    def build(self, class_config, crs_transformer, extent, tmp_dir):
        return SemanticSegmentationLabelStore(
            self.uri,
            extent,
            crs_transformer,
            tmp_dir,
            vector_output=self.vector_output,
            class_config=class_config)

    def update(self, pipeline=None, scene=None):
        if pipeline is not None and scene is not None:
            if self.uri is None:
                self.uri = join(pipeline.predict_uri,
                                '{}.tif'.format(scene.id))

            for vo in self.vector_output:
                vo.uri = join(
                    pipeline.root_uri, 'predict', '{}-{}-{}.json'.format(
                        scene.id, vo.class_id, vo.get_mode()))
