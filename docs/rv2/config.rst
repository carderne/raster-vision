.. _rv2_config:

Configuration Reference
========================

rastervision2.pipeline
------------------------

.. autoclass:: rastervision2.pipeline.pipeline_config.PipelineConfig

rastervision2.core
-------------------

rastervision2.core.analyzer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.core.analyzer.StatsAnalyzerConfig

rastervision2.core.data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.core.data.label_source.ChipClassificationLabelSourceConfig

.. autoclass:: rastervision2.core.data.label_source.SemanticSegmentationLabelSourceConfig

.. autoclass:: rastervision2.core.data.label_store.ChipClassificationGeoJSONStoreConfig

.. autoclass:: rastervision2.core.data.label_store.PolygonVectorOutputConfig

.. autoclass:: rastervision2.core.data.label_store.BuildingVectorOutputConfig

.. autoclass:: rastervision2.core.data.label_store.SemanticSegmentationLabelStoreConfig

rastervision2.core.rv_pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rastervision2.core.rv_pipeline.RVPipelineConfig

.. autoclass:: rastervision2.core.rv_pipeline.SemanticSegmentationConfig

rastervision2.pytorch_backend
-------------------------------

rastervision2.pytorch_learner
-------------------------------

.. autoclass:: rastervision2.pytorch_learner.LearnerConfig
