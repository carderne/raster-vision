.. rst-class:: hide-header

.. currentmodule:: rastervision

|

.. image:: _static/raster-vision-logo-index.png
    :align: center
    :target: https://rastervision.io

|

**Raster Vision** is an open source framework for Python developers building computer
vision models on satellite, aerial, and other large imagery sets (including
oblique drone imagery). There is built-in support for chip classification, object detection, and semantic segmentation using PyTorch and Tensorflow.

.. image:: _static/cv-tasks.png
    :align: center

Raster Vision allows engineers to quickly and repeatably
configure *experiments* that go through core components of a machine learning
workflow: analyzing training data, creating training chips, training models,
creating predictions, evaluating models, and bundling the model files and
configuration for easy deployment.

.. image:: _static/overview-raster-vision-workflow.png
    :align: center

Raster Vision workflows begin when you have a set of images and training data,
optionally with Areas of Interest (AOIs) that describe where the images are labeled. Raster Vision
workflows end with a packaged model and configuration that allows you to
easily utilize models in various  deployment situations. Inside the Raster Vision
workflow, there's the process of running multiple experiments to find the best model
or models to deploy.

The process of running experiments includes executing workflows that perform the following
commands:

* **ANALYZE**: Gather dataset-level statistics and metrics for use in downstream processes.
* **CHIP**: Create training chips from a variety of image and label sources.
* **TRAIN**: Train a model using a variety of "backends" such as TensorFlow or Keras.
* **PREDICT**: Make predictions using trained models on validation and test data.
* **EVAL**: Derive evaluation metrics such as F1 score, precision and recall against the model's predictions on validation datasets.
* **BUNDLE**: Bundle the trained model into a :ref:`predict package`, which can be deployed in batch processes, live servers, and other workflows.

Experiments are configured using a fluent builder pattern that makes configuration easy to read, reuse
and maintain.

.. click:example::

    # tiny_spacenet.py

    import rastervision as rv

    class TinySpacenetExperimentSet(rv.ExperimentSet):
        def exp_main(self):
            base_uri = ('https://s3.amazonaws.com/azavea-research-public-data/'
                        'raster-vision/examples/spacenet')
            train_image_uri = '{}/RGB-PanSharpen_AOI_2_Vegas_img205.tif'.format(base_uri)
            train_label_uri = '{}/buildings_AOI_2_Vegas_img205.geojson'.format(base_uri)
            val_image_uri = '{}/RGB-PanSharpen_AOI_2_Vegas_img25.tif'.format(base_uri)
            val_label_uri = '{}/buildings_AOI_2_Vegas_img25.geojson'.format(base_uri)
            channel_order = [0, 1, 2]
            background_class_id = 2

            # ------------- TASK -------------

            task = rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                                .with_chip_size(300) \
                                .with_chip_options(chips_per_scene=50) \
                                .with_classes({
                                    'building': (1, 'red'),
                                    'background': (2, 'black')
                                }) \
                                .build()

            # ------------- BACKEND -------------

            backend = rv.BackendConfig.builder(rv.PYTORCH_SEMANTIC_SEGMENTATION) \
                .with_task(task) \
                .with_train_options(
                    batch_size=2,
                    num_epochs=1,
                    debug=True) \
                .build()

            # ------------- TRAINING -------------

            train_raster_source = rv.RasterSourceConfig.builder(rv.RASTERIO_SOURCE) \
                                                       .with_uri(train_image_uri) \
                                                       .with_channel_order(channel_order) \
                                                       .with_stats_transformer() \
                                                       .build()

            train_label_raster_source = rv.RasterSourceConfig.builder(rv.RASTERIZED_SOURCE) \
                                                             .with_vector_source(train_label_uri) \
                                                             .with_rasterizer_options(background_class_id) \
                                                             .build()
            train_label_source = rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                                                     .with_raster_source(train_label_raster_source) \
                                                     .build()

            train_scene =  rv.SceneConfig.builder() \
                                         .with_task(task) \
                                         .with_id('train_scene') \
                                         .with_raster_source(train_raster_source) \
                                         .with_label_source(train_label_source) \
                                         .build()

            # ------------- VALIDATION -------------

            val_raster_source = rv.RasterSourceConfig.builder(rv.RASTERIO_SOURCE) \
                                                     .with_uri(val_image_uri) \
                                                     .with_channel_order(channel_order) \
                                                     .with_stats_transformer() \
                                                     .build()

            val_label_raster_source = rv.RasterSourceConfig.builder(rv.RASTERIZED_SOURCE) \
                                                           .with_vector_source(val_label_uri) \
                                                           .with_rasterizer_options(background_class_id) \
                                                           .build()
            val_label_source = rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                                                   .with_raster_source(val_label_raster_source) \
                                                   .build()

            val_scene = rv.SceneConfig.builder() \
                                      .with_task(task) \
                                      .with_id('val_scene') \
                                      .with_raster_source(val_raster_source) \
                                      .with_label_source(val_label_source) \
                                      .build()

            # ------------- DATASET -------------

            dataset = rv.DatasetConfig.builder() \
                                      .with_train_scene(train_scene) \
                                      .with_validation_scene(val_scene) \
                                      .build()

            # ------------- EXPERIMENT -------------

            experiment = rv.ExperimentConfig.builder() \
                                            .with_id('tiny-spacenet-experiment') \
                                            .with_root_uri('/opt/data/rv') \
                                            .with_task(task) \
                                            .with_backend(backend) \
                                            .with_dataset(dataset) \
                                            .with_stats_analyzer() \
                                            .build()

            return experiment


    if __name__ == '__main__':
        rv.main()

Raster Vision uses a ``unittest``-like method for executing experiments. For instance, if the
above was defined in `tiny_spacenet.py`, with the proper setup you could run the experiment
on AWS Batch by running:

.. code:: shell

   > rastervision run aws_batch -p tiny_spacenet.py

See the :ref:`quickstart` for a more complete description of using this example.


.. _documentation:

Documentation
=============

This part of the documentation guides you through all of the library's
usage patterns.

.. toctree::
   :maxdepth: 2

   why
   quickstart
   setup
   experiments
   commands
   runners
   predictor
   cli
   misc
   codebase
   plugins
   CONTRIBUTING
   release

API Reference
-------------

If you are looking for information on a specific function, class, or
method, this part of the documentation is for you.

.. toctree::
   :maxdepth: 10

   api

RV2 Documentation
==================

This is the documentation for a major refactor of RV, and is a work in progress.

.. toctree::
   :maxdepth: 2

   rv2/quickstart
   rv2/setup
   rv2/cli
   rv2/architecture
   rv2/config

CHANGELOG
---------

.. toctree::
   :maxdepth: 3

   changelog
