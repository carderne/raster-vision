from os.path import join
import uuid
import zipfile
import glob

from rastervision.v2.rv.backend import Backend
from rastervision.v2.rv.utils.misc import save_img
from rastervision.v2.core.filesystem import (
    get_local_path, make_dir, upload_or_copy)


class PyTorchChipClassification(Backend):
    def __init__(self, task, learner, tmp_dir):
        self.task = task
        self.learner = learner
        self.tmp_dir = tmp_dir

    def process_scene_data(self, scene, data):
        """Make training chips for a scene.

        This writes a set of image chips to {scene_id}/{class_name}/{scene_id}-{ind}.png

        Args:
            scene: (rv.data.Scene)
            data: (rv.data.Dataset)

        Returns:
            (str) path to directory with scene chips {tmp_dir}/{scene_id}
        """
        scene_dir = join(self.tmp_dir, str(scene.id))

        for ind, (chip, window, labels) in enumerate(data):
            class_id = labels.get_cell_class_id(window)
            # If a chip is not associated with a class, don't
            # use it in training data.
            if class_id is None:
                continue

            class_name = self.task.dataset.class_config.names[class_id]
            class_dir = join(scene_dir, class_name)
            make_dir(class_dir)
            chip_path = join(class_dir, '{}-{}.png'.format(scene.id, ind))
            save_img(chip, chip_path)

        return scene_dir

    def process_sceneset_results(self, training_results, validation_results):
        """Write zip file with chips for a set of scenes.

        This writes a zip file for a group of scenes at {chip_uri}/{uuid}.zip containing:
        train-img/{class_name}/{scene_id}-{ind}.png
        valid-img/{class_name}/{scene_id}-{ind}.png

        This method is called once per instance of the chip command.
        A number of instances of the chip command can run simultaneously to
        process chips in parallel. The uuid in the path above is what allows
        separate instances to avoid overwriting each others' output.

        Args:
            training_results: list of directories generated by process_scene_data
                that all hold training chips
            validation_results: list of directories generated by process_scene_data
                that all hold validation chips
        """
        group = str(uuid.uuid4())
        group_uri = join(self.task.chip_uri, '{}.zip'.format(group))
        group_path = get_local_path(group_uri, self.tmp_dir)
        make_dir(group_path, use_dirname=True)

        with zipfile.ZipFile(group_path, 'w', zipfile.ZIP_DEFLATED) as zipf:

            def _write_zip(scene_dirs, split):
                for scene_dir in scene_dirs:
                    scene_paths = glob.glob(join(scene_dir, '**/*.png'))
                    for path in scene_paths:
                        class_name, fn = path.split('/')[-2:]
                        zipf.write(path, join(split, class_name, fn))

            _write_zip(training_results, 'train')
            _write_zip(validation_results, 'valid')

        upload_or_copy(group_path, group_uri)

    def train(self):
        learner = self.learner.get_learner()(self.learner, self.tmp_dir)
        learner.main()

    def load_model(self):
        pass

    def predict(self, chips, windows):
        pass