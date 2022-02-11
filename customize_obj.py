from deoxys.customize import custom_loss, custom_preprocessor, custom_datareader
from deoxys.data.preprocessor import BasePreprocessor
from deoxys.data.data_reader import H5PatchReader
from deoxys.data.data_generator import H5PatchGenerator
from elasticdeform import deform_random_grid
import numpy as np


@custom_preprocessor
class ChannelRepeater(BasePreprocessor):
    def __init__(self, channel=0):
        if '__iter__' not in dir(channel):
            self.channel = [channel]
        else:
            self.channel = channel

    def transform(self, images, targets):
        return np.concatenate([images, images[..., self.channel]], axis=-1), targets


@custom_preprocessor
class ElasticDeform(BasePreprocessor):
    def __init__(self, sigma=4, points=3):
        self.sigma = sigma
        self.points = points
    def transform(self, x, y):
        return deform_random_grid([x, y], axis=[(1, 2, 3), (1, 2, 3)],
                                  sigma=self.sigma, points=self.points)


@custom_datareader
class H5PatchReaderModified(H5PatchReader):
    def __init__(self, filename, batch_size=32, preprocessors=None,
                 x_name='x', y_name='y', batch_cache=10,
                 train_folds=None, test_folds=None, val_folds=None,
                 fold_prefix='fold',
                 patch_size=128, overlap=0.5, shuffle=False,
                 augmentations=False, preprocess_first=True,
                 drop_fraction=0.1, check_drop_channel=None,
                 bounding_box=False, actual_input_shape=None):
        super().__init__(filename, batch_size, preprocessors,
                 x_name, y_name, batch_cache,
                 train_folds, test_folds, val_folds,
                 fold_prefix,
                 patch_size, overlap, shuffle,
                 augmentations, preprocess_first,
                 drop_fraction, check_drop_channel,
                 bounding_box)
        self.actual_input_shape = actual_input_shape

    @property
    def test_generator(self):
        """

        Returns
        -------
        deoxys.data.DataGenerator
            A DataGenerator for generating batches of data for testing
        """
        return H5PatchGeneratorModified(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            folds=self.test_folds,
            patch_size=self.patch_size, overlap=self.overlap,
            shuffle=False, preprocess_first=self.preprocess_first,
            drop_fraction=0, actual_input_shape = self.actual_input_shape)

    @property
    def val_generator(self):
        """

        Returns
        -------
        deoxys.data.DataGenerator
            A DataGenerator for generating batches of data for validation
        """
        return H5PatchGeneratorModified(
            self.hf, batch_size=self.batch_size, batch_cache=self.batch_cache,
            preprocessors=self.preprocessors,
            x_name=self.x_name, y_name=self.y_name,
            folds=self.val_folds,
            patch_size=self.patch_size, overlap=self.overlap,
            shuffle=False, preprocess_first=self.preprocess_first,
            drop_fraction=0, actual_input_shape = self.actual_input_shape)
    
class H5PatchGeneratorModified(H5PatchGenerator):
    def __init__(self, h5_filename, batch_size=32, batch_cache=10,
                 preprocessors=None,
                 x_name='x', y_name='y',
                 folds=None,
                 patch_size=128, overlap=0.5,
                 shuffle=False,
                 augmentations=False, preprocess_first=True,
                 drop_fraction=0,
                 check_drop_channel=None,
                 bounding_box=False, actual_input_shape=None):
        super().__init__(h5_filename, batch_size, batch_cache,
                 preprocessors,
                 x_name, y_name,
                 folds,
                 patch_size, overlap,
                 shuffle,
                 augmentations, preprocess_first,
                 drop_fraction,
                 check_drop_channel,
                 bounding_box)

        self.actual_input_shape = actual_input_shape

    @property
    def description(self):
        super().description

        if self.actual_input_shape:
            self._description[0]['shape'] = tuple(self.actual_input_shape)
        
        return self._description
        

