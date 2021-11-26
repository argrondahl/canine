from deoxys.customize import custom_loss, custom_preprocessor
from deoxys.data.preprocessor import BasePreprocessor
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
