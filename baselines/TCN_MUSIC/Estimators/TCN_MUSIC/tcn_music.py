"""Expose the TCN-MUSIC regressor and classifier through the common trainer API."""

from Estimators.TCN_MUSIC.tcn_music_model import TemporalConvolutionalNetworkMUSICModel
from Estimators.TCN_MUSIC.tcn_music_classifier_model import TemporalConvolutionalNetworkMUSICClassifierModel
from Estimators.DA_MUSIC.da_music import DeepAugmentedMUSIC


class TemporalConvolutionalNetworkMUSIC(DeepAugmentedMUSIC):
    def initiate_the_regressor(self):
        model = TemporalConvolutionalNetworkMUSICModel()
        return model

    def initiate_the_classifier(self):
        model = TemporalConvolutionalNetworkMUSICClassifierModel(self.regressor_model, self.device)
        return model

    def system_name(self, name=None):
        if name is None:
            self.name = "TCN-MUSIC"
        else:
            self.name = name
        return self.name
