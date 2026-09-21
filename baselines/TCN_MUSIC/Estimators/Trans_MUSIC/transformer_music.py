"""Expose the Trans-MUSIC regressor and classifier through the common trainer API."""

from Estimators.DA_MUSIC.da_music import DeepAugmentedMUSIC
from Estimators.Trans_MUSIC.transformer_music_model import TransMUSICModel
from Estimators.Trans_MUSIC.transformer_music_classifier_model import TransMUSICClassifierModel


class TransformerMUSIC(DeepAugmentedMUSIC):
    def initiate_the_regressor(self):
        model = TransMUSICModel(self.angle_grids, self.rx_ula, self.device)
        return model

    def initiate_the_classifier(self):
        model = TransMUSICClassifierModel(self.regressor_model, self.device)
        return model

    def system_name(self, name=None):
        if name is None:
            self.name = "Trans-MUSIC"
        else:
            self.name = name
        return self.name
