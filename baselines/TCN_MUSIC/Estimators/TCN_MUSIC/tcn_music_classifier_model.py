"""Attach a source-count classifier to the fitted TCN-MUSIC regressor features."""

import torch.cuda
import torch.nn as nn


class TemporalConvolutionalNetworkMUSICClassifierModel(nn.Module):
    def __init__(self,
                 regressor_model,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 ):
        super(TemporalConvolutionalNetworkMUSICClassifierModel, self).__init__()
        self.regressor_model = regressor_model

        self.classifier = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=4),
        ).to(device)

    def forward(self, x):
        with torch.no_grad():
            _, feature_for_number_of_sources_classification = self.regressor_model(x)

        feature_for_number_of_sources_classification = feature_for_number_of_sources_classification.detach()
        estimated_class = self.classifier(feature_for_number_of_sources_classification)
        return estimated_class
