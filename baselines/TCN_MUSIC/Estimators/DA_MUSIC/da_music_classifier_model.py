"""Attach a source-count classifier to the fitted DA-MUSIC regressor features."""

import torch.cuda
import torch.nn as nn


class DAMUSICClassifierModel(nn.Module):
    def __init__(self,
                 regressor_model,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 ):
        super(DAMUSICClassifierModel, self).__init__()
        self.regressor_model = regressor_model

        self.classifier = nn.Sequential(
            nn.Linear(in_features=16, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=4),
        ).to(device)

    def forward(self, x):
        with torch.no_grad():
            _, eigenvalues = self.regressor_model(x)

        eigenvalues = eigenvalues.detach()
        # [batch_size, 2M=16] -> [batch_size, 4]
        estimated_class = self.classifier(eigenvalues)
        return estimated_class
