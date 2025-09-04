import torch
from .baseCBM import _CBM


class LinearCBM(_CBM):
    def config_classifier(self):
        classifier = torch.nn.Linear(len(self.concepts), self.config.num_class)
        classifier.weight.data = self.init_weight_matrix()
        classifier.bias.data.zero_()
        return classifier

    def config_module(self):
        self.predictor.freeze()

    def configure_optimizers(self):
        op = torch.optim.Adam(params=self.classifier.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=op,
                                                      lr_lambda=lambda epoch: (1 - epoch / self.config.epochs) ** 0.9)
        return {'optimizer': op, 'lr_scheduler': scheduler}
