import torch as t

class SAE_Loss(t.nn.Module):
    def __init__(self, l1_weight=0.01):
        super(SAE_Loss, self).__init__()
        self.mse = t.nn.MSELoss(reduction='mean')
        self.l1 = t.nn.L1Loss(reduction='mean')

        self.l1_weight  = l1_weight
        
    def forward(self, feature_values, target_activations, reconstructed_activations):
        mse_loss = self.mse(reconstructed_activations, target_activations)
        l1_loss = self.l1(feature_values, t.zeros_like(feature_values))
        return mse_loss + l1_loss * self.l1_weight