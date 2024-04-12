import torch
import pytorch_lightning as pl
from model_lightning import SdeGenerativeModel
from utils import plot


class SdeGenerativeModel_GradientEstimation(SdeGenerativeModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
    
    def training_step(self, batch, batch_idx):
        loss = super(SdeGenerativeModel_GradientEstimation, self).training_step(batch, batch_idx)
        grad_norm_t =[]
        times=torch.randint(0, self.sde.N, (100,) ,device=batch.device)
        for time in times:
            labels = time.repeat(batch.shape[0],*time.shape)
            gradients = self.compute_grad(self.score_model, batch, t=labels)
            grad_norm = gradients.norm(2, dim=1).max().item()
            grad_norm_t.append(grad_norm)
        image = plot(times.cpu().numpy(),
                        grad_norm_t,
                        'Gradient Norms Epoch: ' + str(self.current_epoch)
                        )
        self.logger.experiment.add_image('grad_norms', image, self.current_epoch)
        return loss

     
    