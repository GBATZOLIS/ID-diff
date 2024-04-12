import losses
import torch
import pytorch_lightning as pl
import sde_lib
import sampling
from models.ema import ExponentialMovingAverage
from models import utils as mutils
from utils import scatter
from models import ddpm, ncsnv2, fcn
from . import BaseSdeGenerativeModel
from iunets.layers import InvertibleDownsampling2D
from . import utils
from sampling.unconditional import get_inpainting_fn

def permute_channels(haar_image, forward=True):
        permuted_image = torch.zeros_like(haar_image)
        if forward:
            for i in range(4):
                if i == 0:
                    k = 1
                elif i == 1:
                    k = 0
                else:
                    k = i
                for j in range(3):
                    permuted_image[:, 3*k+j, :, :] = haar_image[:, 4*j+i, :, :]
        else:
            for i in range(4):
                if i == 0:
                    k = 1
                elif i == 1:
                    k = 0
                else:
                    k = i
                
                for j in range(3):
                    permuted_image[:,4*j+k,:,:] = haar_image[:, 3*i+j, :, :]

        return permuted_image

@utils.register_lightning_module(name='haar_multiscale')
class HaarMultiScaleSdeGenerativeModel(BaseSdeGenerativeModel.BaseSdeGenerativeModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.haar_transform = InvertibleDownsampling2D(3, stride=2, method='cayley', init='haar', learnable=False)
        self.inpainting_fn = get_inpainting_fn(config, self.sde, self.sampling_eps)
    
    def training_step(self, batch, batch_idx):
        batch = self.haar_transform(batch) #apply the haar transform
        batch = permute_channels(batch) #group the frequency bands: 0:3->LL, 3:6->LH, 6:9->HL, 9:12->HH
        loss = self.train_loss_fn(self.score_model, batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch = self.haar_transform(batch) 
        batch = permute_channels(batch)
        loss = self.eval_loss_fn(self.score_model, batch)
        self.log('eval_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def sample(self, show_evolution=False, space='haar'):
        if space=='haar':
            return self.sampling_fn(self.score_model, show_evolution=show_evolution)
        elif space=='image':
            samples=self.sampling_fn(self.score_model, show_evolution=show_evolution)
            back_permuted_samples = permute_channels(samples, forward=False)
            image = self.haar_transform.inverse(back_permuted_samples)
            return image
    
    def inpaint(self, dc_coefficients, space='haar'):
        #input:dc coefficients
        #output: the entire image in haar or image space
        #functionality: inpaints the detail coefficients given the dc coefficients
        shape = dc_coefficients.shape
        new_shape = tuple([shape[0], shape[1]*4]+list(shape[2:]))
        up_sample = torch.zeros(new_shape).astype(dc_coefficients)
        up_sample[:,:3,::] = dc_coefficients
        mask = torch.cat([torch.ones(3, dtype=torch.float32), torch.zeros(up_sample.size(1)-3, dtype=torch.float32)]).astype(up_sample)
        mask = mask.view(1, up_sample.size(1), 1, 1)

        inpainted_haar, _ = self.inpainting_fn(self.score_model, up_sample, mask)

        if space=='haar':
            return inpainted_haar
        elif space=='image':
            inpainted_haar = permute_channels(inpainted_haar, forward=False)
            image = self.haar_transform.inverse(inpainted_haar)
            return image


    


    
    







#what follows is the sampling code for multiple scales
"""
    def load_scale_models(configs, workdir, num_samples):
    # Initialize models.
    scale = {}
    for config in configs:
        scale[config.data.image_size]={}

        score_model = mutils.create_model(config)
        print('Model trainable parameters: ', sum(p.numel() for p in score_model.parameters() if p.requires_grad))
        ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
        optimizer = losses.get_optimizer(config, score_model.parameters())
        state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0, epoch=0)

        checkpoint_meta_dir = os.path.join(workdir, 'resolution_%d' % config.data.image_size, "checkpoints-meta", "checkpoint.pth")
        # Resume training when intermediate checkpoints are detected
        state = restore_checkpoint(checkpoint_meta_dir, state, torch.device('cpu'))
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())
        score_model.to(torch.device('cuda:0'))
        score_model.eval()
        scale[config.data.image_size]['score_model'] = score_model

        sde, sampling_eps = load_sde(config)
        scale[config.data.image_size]['sde'] = sde

        sampling_shape = (num_samples, config.data.num_channels,
                        config.data.effective_image_size, config.data.effective_image_size)
        scale[config.data.image_size]['sampling_shape'] = sampling_shape

        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, sampling_eps)
        scale[config.data.image_size]['sampling_fn'] = sampling_fn

        inpainting_fn = sampling.get_inpainting_fn(config, sde, sampling_shape, sampling_eps)
        scale[config.data.image_size]['inpainting_fn'] = inpainting_fn
    
    return scale


    def auto_regressive_sampling(configs, workdir, eval_folder, num_samples):
    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    tf.io.gfile.makedirs(eval_dir)
    this_sample_dir = os.path.join(eval_dir, "autoregressive_sampling")
    tf.io.gfile.makedirs(this_sample_dir)

    # Create the Haar Transform
    haar_transform = InvertibleDownsampling2D(3, stride=2, method='cayley', init='haar', learnable=False).to(torch.device('cuda:0'))
    scale = load_scale_models(configs, workdir, num_samples)
    with torch.no_grad():
        for i, resolution in tqdm(enumerate(sorted(scale.keys()))):
        if i==0:
            sample, n = scale[resolution]['sampling_fn'](scale[resolution]['score_model'])
            sample = permute_channels(sample, forward=False)
            sample = haar_transform.inverse(sample)
            print(sample.size())

            fout = os.path.join(this_sample_dir, "auto_regressive_sampling_resolution_%d.png" % resolution)
            grid = make_grid(normalise_per_image(sample.cpu()), nrow=int(np.sqrt(sample.size(0))))
            save_image(grid, fout)

        else:
            up_sample = torch.zeros(scale[resolution]['sampling_shape']).to(torch.device('cuda:0'))
            up_sample[:,:3,::] = sample
            sample = up_sample

            inpainting_mask = torch.cat([torch.ones(3, dtype=torch.float32), torch.zeros(sample.size(1)-3, dtype=torch.float32)]).to(torch.device('cuda:0')).view(1, sample.size(1), 1, 1)
            sample = scale[resolution]['inpainting_fn'](scale[resolution]['score_model'], sample, inpainting_mask, return_evolution=False)
            sample = permute_channels(sample, forward=False)
            sample = haar_transform.inverse(sample)

            fout = os.path.join(this_sample_dir, "auto_regressive_sampling_resolution_%d.png" % resolution)
            grid = make_grid(normalise_per_image(sample.cpu()), nrow=int(np.sqrt(sample.size(0))))
            save_image(grid, fout)

    def super_resolution_sampling(configs, workdir, eval_folder, num_samples):
    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    tf.io.gfile.makedirs(eval_dir)
    this_sample_dir = os.path.join(eval_dir, "super_resolution_sampling")
    tf.io.gfile.makedirs(this_sample_dir)

    config = configs[-1] #smallest_scale_config
    test_dataset = datasets.HaarDecomposedDataset(config, config.data.uniform_dequantization, phase='test')
    test_dataloader = DataLoader(test_dataset, batch_size=num_samples, num_workers=config.eval.workers, shuffle=False)

    target_resolution = configs[0].data.image_size
    smallest_resolution = configs[-1].data.image_size

    # Create the Haar Transform
    haar_transform = InvertibleDownsampling2D(3, stride=2, method='cayley', init='haar', learnable=False).to(torch.device('cuda:0'))
    scale = load_scale_models(configs, workdir, num_samples)

    with torch.no_grad():
        for j, test_batch in enumerate(test_dataloader):
        tf.io.gfile.makedirs(os.path.join(this_sample_dir, 'batch_%d' % j))
        for i, resolution in tqdm(enumerate(sorted(scale.keys()))):
            if i==0:
            sample = test_batch.to(torch.device('cuda:0'))
            fout = os.path.join(this_sample_dir, 'batch_%d' % j, "resolution_%d.png" % resolution)
            grid = make_grid(normalise_per_image(sample.cpu()), nrow=int(np.sqrt(sample.size(0))))
            save_image(grid, fout)

            else:
            up_sample = torch.zeros(scale[resolution]['sampling_shape']).to(torch.device('cuda:0'))
            up_sample[:,:3,::] = sample
            sample = up_sample

            inpainting_mask = torch.cat([torch.ones(3, dtype=torch.float32), torch.zeros(sample.size(1)-3, dtype=torch.float32)]).to(torch.device('cuda:0')).view(1, sample.size(1), 1, 1)
            sample = scale[resolution]['inpainting_fn'](scale[resolution]['score_model'], sample, inpainting_mask, return_evolution=False)
            sample = permute_channels(sample, forward=False)
            sample = haar_transform.inverse(sample)

            fout = os.path.join(this_sample_dir, 'batch_%d' % j, "super_resolution_%d.png" % resolution)
            grid = make_grid(normalise_per_image(sample.cpu()), nrow=int(np.sqrt(sample.size(0))))
            save_image(grid, fout)
"""