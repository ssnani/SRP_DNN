import torch 
import torchaudio
import torchaudio.transforms as T
from torch.nn import functional as F
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing import Any, Optional

from dataset import benchmark2_array_setup
import numpy as np

class Losscallbacks(Callback):

    def __init__(self):
        self.mic_pos = benchmark2_array_setup.mic_pos
        """
        elevations = np.deg2rad(np.linspace(0,180,37))  #radians
		azimuths = np.deg2rad(np.linspace(0,360,73))

		theta_grid = torch.cartesian_prod(elevations, azimuths)
		self.u_theta = torch.cat( ( torch.cat( ( torch.sin(theta_grid[:,[0]])*torch.cos(theta_grid[:,[1]]), torch.sin(theta_grid[:,[0]])*torch.sin(theta_grid[:,[1]]) ), dim=1), torch.cos(theta_grid[:,[0]]) ), dim = 1)
		pi = math.pi
		self.w = torch.from_numpy(np.array([(2*pi/512)*k for k in range(1,257)])).unsqueeze(dim=0).to(torch.float32)
        """
    def on_test_batch_end_numpy(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
        ) -> None:
        pass

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
        ) -> None:

        _, tgt_dp_phdiff_grid, doA = batch

        batch_size, num_mic_pairs, num_tous, n_freq = tgt_dp_phdiff_grid.shape
        tgt_dp_phdiff_grid = torch.reshape(tgt_dp_phdiff_grid, (batch_size*num_mic_pairs, num_tous, n_freq))

        #tgt_dp_phdiff_grid = tgt_dp_phdiff_grid[:48,:,:]

        est_phdiff = outputs['est_phdiff']
        est_phdiff = torch.permute(est_phdiff, (1,0,2))         #(num_micpairs, n_frames, n_freq) -> (n_frames, num_mic_pairs, 512)
        est_phdiff = est_phdiff.unsqueeze(dim=3)#.to(dtype=torch.float32)

       # print(f"tgt_dp_phdiff_grid: {tgt_dp_phdiff_grid.shape} , {tgt_dp_phdiff_grid.dtype}, est_phdiff: {est_phdiff.shape}, {est_phdiff.dtype}")
        """
        dot_p = torch.matmul(tgt_dp_phdiff_grid, est_phdiff) #to avoid cuda memory error
        SRP = torch.mean(dot_p.squeeze(), dim=1) / 256
        _doA_info = torch.max(SRP, dim=1)
        _doA_est_indices  = _doA_info[1]
        """
        
        n_frames = est_phdiff.shape[0]

        dot_p = torch.matmul(tgt_dp_phdiff_grid, est_phdiff[: n_frames//2,:,:] ) #to avoid cuda memory error
        SRP = torch.mean(dot_p.squeeze(), dim=1) / 256
        del dot_p
        torch.cuda.empty_cache()

        _doA_info = torch.max(SRP, dim=1)
        _doA_est_indices  = _doA_info[1]

        dot_p = torch.matmul(tgt_dp_phdiff_grid, est_phdiff[n_frames//2:,:,:] ) #to avoid cuda memory error
        SRP2 = torch.mean(dot_p.squeeze(), dim=1) / 256
        del dot_p
        torch.cuda.empty_cache()

        _doA_info = torch.max(SRP2, dim=1)
        _doA_est_indices2  = _doA_info[1]

        _doA_est_indices = torch.cat((_doA_est_indices,_doA_est_indices2), dim=0)
        SRP = torch.cat((SRP,SRP2), dim=0)
        
        torch.save({"tgt_dp_phdiff_grid": tgt_dp_phdiff_grid, "est_phdiff": est_phdiff, "SRP": SRP }, 'dbg_SRP_float32_data_norm_last_ckpt.pt')

        #indices to angle map
        elevation = (_doA_est_indices // 73)*5
        azimuth = (_doA_est_indices % 73)*5   #-180 + (_doA_est_indices % 73)*5

        batch_size, out_frames, angles = doA.shape
        doA = torch.reshape(torch.rad2deg(doA), (batch_size*out_frames, angles))   #doA in degrees

        #print(f"elevation: {elevation.shape}, doA: {doA.shape}")
        elevation_result = torch.abs(elevation - doA[:,0])<=30
        elevation_result = torch.mean(elevation_result.float())
        azimuth_result = torch.abs(azimuth - doA[:,1])<=30
        azimuth_result = torch.mean(azimuth_result.float())

        self.log("elevation_result", elevation_result, on_step=True,logger=True)
        self.log("azimuth_result", azimuth_result, on_step=True,logger=True)
        
        _dct = {"elevation_pred": elevation, "azimuth_pred": azimuth,"elevation_label": doA[:,0], "azimuth_label": doA[:,1] }
        print(_dct)
        #self.log_dict({"elevation_pred": elevation, "azimuth_pred": azimuth,"elevation_label": doA[:,0], "azimuth_label": doA[:,1] }, on_step=True,logger=True)

        return




class GradNormCallback(Callback):
    """
    Logs the gradient norm.
    """

    def ___on_after_backward(self, trainer: "pl.Trainer", model): 
        grad_norm_dict = gradient_norm_per_layer(model)
        self.log_dict(grad_norm_dict)

    def on_before_optimizer_step(self, trainer, model, optimizer, optimizer_idx=0): #trainer, model, 
        grad_norm_dict = gradient_norm_per_layer(model)
        self.log_dict(grad_norm_dict)

def gradient_norm_per_layer(model):
    total_norm = {}
    for layer_name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.detach().data.norm(2)
            total_norm[layer_name] = param_norm #.item() ** 2
    #total_norm = total_norm ** (1. / 2)
    return total_norm






