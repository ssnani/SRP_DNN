
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary


from network import SRPDNN
from dataset_2 import Parameter, LibriSpeechDataset, RandomTrajectoryDataset, LocataDataset, benchmark2_array_setup, paths, SRPDNN_features, SRPDNN_Test_features
from debug_flags import _control_flags
from os.path import exists
from callbacks import Losscallbacks

class SRPDNN_Model(pl.LightningModule):
	def __init__(self):
		super().__init__()
		self.model = SRPDNN(256) 
		
	def configure_optimizers(self):
		_optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
		_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(_optimizer, mode='min',factor=0.5, patience=3)
		return {"optimizer": _optimizer, "lr_scheduler": _lr_scheduler, "monitor": 'val_loss'}
	
	def forward(self, input_batch):
		log_ms_ph, tgt_dp_phdiff, doA = input_batch

		num_mic_pairs, n_chnls, n_freq, n_frms = log_ms_ph.shape
		num_mic_pairs, reduced_n_frms, out_n_freq = tgt_dp_phdiff.shape

		pred_phdiff = self.model(log_ms_ph)
		return pred_phdiff, tgt_dp_phdiff


	def __forward(self, input_batch):

		#to avoid gpu memory errors
		log_ms_ph, tgt_dp_phdiff, doA = input_batch

		batch_size, num_mic_pairs, n_chnls, n_freq, n_frms = log_ms_ph.shape
		
		batch_size, num_mic_pairs, reduced_n_frms, out_n_freq = tgt_dp_phdiff.shape

		log_ms_ph = torch.reshape(log_ms_ph, (batch_size*num_mic_pairs, n_chnls, n_freq, n_frms))
		
		tgt_dp_phdiff = torch.reshape(tgt_dp_phdiff, (batch_size*num_mic_pairs, reduced_n_frms, out_n_freq))

		red_batch_size = log_ms_ph.shape[0] //2
		pred_phdiff = self.model(log_ms_ph[:red_batch_size, :,:,:])
		pred_phdiff2 = self.model(log_ms_ph[red_batch_size:, :,:,:])
		pred_phdiff = torch.cat((pred_phdiff, pred_phdiff2), dim=0)

		return pred_phdiff, tgt_dp_phdiff


	def training_step(self, train_batch, batch_idx):
		pred_phdiff, tgt_dp_phdiff = self.forward(train_batch)
		loss = F.mse_loss(pred_phdiff, tgt_dp_phdiff)
		self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		return {"loss" : loss , "est_phdiff" : pred_phdiff }


	def validation_step(self, val_batch, batch_idx):
		pred_phdiff, tgt_dp_phdiff = self.forward(val_batch)
		loss = F.mse_loss(pred_phdiff, tgt_dp_phdiff)
		self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		return {"loss" : loss , "est_phdiff" : pred_phdiff }

	def test_step(self, test_batch, batch_idx):
		pred_phdiff, _ = self.forward(test_batch)
		#loss = F.mse_loss(pred_phdiff, tgt_dp_phdiff)
		#self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		return { "est_phdiff" : pred_phdiff }		#"loss" : loss ,


if __name__ =='__main__':

	_path_info = paths()
	print(_control_flags)
	if _control_flags._TRAIN_: 

		T = 20 										# Trajectory length (s)
		array_setup = benchmark2_array_setup
		nb_points = 64
		room_sz = Parameter([3,3,2.5], [10,8,6]) 	# Random room sizes from 3x3x2.5 to 10x8x6 meters
		T60 = Parameter(0.2, 1.3)					# Random reverberation times from 0.2 to 1.3 seconds
		abs_weights = Parameter([0.5]*6, [1.0]*6)  # Random absorption weights ratios between walls
		SNR = Parameter(5, 30)
		array_pos = Parameter([0.4, 0.1, 0.1],[0.6, 0.9, 0.5]) 

		sourceDataset = LibriSpeechDataset(_path_info.train_file_path, T, return_vad=True)
		train_dataset = RandomTrajectoryDataset(sourceDataset, room_sz, T60, abs_weights, array_setup, array_pos, SNR, nb_points, transforms=[SRPDNN_features()])

		dev_sourceDataset = LibriSpeechDataset(_path_info.dev_file_path, T, return_vad=True)
		dev_dataset = RandomTrajectoryDataset(dev_sourceDataset, room_sz, T60, abs_weights, array_setup, array_pos, SNR, nb_points, transforms=[SRPDNN_features()])


		train_loader = DataLoader(train_dataset, batch_size=32)#, num_workers=2)
		val_loader = DataLoader(dev_dataset, batch_size=32)#, num_workers = 2)

		# model
		model = SRPDNN_Model()

		tb_logger = pl_loggers.TensorBoardLogger(save_dir=_path_info.log_path, version="train_Single_MovingSrc_lr1e-4_32bit_precision")
		checkpoint_callback = ModelCheckpoint(dirpath=_path_info.model_path, save_last = True, save_top_k=1, monitor='val_loss')
		model_summary = ModelSummary(max_depth=1)
		early_stopping = EarlyStopping('val_loss')

		# training
		trainer = pl.Trainer(gpus=4, num_nodes=1, precision=32,
						max_epochs = 100,
						callbacks=[checkpoint_callback, early_stopping, model_summary],
						logger=tb_logger,
						strategy="ddp_find_unused_parameters_false", #ddp",
						check_val_every_n_epoch=1,
						log_every_n_steps = 1,
						num_sanity_val_steps=-1,
						profiler="simple",
						fast_dev_run=False)
					

		if exists(_path_info.abs_last_model_file_name):
			trainer.fit(model, train_loader, val_loader, ckpt_path=_path_info.abs_last_model_file_name)
		else:
			trainer.fit(model, train_loader, val_loader)

	else:
		if not _control_flags._TEST_LOCATA_:
			T = 20 										# Trajectory length (s)
			array_setup = benchmark2_array_setup
			nb_points = 64
			room_sz = Parameter([3,3,2.5], [10,8,6]) 	# Random room sizes from 3x3x2.5 to 10x8x6 meters
			T60 = Parameter(0.2, 1.3)					# Random reverberation times from 0.2 to 1.3 seconds
			abs_weights = Parameter([0.5]*6, [1.0]*6)  # Random absorption weights ratios between walls
			SNR = Parameter(5, 30)
			array_pos = Parameter([0.4, 0.1, 0.1],[0.6, 0.9, 0.5]) 

			sourceDataset = LibriSpeechDataset(_path_info.test_file_path, T, return_vad=True)
			test_dataset = RandomTrajectoryDataset(sourceDataset, room_sz, T60, abs_weights, array_setup, array_pos, SNR, nb_points, transforms=[SRPDNN_features(1,2)])
		else:
			_locata_speech_path = '/fs/scratch/PAS0774/Shanmukh/Databases/Locata/eval/'
			test_dataset = LocataDataset(_locata_speech_path, 'benchmark2', 16000, [3], dev=True, transforms=[SRPDNN_Test_features(1,2)])

		test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)
		tb_logger = pl_loggers.TensorBoardLogger(save_dir=_path_info.log_path, version=_control_flags.test_exp_name)

		trainer = pl.Trainer(gpus=1, num_nodes=1, precision=32, #gpus=1, accelerator="cpu"
						callbacks=[Losscallbacks()],
						logger=tb_logger
						)

		model = SRPDNN_Model.load_from_checkpoint(_path_info.abs_model_file_name)

		trainer.test(model, dataloaders=test_loader )
	
