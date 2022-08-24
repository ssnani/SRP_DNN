
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary


class SRPDNN_Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        pass


    def configure_optimizers(self):
        pass
    
    def forward(self, input):
        pass

    def training_step(self, train_batch, batch_idx):
        pass 

    def validation_step(self, val_batch, batch_idx):
        pass

    def test_step(self, test_batch, batch_idx):
        pass


if __name__ =='__main__':

    _path_info = paths()
    if _control_flags._TRAIN_: 

		train_dataset = IEEE_Dataset(_path_info.train_file_path)
		dev_dataset = IEEE_Dataset(_path_info.dev_file_path)

		train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4)
		val_loader = DataLoader(dev_dataset, batch_size=32, num_workers = 4)
		# model
		model = SRPDNN_Model()

		tb_logger = pl_loggers.TensorBoardLogger(save_dir=_path_info.log_path)
		checkpoint_callback = ModelCheckpoint(dirpath=_path_info.model_path, monitor='val_loss')
		model_summary = ModelSummary(max_depth=1)
		early_stopping = EarlyStopping('val_loss')

		# training
		trainer = pl.Trainer(gpus=4, num_nodes=1, precision=16,
						max_epochs = 100,
						callbacks=[checkpoint_callback, early_stopping, Losscallbacks(), model_summary],
						logger=tb_logger,
						strategy="ddp",
						check_val_every_n_epoch=1,
						num_sanity_val_steps=-1,
						profiler="simple",
						fast_dev_run=False)
					

		trainer.fit(model, train_loader, val_loader)

	else:

		test_dataset = IEEE_TestDataset(_path_info.test_file_path)

		test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)
		tb_logger = pl_loggers.TensorBoardLogger(save_dir=_path_info.log_path, version="")

		trainer = pl.Trainer(gpus=1, num_nodes=1, precision=16, #gpus=1, accelerator="cpu"
						callbacks=[Losscallbacks()],
						logger=tb_logger
						)

		model = SRPDNN_Model.load_from_checkpoint(_path_info.abs_model_file_name)

		trainer.test(model, dataloaders=test_loader )
    
