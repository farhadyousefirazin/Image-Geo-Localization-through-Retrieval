import torch
import numpy as np
import torchvision.models
import pytorch_lightning as pl
from torchvision import transforms as tfm
from pytorch_metric_learning import losses
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import logging
from os.path import join

import utils
import parser
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset

class LightningModel(pl.LightningModule):
    def __init__(self, test_dataset, descriptors_dim=512, num_preds_to_save=0, save_only_wrong_preds=True):
        super().__init__()
        self.test_dataset = test_dataset
        self.num_preds_to_save = num_preds_to_save
        self.save_only_wrong_preds = save_only_wrong_preds
        self.model = torchvision.models.resnet18(pretrained=True)
        
        # Truncate the model at conv3 (layer2 in PyTorch ResNet implementation)
        self.model = torch.nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
            self.model.layer1,
            self.model.layer2
        )
        
        # Replace the average pooling layer with AdaptiveAvgPool2d
        self.pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer for the desired descriptors dimension
        self.fc = torch.nn.Linear(self.model[-1][-1].conv2.out_channels, descriptors_dim)
        
        # Set the loss function
        self.loss_fn = losses.CircleLoss(m=0.25, gamma=64)

    def load_model_state_dict(self, state_dict):
        model_state_dict = self.state_dict()
        for key in list(state_dict.keys()):
            if key not in model_state_dict:
                print(f"Skipping unexpected key {key} in state_dict")
                del state_dict[key]
        self.load_state_dict(state_dict)

    def forward(self, images):
        x = self.model(images)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        descriptors = self.fc(x)
        return descriptors
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2.5e-3, weight_decay=1e-3)
        return optimizer

    def loss_function(self, descriptors, labels):
        loss = self.loss_fn(descriptors, labels)
        return loss

    def training_step(self, batch, batch_idx):
        images, labels = batch
        num_places, num_images_per_place, C, H, W = images.shape
        images = images.view(num_places * num_images_per_place, C, H, W)
        labels = labels.view(num_places * num_images_per_place)

        descriptors = self(images)
        loss = self.loss_function(descriptors, labels)
        
        self.log('loss', loss.item(), logger=True)
        return {'loss': loss}

    def inference_step(self, batch):
        images, _ = batch
        descriptors = self(images)
        return descriptors.cpu().numpy().astype(np.float32)

    def validation_step(self, batch, batch_idx):
        return self.inference_step(batch)

    def test_step(self, batch, batch_idx):
        return self.inference_step(batch)

    def validation_epoch_end(self, all_descriptors):
        return self.inference_epoch_end(all_descriptors, self.val_dataset, 'val')

    def test_epoch_end(self, all_descriptors):
        return self.inference_epoch_end(all_descriptors, self.test_dataset, 'test', self.num_preds_to_save)

    def inference_epoch_end(self, all_descriptors, inference_dataset, split, num_preds_to_save=0):
        all_descriptors = np.concatenate(all_descriptors)
        queries_descriptors = all_descriptors[inference_dataset.database_num:]
        database_descriptors = all_descriptors[:inference_dataset.database_num]

        recalls, recalls_str = utils.compute_recalls(
            inference_dataset, queries_descriptors, database_descriptors,
            self.logger.log_dir, num_preds_to_save, self.save_only_wrong_preds
        )

        logging.info(f"Epoch[{self.current_epoch:02d}]): " +
                      f"recalls: {recalls_str}")

        self.log(f'{split}/R@1', recalls[0], prog_bar=False, logger=True)
        self.log(f'{split}/R@5', recalls[1], prog_bar=False, logger=True)

def get_datasets_and_dataloaders(args):
    train_transform = tfm.Compose([
        tfm.RandAugment(num_ops=3),
        tfm.ToTensor(),
        tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    #train_dataset = TrainDataset(
        #dataset_folder=args.train_path,
        #img_per_place=args.img_per_place,
        #min_img_per_place=args.min_img_per_place,
        #transform=train_transform
    #)
    #val_dataset = TestDataset(dataset_folder=args.val_path)
    test_dataset = TestDataset(dataset_folder=args.test_path)
    #train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                              #shuffle=True)
    #val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    return test_dataset, test_loader


if __name__ == '__main__':
    args = parser.parse_arguments()
    #utils.setup_logging(join('logs', 'lightning_logs', args.exp_name), console='info')

    test_dataset, test_loader = get_datasets_and_dataloaders(args)
    model = LightningModel(test_dataset, args.descriptors_dim, args.num_preds_to_save,
                           args.save_only_wrong_preds)

    checkpoint_cb = ModelCheckpoint(
        monitor='val/R@1',
        filename='_epoch({epoch:02d})_R@1[{val/R@1:.4f}]_R@5[{val/R@5:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=False,
        save_top_k=1,
        save_last=True,
        mode='max'
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/", version=args.exp_name)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],
        default_root_dir='./logs',
        num_sanity_val_steps=0,
        precision=16,
        max_epochs=10,  # Set this to 5 for 5 epochs
        check_val_every_n_epoch=1,
        logger=tb_logger,
        callbacks=[checkpoint_cb],
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=20,
    )
    
    checkpoint = torch.load(join(trainer.default_root_dir, 'lightning_logs', args.exp_name, 'checkpoints', 'last.ckpt'))
    model.load_model_state_dict(checkpoint['state_dict'])

    trainer.test(model=model, dataloaders=test_loader, ckpt_path=None)

