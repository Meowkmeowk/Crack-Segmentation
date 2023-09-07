# Sample run in cmd line
# > python .\train_segnet.py -train_data_dir ./data/crack_uas/crack_uas_shc/tiled_ds0/train/ 
#                            -valid_data_dir ./data/crack_uas/crack_uas_shc/tiled_ds0/val/ 
#                            -num_epochs 60 -batch_size 4 -lr_decay_factor 0.5 -lr_decay_freq 5 
#                            -ckpt_save_dir "checkpoints/training on crack_uas_shc/fpn/" 
#                            -seg_model unet -encoder vgg16 

import argparse, gc, os, pickle, shutil, tqdm
from pathlib import Path
from typing import Union

import numpy as np
import scipy.ndimage as ndimage

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from segmentation_models_pytorch import Linknet, FPN, MAnet, UnetPlusPlus, DeepLabV3, DeepLabV3Plus, Unet 

import wandb

from data_loader import ImgDataSet
from unet.unet_transfer import UNet16, UNetResNet

# Optimizer
from pytorch_ranger import Ranger



os.environ["NVML_DLL_PATH"] = "C:\\Windows\\System32\\nvml.dll"


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val:float, n:int=1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def create_model(device:str, unet_type:str="vgg16") -> Union[UNet16,UNetResNet]:
    if unet_type.lower() == 'vgg16':
        print("Initializing a VGG16-based model...", end="")
        model = UNet16(pretrained=True)
    elif unet_type.lower() == 'resnet101':
        encoder_depth = 101
        num_classes = 1
        print("Initializing a ResNet101-based model...", end="")
        model = UNetResNet(encoder_depth=encoder_depth, num_classes=num_classes, pretrained=True)
        del encoder_depth, num_classes
    elif unet_type.lower() == 'resnet34':
        encoder_depth = 34
        num_classes = 1
        print("Initializing a ResNet34-based model...", end="")
        model = UNetResNet(encoder_depth=encoder_depth, num_classes=num_classes, pretrained=True)
        del encoder_depth, num_classes
    else:
        assert False
    print("done.")

    model.eval()
    return model.to(device)

def create_seg_model(device:str, seg_model: str, encoder: str):

    # U-Net
    if seg_model.lower() == 'unet':
        if encoder.lower() == 'vgg16':
            print("Initializing a VGG16-based model...", end="")
            model = UNet16(pretrained=True)
        elif encoder.lower() == 'resnet101':
            encoder_depth = 101
            num_classes = 1
            print("Initializing a ResNet101-based model...", end="")
            model = UNetResNet(encoder_depth=encoder_depth, num_classes=num_classes, pretrained=True)
            del encoder_depth, num_classes
        elif encoder.lower() == 'resnet34':
            encoder_depth = 34
            num_classes = 1
            print("Initializing a ResNet34-based model...", end="")
            model = UNetResNet(encoder_depth=encoder_depth, num_classes=num_classes, pretrained=True)
            del encoder_depth, num_classes
        else:
            assert False

    elif seg_model.lower() == 'linknet':
        num_classes = 1
        print("Initializing a LinkNet-based model...")
        model = Linknet(encoder_name=encoder, encoder_weights='imagenet', classes=num_classes)

    elif seg_model.lower() == 'fpn':
        num_classes = 1
        print("Initializing a FPN-based model...")
        model = FPN(encoder_name=encoder, encoder_weights='imagenet', classes=num_classes)

    elif seg_model.lower() == 'manet':
        num_classes = 1
        print("Initializing a MAnet-based model...")
        model = MAnet(encoder_name=encoder, encoder_weights='imagenet', classes=num_classes)

    elif seg_model.lower() == 'unet++':
        num_classes = 1
        print("Initializing a Unet++ based model...")
        model = UnetPlusPlus(encoder_name=encoder, encoder_weights='imagenet', classes=num_classes)

    elif seg_model.lower() == 'deeplabv3':
        num_classes = 1
        print("Initializing a DeepLabV3-based model...")
        model = DeepLabV3(encoder_name=encoder, encoder_weights='imagenet', classes=num_classes)

    elif seg_model.lower() == 'deeplabv3plus':
        num_classes = 1
        print("Initializing a DeepLabV3+ based model...")
        model = DeepLabV3Plus(encoder_name=encoder, encoder_weights='imagenet', classes=num_classes)
    
    else:
            assert False
    

    print("done.")

    model.eval()
    return model.to(device)






def create_seg_model_smp(device:str, args:argparse.Namespace):
    # U-Net
    if args.seg_model.lower() == 'fpn':
         print(f"Initializing a '{args.seg_model}' architecture. encoder: {args.encoder}.", end="")
         model =  FPN(
                        encoder_name=args.encoder, 
                        encoder_weights=args.encoder_weights,
                        encoder_depth=args.encoder_depth,
                        in_channels=args.in_channels,
                        classes=args.classes,
                        # decoder_use_batchnorm=args.decoder_use_batchnorm,
                    )
    else:
        assert False
    print("done.")
    model.eval()
    return model.to(device)






def adjust_learning_rate(optimizer:torch.optim.Optimizer, epoch:int, lr:float, decay_factor:float=0.10, decay_freq:int=10) -> None:
    """
    Decays the learning rate by a factor of 0.10 every `decay_freq` epochs.
    """
    lr = lr * (decay_factor ** (epoch // decay_freq))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
   
def validate(model:Union[UNet16,UNetResNet], valid_dataloader:torch.utils.data.DataLoader, criterion:torch.nn.BCEWithLogitsLoss) -> float:
    losses = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(valid_dataloader):
            input_var = Variable(input).cuda()
            target_var = Variable(target).cuda()

            output = model(input_var)
            loss = criterion(output, target_var)

            losses.update(loss.item(), input_var.size(0))

    return losses.avg

def train(model, train_dataloader:torch.utils.data.DataLoader, valid_dataloader:torch.utils.data.DataLoader, criterion:torch.nn.BCEWithLogitsLoss, optimizer:torch.optim.Optimizer, args:argparse.Namespace) -> None:
    # Log model and hyperparameters
    wandb.watch(model, criterion, log="all", log_freq=10)
   
    # Initialize model parameters, optimizer state, and other logs
    if args.resume_from_ckpt != "":     # Resume training
        if not Path(args.resume_from_ckpt).exists():
            raise IOError(f"Checkpoint file '{args.resume_from_ckpt}' does not exist.")
        else:
            checkpoint = torch.load(args.resume_from_ckpt)            
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            starting_epoch = checkpoint["epoch"] + 1
            least_valid_loss_so_far = checkpoint["least validation loss so far"]
            
            del checkpoint
            print(f"Restored states from '{args.resume_from_ckpt}'. Least validation loss so far: {least_valid_loss_so_far}.")
            print(f"Training resumed from epoch {starting_epoch}.")
    elif args.start_from_ckpt != "":    # Warm-start model for training
        if not Path(args.start_from_ckpt).exists():
            raise IOError(f"Checkpoint file '{args.start_from_ckpt}' does not exist.")
        else:
            checkpoint = torch.load(args.start_from_ckpt)            
            model.load_state_dict(checkpoint["model"])
            starting_epoch = 1
            least_valid_loss_so_far = 999
            
            del checkpoint
            print(f"Restored model parameters from '{args.start_from_ckpt}'.")
            print(f"Training from epoch {starting_epoch}.")
    else:                               # Vanilla training
        starting_epoch = 1
        least_valid_loss_so_far = 999
        print(f"Training from epoch {starting_epoch}.")

    #   batch accumulation parameter
    accumulation_steps = args.acc_grad

    # Loop through epochs
    for epoch in range(starting_epoch, args.num_epochs+1):
        adjust_learning_rate(optimizer, epoch, args.init_lr, decay_factor=args.lr_decay_factor, decay_freq=args.lr_decay_freq)

        tq = tqdm.tqdm(total=(len(train_dataloader)*args.batch_size))
        tq.set_description(f"Epoch no. {epoch}")

        train_losses = AverageMeter()
        model.train()
                  
        #   Loop through batches
        for step, (_input_imgs_, _target_masks_) in enumerate(train_dataloader):
            input_imgs = Variable(_input_imgs_).cuda()
            target_masks = Variable(_target_masks_).cuda()

            # ---Forward pass
            pred_masks = model(input_imgs)      
            
            pred_masks_flat = pred_masks.view(-1)
            target_masks_flat = target_masks.view(-1)
           
            # ---Compute loss
            train_loss = criterion(pred_masks_flat, target_masks_flat)
            train_losses.update(train_loss)

            tq.set_postfix(loss=f"{train_losses.avg:.5f}")
            tq.update(args.batch_size)

            # ---Backward pass
            train_loss.backward()
            
            if ((step+1) % accumulation_steps == 0):
                # ---Update parameters
                optimizer.step()

                # ---Reset gradient tensors (Compute gradients)
                optimizer.zero_grad()               
            
        #   Evaluate on validation set
        valid_loss = validate(model, valid_dataloader, criterion)                
        tq.close()
        print(f"   Validation loss at epoch {epoch}: {valid_loss}")

        #   Save model parameters, optimizer state, and logs at the epoch with least validation loss so far
        if valid_loss <= least_valid_loss_so_far:
            least_valid_loss_so_far = valid_loss
            torch.save({
                "epoch" : epoch, "training loss" : train_losses.avg, "validation loss" : valid_loss,
                "least validation loss so far" : least_valid_loss_so_far,
                "model" : model.state_dict(), "optimizer" : optimizer.state_dict()
            }, os.path.join(args.ckpt_save_dir, f"least-valid-loss-epoch.pt"))
        
        # #   Save model parameters, optimizer state, and logs at the current epoch
        # torch.save({
        #     "epoch" : epoch, "training loss" : train_losses.avg, "validation loss" : valid_loss,
        #     "least validation loss so far" : least_valid_loss_so_far,
        #     "model" : model.state_dict(), "optimizer" : optimizer.state_dict()
        # }, os.path.join(args.ckpt_save_dir, f"epoch-{epoch:05d}.pt"))

        learn_rate = optimizer.param_groups[0]['lr']

        #   Log validation loss and training loss for visualization
        wandb.log({'validation loss': valid_loss,
                   'training loss': train_losses.avg,
                    'least validation loss so far': least_valid_loss_so_far,
                   'learning rate': learn_rate,
                   'epoch': epoch, })

        # [EARL] added one more indent
        # Workspace cleanup
        del step, train_losses 
        del _input_imgs_, _target_masks_
        del input_imgs, target_masks, pred_masks
        del pred_masks_flat, target_masks_flat
        del train_loss, valid_loss 
    del starting_epoch, epoch, least_valid_loss_so_far



if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='Training a UNet')
    #   Datasets
    parser.add_argument(
        '-train_data_dir', type=str, required=True,
        help='Directory to the training data. Must contain the subdirectories `images` and `masks`.'
    )
    parser.add_argument(
        '-valid_split', type=float, default=0.15, required=False,
        help="Fraction (exclusively between 0 and 1) of the specified training dataset to be treated as the validation set. This is ignored if `-valid_data_dir` is specified."
    )
    parser.add_argument(
        '-valid_data_dir', type=str, default="", required=False,
        help='Directory to the training data. Must contain the subdirectories `images` and `masks`. If specified, this supercedes splitting a fraction from the training set.'
    )
    #   Training duration, loading, optimization, and regularization
    parser.add_argument(
        '-num_epochs', default=10, type=int,
        help='Number of epochs to run.'
    )
    parser.add_argument(
        '-batch_size',  default=4, type=int,
        help='Batch size for the training and validation sets.'
    )
    parser.add_argument(
        '-init_lr', default=0.001, type=float,
        help='Initial learning rate for the optimizer.'
    )
    parser.add_argument(
        "-lr_decay_factor", default=0.10, type=float,
        help="Factor by which the learning rate is decayed."
    )
    parser.add_argument(
        "-lr_decay_freq", default=10, type=int,
        help="No. of epochs after which the learning rate is decayed."
    )
    parser.add_argument(
        '-momentum', default=0.9, type=float,
        help='Momentum for the optimizer.'
    )
    parser.add_argument(
        '-weight_decay', default=1e-4, type=float,
        help='Weight decay factor.'
    )
    parser.add_argument(
        "-start_from_ckpt", default="", type=str,
        help="Checkpoint from which the model parameters are initialized at the start of training."
    )
    parser.add_argument(
        "-resume_from_ckpt", default="", type=str,
        help="Checkpoint from which the model parameters, the optimizer state, the epoch count, and other logs are initialized at the start of training."
    )
    #   Output directory and UNet type
    parser.add_argument(
        '-ckpt_save_dir', type=str, default="checkpoints/",
        help='Directory to which `.pt` files are saved.'
    )
    parser.add_argument(
        '-seg_model', type=str, required=True, default='unet', choices=['unet', 'linknet', 'fpn', 'manet', 'unet++', 'deeplabv3', 'deeplabv3plus'],
        help="Type of segmentation model to use. Currently supported values are 'unet', 'linknet', 'fpn', 'manet', 'unet++', 'deeplabv3', 'deeplabv3plus'."
    )
    parser.add_argument(
        '-encoder', type=str, required=True, default='vgg16', 
        # choices=['vgg16', 'resnet18', 'resnet34', 'resnet101', 'xception', 'mobilenet_v2'],
        help="Type of encoder to use. Currently supported values are 'vgg16', 'resnet18', 'resnet34', 'resnet101', 'xception', and 'mobilenet_v2'."
    )
    parser.add_argument(
        '-encoder_weights', type=str, required=True, default='imagenet', 
        choices=['imagenet', 'ssl', 'swsl', 'instagram', 'imagenet+background', 'advprop', 'noisy-student', 'imagenet+5k'],
        help="Pretrained weights for the model. Find documentation at https://github.com/qubvel/segmentation_models.pytorch"
    )
    parser.add_argument(
        '-encoder_depth', type=int, required=True, default='5',
        help="Number of stages used in decoder. Larger depth means more features are generated."
    )
    parser.add_argument(
        '-in_channels', type=int, required=True, default='3',
        help="Number of input channels for model."
    )
    parser.add_argument(
        '-classes', type=int, required=True, default='1',
        help="Number of classes for output."
    )
    parser.add_argument(
        '-decoder_use_batchnorm', type=str, default="True", choices=["True", "False", "'inplace'"],
        help='If True, BatchNormalisation layer between Conv2D and Activation layers is used. If inplace InplaceABN will be used, allows to decrease memory consumption.'
    )
    #   Stuff I don't quite understand, and random seed
    parser.add_argument(
        '-num_workers', default=4, type=int,
        help='[IDK] Number of workers.'
    )
    parser.add_argument(
        '-print_freq', default=10, type=int,
        help='[IDK] Print frequency'
    )
    parser.add_argument(
        '-rand_seed', type=int, default=42, required=False,
        help="Seed for random number generators. Currently only used in splitting the dataset into training and validation subsets."
    )
    parser.add_argument(
        '-wandb_run_id', type=str, default="", required=False,
        help="The wandb run ID to resume runs."
    )
    parser.add_argument(
        '-acc_grad', type=int, default="1", required=False,
        help="Accumulated gradient to increase batch size. (global batch size = acc_grad * batch_size)"
    )
    #parser.argument(
    # 'dropout', type=str, default="1", required=False
    # help="dropout thingyyy"
    # )
    #parser.argument(
    # 'activation', type=str, default="sigmoid", choices=["softmax", 'ReLU]
    # help="activation thingyyyy"
    # )
    #parser.argument(
    # 'pooling', type=str, default="max",  choices=[],
    # help="poolinggggggg thingyyy"
    # )

    args = parser.parse_args()



    # Prepare dataset and dataloaders
    #   Preprocessing pipelines
    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]

    #CMYK
    # channel_means = [0.010415009440811905, 0.000000000000000888178419700125232, 0.05753630442638602, 0.4932481167415399]
    # channel_stds  = []


    train_img_xfms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(channel_means, channel_stds)])
    train_mask_xfms = transforms.Compose([transforms.Grayscale(1),
                                        transforms.ToTensor()])
    valid_img_xfms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(channel_means, channel_stds)])
    valid_mask_xfms = transforms.Compose([transforms.Grayscale(1),
                                        transforms.ToTensor()])

    # If there is no test or validation dataset, split from training dataset
    if args.valid_data_dir == "":
        if (args.valid_split <= 0.0) or (args.valid_split >= 1.0):
            raise ValueError("The argument '-valid_split' should be exclusively between 0 and 1.")
        else:
            imgs_dir = os.path.join(args.train_data_dir, 'images')
            masks_dir = os.path.join(args.train_data_dir, 'masks')

            imgs_file_paths = [path.name for path in Path(imgs_dir).glob('*') if path.suffix in ['.jpg', '.png']]
            mask_file_paths = [path.name for path in Path(masks_dir).glob('*') if path.suffix in ['.jpg', '.png']]

            train_valid_dataset = ImgDataSet(
                img_dir=imgs_dir, img_fnames=imgs_file_paths, img_transform=train_img_xfms,
                mask_dir=masks_dir, mask_fnames=mask_file_paths, mask_transform=train_mask_xfms
            )

            valid_data_size = int(len(train_valid_dataset) * args.valid_split)
            train_data_size = len(train_valid_dataset) - valid_data_size

            train_dataset, valid_dataset = random_split(
                train_valid_dataset, [train_data_size, valid_data_size],
                generator=torch.Generator().manual_seed(args.rand_seed)
            )
            
            del train_valid_dataset
    else:
        # ---Training set---
        imgs_dir = os.path.join(args.train_data_dir, 'images')
        masks_dir = os.path.join(args.train_data_dir, 'masks')
        
        imgs_file_paths = [path.name for path in Path(imgs_dir).glob('*') if path.suffix in ['.jpg', '.png']]
        mask_file_paths = [path.name for path in Path(masks_dir).glob('*') if path.suffix in ['.jpg', '.png']]
        
        # Convert train_dataset to map-style dataset
        train_dataset = ImgDataSet(
            img_dir=imgs_dir, img_fnames=imgs_file_paths, img_transform=train_img_xfms,
            mask_dir=masks_dir, mask_fnames=mask_file_paths, mask_transform=train_mask_xfms
        )
        train_data_size = int(len(train_dataset))

        # ---Validation set---
        imgs_dir = os.path.join(args.valid_data_dir, 'images')
        masks_dir = os.path.join(args.valid_data_dir, 'masks')
        
        imgs_file_paths = [path.name for path in Path(imgs_dir).glob('*') if path.suffix in ['.jpg', '.png']]
        mask_file_paths = [path.name for path in Path(masks_dir).glob('*') if path.suffix in ['.jpg', '.png']]
        
        # Convert valid_dataset to map-style dataset
        valid_dataset = ImgDataSet(
            img_dir=imgs_dir, img_fnames=imgs_file_paths, img_transform=valid_img_xfms,
            mask_dir=masks_dir, mask_fnames=mask_file_paths, mask_transform=valid_mask_xfms
        )
        valid_data_size = int(len(valid_dataset))
    
    print(f"Training on {train_data_size} images. Validating on {valid_data_size} images.")

    #   Data loaders
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers) #, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers) #, drop_last=True)
    
    print(f"Len of train dataloader: {len(train_dataloader)} Len of train ds: {train_data_size}")
    print(f"Len of valid dataloader: {len(valid_dataloader)} Len of valid ds: {valid_data_size}")
    
    #   Workspace cleanup
    del channel_means, channel_stds
    del train_img_xfms, train_mask_xfms
    del valid_img_xfms, valid_mask_xfms
    del imgs_dir, masks_dir
    del imgs_file_paths, mask_file_paths
    del train_data_size, valid_data_size
    del train_dataset, valid_dataset

    # Initialize model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model = create_model(device, args.unet_type)
    model = create_seg_model_smp(device, args)
    model.cuda()

    del device

    # Initialize optimizer
    optimizer = Ranger(model.parameters(), args.init_lr,
                                # momentum=args.momentum,
                                weight_decay=args.weight_decay)

    criterion = nn.BCEWithLogitsLoss().to("cuda")



    # Create save directory
    os.makedirs(args.ckpt_save_dir, exist_ok=True)

    # Get wandb run ID
    if args.wandb_run_id != "":
        id = args.wandb_run_id
    else:
        id = wandb.util.generate_id()

    # Get config parameters of current run for wandb logging
    hyperparameters = dict(
        wandb_run_id = id,
        dataset = "crack_uas_shc",
        optimizer = type(optimizer).__name__,
        loss_criterion = type(criterion).__name__,
        architecture = args.seg_model, 
        unet_type = args.encoder,
        encoder_weights  = args.encoder_weights,
        encoder_depth = args.encoder_depth,
        in_channels = args.in_channels,
        classes = args.classes,
        decoder_use_batchnorm=args.decoder_use_batchnorm,
        resume_from_ckpt = args.resume_from_ckpt,
        epochs = args.num_epochs,
        batch_size = args.batch_size,
        accumulated_grad = args.acc_grad,
        num_workers = args.num_workers,
        initial_learning_rate = args.init_lr,
        lr_decay_factor = args.lr_decay_factor,
        lr_decay_freq = args.lr_decay_freq,
        momentum = args.momentum,
        weight_decay = args.weight_decay,
        random_seed = args.rand_seed,
        #dropout
        #pooling
        #activation

    )
    

    #   Initialize wandb
    with wandb.init(project="gridsearchUnet", id=id, resume="allow", config=hyperparameters):
        config = wandb.config

        # Commence training
        print(f"Commence training...")
        train(model, train_dataloader, valid_dataloader, criterion, optimizer, args)
    
    # Workspace cleanup and garbage collection
    del parser, args
    del model, optimizer, criterion
    del train_dataloader, valid_dataloader
    del hyperparameters, config, id
    
    gc.collect()
