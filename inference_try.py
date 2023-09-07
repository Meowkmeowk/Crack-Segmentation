import argparse, gc, os
from os.path import join
from pathlib import Path
from tqdm import tqdm

import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image

import torch
# import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms

# from unet.unet_transfer import UNet16, input_size
# from utils import load_unet_vgg16, load_unet_resnet_101, load_unet_resnet_34, load_linknet_vgg16, load_fpn_vgg16, load_manet_vgg16, load_unetplusplus_vgg16

import segmentation_models_pytorch as smp
import time

start_time = time.time()

os.environ["NVML_DLL_PATH"] = "C:\\Windows\\System32\\nvml.dll"

def create_seg_model_smp(device:str, args:argparse.Namespace):
    model = None
    if args.seg_model.lower() == 'unet':
         model =  smp.Unet(
                        encoder_name=args.encoder, 
                        encoder_weights=args.encoder_weights,
                        encoder_depth=args.encoder_depth,
                        in_channels=args.in_channels,
                        classes=args.classes,
                        decoder_use_batchnorm=args.decoder_use_batchnorm,
                    )
    elif args.seg_model.lower() == 'unet++':
         model =  smp.UnetPlusPlus(
                        encoder_name=args.encoder, 
                        encoder_weights=args.encoder_weights,
                        encoder_depth=args.encoder_depth,
                        in_channels=args.in_channels,
                        classes=args.classes,
                        decoder_use_batchnorm=args.decoder_use_batchnorm,
                    )
    elif args.seg_model.lower() == 'fpn':
         model =  smp.FPN(
                        encoder_name=args.encoder, 
                        encoder_weights=args.encoder_weights,
                        encoder_depth=args.encoder_depth,
                        in_channels=args.in_channels,
                        classes=args.classes,
                        # decoder_use_batchnorm=args.decoder_use_batchnorm,
                    )
    elif args.seg_model.lower() == 'deeplabv3':
         model =  smp.DeepLabV3(
                        encoder_name=args.encoder, 
                        encoder_weights=args.encoder_weights,
                        encoder_depth=args.encoder_depth,
                        in_channels=args.in_channels,
                        classes=args.classes,
                        decoder_use_batchnorm=args.decoder_use_batchnorm,
                    )
    elif args.seg_model.lower() == 'deeplabv3plus':
         model =  smp.DeepLabV3Plus(
                        encoder_name=args.encoder, 
                        encoder_weights=args.encoder_weights,
                        encoder_depth=args.encoder_depth,
                        in_channels=args.in_channels,
                        classes=args.classes,
                        decoder_use_batchnorm=args.decoder_use_batchnorm,
                    )
    elif args.seg_model.lower() == 'linknet':
         model =  smp.Linknet(
                        encoder_name=args.encoder, 
                        encoder_weights=args.encoder_weights,
                        encoder_depth=args.encoder_depth,
                        in_channels=args.in_channels,
                        classes=args.classes,
                        decoder_use_batchnorm=args.decoder_use_batchnorm,
                    )
    elif args.seg_model.lower() == 'manet':
         model =  smp.MAnet(
                        encoder_name=args.encoder, 
                        encoder_weights=args.encoder_weights,
                        encoder_depth=args.encoder_depth,
                        in_channels=args.in_channels,
                        classes=args.classes,
                        decoder_use_batchnorm=args.decoder_use_batchnorm,
                    )
    else:
        assert False
    
    if model is not None:
        print(f"[MODEL] Initializing a '{args.seg_model}' architecture. encoder: {args.encoder}.", end="") 
        if not Path(args.model_path).exists():
            raise IOError(f"Checkpoint file '{args.model_path}' does not exist.")
        else:
            checkpoint = torch.load(args.model_path)            
            model.load_state_dict(checkpoint["model"])
            del checkpoint
            print(f"[MODEL] Restored model parameters from '{args.model_path}'.")
        model.eval()
        return model.to(device)
    else:
        exit()



def evaluate_img(model, img):
    input_size = (448, 448)
    input_width, input_height = input_size[0], input_size[1]

    img_1 = cv.resize(img, (input_width, input_height), cv.INTER_AREA)
    
    X = train_tfms(Image.fromarray(img_1))
    X = Variable(X.unsqueeze(0)).cuda()  # [1, N, H, W]
    
    mask = model(X)

    mask = torch.sigmoid(mask[0, 0]).data.cpu().numpy()
    mask = cv.resize(mask, (img_width, img_height), cv.INTER_AREA)
    return mask

def evaluate_img_patch(model, img):
    input_size = (448, 448)
    input_width, input_height = input_size[0], input_size[1]

    img_height, img_width, img_channels = img.shape

    if img_width < input_width or img_height < input_height:
        return evaluate_img(model, img)

    stride_ratio = 0.1
    stride = int(input_width * stride_ratio)

    normalization_map = np.zeros((img_height, img_width), dtype=np.int16)

    patches = []
    patch_locs = []
    for y in range(0, img_height - input_height + 1, stride):
        for x in range(0, img_width - input_width + 1, stride):
            segment = img[y:y + input_height, x:x + input_width]
            normalization_map[y:y + input_height, x:x + input_width] += 1
            patches.append(segment)
            patch_locs.append((x, y))

    patches = np.array(patches)
    if len(patch_locs) <= 0:
        return None

    preds = []
    for i, patch in enumerate(patches):
        patch_n = train_tfms(Image.fromarray(patch))
        X = Variable(patch_n.unsqueeze(0)).cuda()  # [N, 1, H, W]
        masks_pred = model(X)
        mask = torch.sigmoid(masks_pred[0, 0]).data.cpu().numpy()
        preds.append(mask)

    probability_map = np.zeros((img_height, img_width), dtype=float)
    for i, response in enumerate(preds):
        coords = patch_locs[i]
        probability_map[coords[1]:coords[1] + input_height, coords[0]:coords[0] + input_width] += response

    return probability_map

def disable_axis():
    plt.axis('off')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_ticklabels([])
    plt.gca().axes.get_yaxis().set_ticklabels([])









if __name__ == '__main__':


    matplotlib.use("Agg")   # Fix for "Fail to allocate bitmap"

    parser = argparse.ArgumentParser()
    parser.add_argument('-img_dir',type=str, help='input dataset directory')
    parser.add_argument('-model_path', type=str, help='trained model path')
    parser.add_argument('-seg_model', type=str, choices=['unet', 'linknet', 'fpn', 'manet', 'unet++', 'deeplabv3', 'deeplabv3plus'])
    parser.add_argument('-encoder', type=str, required=True, default='mit_b0')
    parser.add_argument('-encoder_weights', type=str, required=True, default='imagenet')
    parser.add_argument('-encoder_depth', type=int, required=True, default='5')
    parser.add_argument('-in_channels', type=int, required=True, default='3')
    parser.add_argument('-classes', type=int, required=True, default='1')
    parser.add_argument('-decoder_use_batchnorm', type=str, default="True", choices=["True", "False", "'inplace'"])
    parser.add_argument('-out_viz_dir', type=str, default='', required=False, help='visualization output dir')
    parser.add_argument('-out_pred_dir', type=str, default='', required=False,  help='prediction output dir')
    parser.add_argument('-threshold', type=float, default=0.2 , help='threshold to cut off crack response')
    parser.add_argument('-img_filter', type=str, default='', required=False,  help='filter images for glob')
    args = parser.parse_args()


    #cleaning or creating directories and removing files within those directories
    if args.out_viz_dir != '':
        os.makedirs(args.out_viz_dir, exist_ok=True)
        for path in Path(args.out_viz_dir).glob('*.*'):
            os.remove(str(path))

    if args.out_pred_dir != '':
        os.makedirs(args.out_pred_dir, exist_ok=True)
        for path in Path(args.out_pred_dir).glob('*.*'):
            os.remove(str(path))


    # load `trained` model here using smp
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = create_seg_model_smp(device, args)
    
    # channel_means = [0.485, 0.456, 0.406, 0.400]
    # channel_stds  = [0.229, 0.224, 0.225, 0.200]
    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]
    paths = [path for path in Path(args.img_dir).glob('*.jpg')]

    if args.img_filter != '':
        paths = [path for path in  Path(args.img_dir).glob('*' + args.img_filter + '*')]
    else: 
        paths = [path for path in  Path(args.img_dir).glob('*.jpg')]

    count = 0
    latency_values = []
    for file_name in paths:
        # print(f"filename: {file_name}")
        count += 1
    print(f"No. of images for inference: {count}")

    start_time = time.time()

   
    for path in tqdm(paths):
        #print(str(path))
        train_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(channel_means, channel_stds)])

        img_0 = Image.open(str(path))
        img_0 = np.asarray(img_0)
        if len(img_0.shape) != 3:
            print(f"\nImage '{path.name}' has an incorrect shape {img_0.shape}.")
            continue
        infer_start_time = time.time()

        img_0 = img_0[:,:,:3]
        img_height, img_width, img_channels = img_0.shape
        prob_map_full = evaluate_img(model, img_0)

        infer_end_time = time.time()
        latency = infer_end_time - infer_start_time
        latency_values.append(latency)

        if args.out_pred_dir != '':
            cv.imwrite(filename=join(args.out_pred_dir, f'{path.stem}.png'), img=(prob_map_full * 255).astype(np.uint8))

        if args.out_viz_dir != '':
            # plt.subplot(121)
            # plt.imshow(img_0), plt.title(f'{img_0.shape}')
            if img_0.shape[0] > 2000 or img_0.shape[1] > 2000:
                img_1 = cv.resize(img_0, None, fx=0.2, fy=0.2, interpolation=cv.INTER_AREA)
            else:
                img_1 = img_0

            # plt.subplot(122)
            # plt.imshow(img_0), plt.title(f'{img_0.shape}')
            # plt.show()

            prob_map_patch = evaluate_img_patch(model, img_1)

            #plt.title(f'name={path.stem}. \n cut-off threshold = {args.threshold}', fontsize=4)
            prob_map_viz_patch = prob_map_patch.copy()
            prob_map_viz_patch = prob_map_viz_patch/ prob_map_viz_patch.max()
            prob_map_viz_patch[prob_map_viz_patch < args.threshold] = 0.0
            fig = plt.figure()
            st = fig.suptitle(f'name={path.stem} \n cut-off threshold = {args.threshold}', fontsize="x-large")
            ax = fig.add_subplot(231)
            ax.imshow(img_1)
            ax = fig.add_subplot(232)
            ax.imshow(prob_map_viz_patch)
            ax = fig.add_subplot(233)
            ax.imshow(img_1)
            ax.imshow(prob_map_viz_patch, alpha=0.4)

            prob_map_viz_full = prob_map_full.copy()
            prob_map_viz_full[prob_map_viz_full < args.threshold] = 0.0                        
            ax = fig.add_subplot(234)
            ax.imshow(img_0)
            ax = fig.add_subplot(235)
            ax.imshow(prob_map_viz_full)
            ax = fig.add_subplot(236)
            ax.imshow(img_0)
            ax.imshow(prob_map_viz_full, alpha=0.4)

            plt.savefig(join(args.out_viz_dir, f'{path.stem}.png'), dpi=500)
            plt.close('all')
            gc.collect()

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

total_latency = sum(latency_values)
average_latency = total_latency / len(latency_values)
print(f"Average Latency: {average_latency:.4f} seconds")







