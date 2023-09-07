# Sample run in cmd line
# > python .\evaluate_unet.py -ground_truth_dir .\data\crack_uas\crack_uas_shc\tiled_ds0\val\masks-png\ 
#                           -pred_dir '.\experiments\tcuas-shc\unet\vgg16-inference (0.5)\pred-png' 
#                           -threshold 0.75 
#                           -img_filter uas

from pathlib import Path
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
# from torchmetrics import IoU, F1
from torchmetrics import JaccardIndex, F1Score
from sklearn.metrics import confusion_matrix, classification_report
import os

def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def general_dice(y_true, y_pred):
    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    return dice(y_true, y_pred)

def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)

def general_jaccard(y_true, y_pred):
    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    return jaccard(y_true, y_pred)

def perf_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    #print(f"CF: {tn, fp, fn, tp}")
    iou_val = tp / (tp+fp+fn + 1e-15)
    
    if tp == 0 or fp == 0 or fn == 0:
        return iou_val, 1, 1, 1
    elif tp == 0 and (fp>0 or fn>0):
        return iou_val, 0, 0, 0
    
    precision = tp / (tp + fp + 1e-15)
    recall = tp / (tp + fn + 1e-15)
    f1 = (2*precision*recall) / (precision + recall)
    return iou_val, precision, recall, f1

def miou(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    #print(f"CF: {tn, fp, fn, tp}")
    miou = tp / (fn+fp-fn + 1e-15)
    return miou

def pixel_accuracy(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    #print(f"CF: {tn, fp, fn, tp}")
    pix_acc= (tp + tn) / (tp+tn+fp+fn + 1e-15)
    return pix_acc


#########

def iou_metric(y_true, y_pred):
    iou = JaccardIndex(num_classes=2)
    result = iou(y_pred, y_true)
    return result

def F1_metric(y_true, y_pred):
    f1 = F1Score(num_classes=2, mdmc_average='samplewise')
    result = f1(y_pred, y_true)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('-ground_truth_dir', type=str,  required=True, help='path where ground truth images are located')
    arg('-pred_dir', type=str, required=True,  help='path with predictions')
    arg('-threshold', type=float, default=0.2, required=False,  help='crack threshold detection')
    arg('-img_filter', type=str, default='', required=False,  help='filter images for glob')

    args = parser.parse_args()

    result_dice = []
    result_jaccard = []
    result_iou = []
    result_miou = []
    result_pixelacc = []

    result_precision = []
    result_recall = []
    result_f1 = []
    count = 0

    y_true_full = []
    y_pred_full = []
    
    if args.img_filter != '':
        paths = [path for path in  Path(args.ground_truth_dir).glob('*' + args.img_filter + '*')]
        print('There are {} images for inference.'.format(len(paths)))
    else: 
        paths = [path for path in  Path(args.ground_truth_dir).glob('*.PNG')]

    for file_name in paths:
        #print(f"filename: {file_name}")
        count += 1

    print(f"No. of images evaluated: {count}")

    for file_name in tqdm(paths):        
        y_true = (cv2.imread(str(file_name), 0) > 0).astype(np.uint8)

        pred_file_name = Path(args.pred_dir) / file_name.name
        if not pred_file_name.exists():
            print(f'missing prediction for file {file_name.name}')
            continue

        pred_image = (cv2.imread(str(pred_file_name), 0) > 255 * args.threshold).astype(np.uint8)
        y_pred = pred_image

        #print(f"y_true shape: {y_true.shape}")
        #print(f"y_pred shape: {y_true.shape}")
        #print(f"filename: {file_name}")

        # print(y_true.max(), y_true.min())
        # plt.subplot(131)
        # plt.imshow(y_true)
        # plt.subplot(132)
        # plt.imshow(y_pred)
        # plt.subplot(133)
        # plt.imshow(y_true)
        # plt.imshow(y_pred, alpha=0.5)
        # plt.show()

        result_dice += [dice(y_true, y_pred)]
        result_jaccard += [jaccard(y_true, y_pred)]
       
        i, p ,r, f1 = perf_metrics(y_true.flatten(), y_pred.flatten())

        result_iou += [i]
        result_precision += [p]
        result_recall += [r]
        result_f1 += [f1]


        #result_miou += [miou(y_true.flatten(), y_pred.flatten())]
        #result_pixelacc += [pixel_accuracy(y_true.flatten(), y_pred.flatten())]
        
        y_true_full.append(y_true.flatten())
        y_pred_full.append(y_pred.flatten())
 
    ### For debugging
    #pred_filename = paths[count-1]
    #if not pred_filename.exists():
    #    print(f'missing prediction for file {pred_filename}')
    #print(pred_filename)
    #pred_image = (cv2.imread(str(pred_filename), 0) > 255 * args.threshold).astype(np.uint8)
    
    #for row in pred_image:
    #    print(row)
    ###

    y_true_full = np.concatenate(y_true_full).ravel()
    y_pred_full = np.concatenate(y_pred_full).ravel()

    #print(f"Size of true: {y_true_full.size}")
    #print(f"Size of pred: {y_pred_full.size}")
    #target_names = ['noncrack', 'crack']
    #print(classification_report(y_true_full, y_pred_full, target_names))
    
    print('Dice = ', np.mean(result_dice), np.std(result_dice))
    print('Jaccard = ', np.mean(result_jaccard), np.std(result_jaccard))
    print('IOU = ', np.mean(result_iou), np.std(result_iou))
    print('Precision = ', np.mean(result_precision), np.std(result_precision))
    print('Recall = ', np.mean(result_recall), np.std(result_recall))
    print('F1 = ', np.mean(result_f1), np.std(result_f1))


    #print('MIOU = ', np.mean(result_miou)/3.0)
    #print('Pixel Accuracy = ', np.mean(result_pixelacc), np.std(result_pixelacc))
