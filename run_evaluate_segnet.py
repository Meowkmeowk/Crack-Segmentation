import subprocess

subprocess.run([
    "python", "evaluate_unet.py",
    "-ground_truth_dir", "data/crack_uas_shc/tiled_ds0/val/masks-png/",
    "-pred_dir", "data/crack_uas_shc/tiled_ds0/experiments/unet/phase2/vgg16_t050/pred",
    "-csv_dir", "data/crack_uas_shc/tiled_ds0/experiments/unet/RGB/vgg16_t050/vgg16.csv",
    "-img_filter", "uas",
    "-threshold", "0.50",
])


