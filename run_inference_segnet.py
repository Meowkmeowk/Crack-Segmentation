# import subprocess

# subprocess.run([
#     "python", "inference_se.py",
#     "-img_dir", "data/crack_uas/sample_to_delete_later/tiled_ds0/test/",
#     "-model_path", "checkpoints/training on crack_uas_shc_hyper/unet/mit_b1_w60/least-valid-loss-epoch.pt",   # resume training where epoch number will be continued
#     "-seg_model", "fpn",
#     "-encoder", "mit_b1",   
#     "-encoder_weights", "imagenet",
#     "-encoder_depth", "5",
#     "-in_channels", "3",
#     "-classes", "1",
#     "-decoder_use_batchnorm", "True",
#    "-out_viz_dir", "data/crack_uas/sample_to_delete_later/tiled_ds0/test/experiments/unet/mit_b1/viz",
#     "-out_pred_dir", "data/crack_uas/sample_to_delete_later/tiled_ds0/test/experiments/unet/mit_b1/pred",
#     "-threshold", "0.5",
# ])


import subprocess

subprocess.run([
    "python", "inference_try.py",
    "-img_dir", "data/crack_uas/crack_uas_shc/tiled_ds0/val/images",
    "-model_path", "checkpoints/training on crack_uas_shc_hyper/corrected/unet/mit_b0/least-valid-loss-epoch.pt",   # resume training where epoch number will be continued
    "-seg_model", "unet",
    "-encoder", "mit_b0",   
    "-encoder_weights", "imagenet",
    "-encoder_depth", "5",
    "-in_channels", "3",
    "-classes", "1",
    "-decoder_use_batchnorm", "True",
    # "-out_viz_dir", "data/crack_uas_shc/tiled_ds0/experiments/unet/mit_b3_t075/viz",
    # "-out_pred_dir", "data/crack_uas_shc/tiled_ds0/experiments/unet/mit_b3_t075/pred",
    "-out_pred_dir", "data/latency/UNet_mit_b0",

    "-img_filter", "uas",
    "-threshold", "0.75",
    # "-nsamples", "746",
    
])


