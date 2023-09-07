import subprocess
# import requests 

# # Define a new session object
# session = requests.Session()

# # Use the session object to send a GET request
# response = session.get("https://python.org", ssl=False)

# import requests
# from requests.packages.urllib3.exceptions import InsecureRequestWarning

# # Disable SSL verification warnings
# requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# # Send a GET request with SSL verification disabled
# response = requests.get("https://python.org", verify=False)

# # Print the response content
# print(response.content)

# import requests

# # Send a GET request with SSL verification disabled
# response = requests.get("https://python.org", verify=False)

# Print the response content
# print(response.content)




subprocess.run([
    "python", "train_segnet.py",
    "-train_data_dir", "data/crack_uas/crack_uas_shc/tiled_ds0/train/",
    "-valid_data_dir", "data/crack_uas/crack_uas_shc/tiled_ds0/val/",
    
    
    # Update from time to time
    "-seg_model", "fpn",   
    "-encoder", "vgg16",   
    "-encoder_weights", "imagenet", 
    "-encoder_depth", "5",
    "-in_channels", "3",
    "-classes", "1",
    "-ckpt_save_dir", "checkpoints/test",    
    
+

    # if resume_from_ckpt, add current epoch to num_epochs
    # "-start_from_ckpt", "checkpoints/training on crack_uas_shc_hyper/manet/mit_b3/least-valid-loss-epoch.pt",   # warm-starting model in initialization, starts at epoch 1 with pre-trained weights
    # "-resume_from_ckpt", "checkpoints/training on crack_uas_shc_hyper/corrected/fpn/vgg16/least-valid-loss-epoch.pt",   # resume training where epoch number will be continued
    "-num_epochs", "60",


    # parameters with GPU consideration
    "-batch_size", "4",
    "-acc_grad", "1",  # need to replace 16gb ram when 4
    "-num_workers", "2",


    # hyperparameters (comment out to modify, otherwise default values)
    # "-init_lr", "0.001", # 0.001
    # "-lr_decay_factor", "0.5", # 0.10
    # "-lr_decay_freq", "5", # 10
    # "-momentum", "0.9", # 0.9
    # "-weight_decay", "0.0001", # 1e-4/0.0004
    # "-random_seed", "42", #42

    # "http.proxyStrictSSL", False

])


# spm 
# https://github.com/qubvel/segmentation_models.pytorch
# https://segmentation-modelspytorch.readthedocs.io/en/latest/


# hyperparameter records
# https://docs.google.com/spreadsheets/d/1AtYHGLWbW9Jk25a7ftAdhDMT-8iLc1oRwVa2iA72lzo/edit?usp=sharing

# TO[DOne] resume_from_ckpt at epoch 5 of mit_b3 [done]
# TO[DOne] tried Ranger, instead of SGD

# TODO after training, evaluate and reflect results to wandb
# or TODO wandb.log precision, recall, etc. ; go to train_segnet.py @279 for logging in wandb 

# TODO in GColab, save only the least-valid-loss-epoch.pt instead of entire pt files 

# TODO to obtain training stability and converge of the model, the whitening and contrast normalization
# can be used. It can be done by computing the channel_means and channel_stds of the dataset prior to training.