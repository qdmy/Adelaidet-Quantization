#!/bin/bash

# Configure the resources required
#SBATCH -p batch                                                # partition (this is the queue your job will be added to)
#SBATCH -n 1                                                    # number of tasks (sequential job starts 1 task) (check this if your job unexpectedly uses 2 nodes)
#SBATCH -c 8                                                    # number of cores (sequential job calls a multi-thread program that uses 8 cores)
#SBATCH --time=24:00:00                                          # time allocation, which has the format (D-HH:MM), here set to 1 hour
#SBATCH --gres=gpu:4                                           # generic resource required (here requires 1 GPUs)
#SBATCH --mem=128GB                                              # memory pool for all cores (here set to 16 GB)

# Configure notifications 
#SBATCH --mail-type=END                                         # Type of email notifications will be sent (here set to END, which means an email will be sent when the job is done)
#SBATCH --mail-type=FAIL                                        # Type of email notifications will be sent (here set to FAIL, which means an email will be sent when the job is fail to complete)
#SBATCH --mail-user=liujing_95@outlook.com                    # Email to which notification will be sent

# Execute your script (due to sequential nature, please select proper compiler as your script corresponds to)
# bash ./run_ssc-phoenix.sh                                       # bash script used here for demonstration purpose, you should select proper compiler for your needs
python tools/train_net.py --num-gpus 4 --config-file configs/FCOS-COCO-Detection/fcos_R_50_1x-Full-BN_SyncBN_dorefa_clip4bit.yaml --dist-url tcp://127.0.0.1:62211