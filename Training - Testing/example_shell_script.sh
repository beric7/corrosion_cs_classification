#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1
#SBATCH -t 6:00:00
#SBATCH -p v100_normal_q
#SBATCH -A infraeval
module load gcc cmake
module load cuda/9.0.176 
module load cudnn/7.1
module load Anaconda
source activate TF2

cd $PBS_O_WORKDIR
cd ~/COCO-Bridge-2020/MODELS/segmentation_corrosion/deeplabV3plus/

python main_plus.py -data_directory './DATA/augmented/' \
-exp_directory './stored_weights/var_reviewed_augmented_batch_2_l1/' \
--epochs 20 --batch 2 --loss 'l1' \
--pretrained './stored_weights/var_reviewed_augmented_batch_2_l1/weights_6.pt' \
--epoch_number 6

exit
