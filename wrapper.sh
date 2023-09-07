#!/bin/bash -l

#SBATCH
#SBATCH --job-name=GCN
#SBATCH --partition=defq
#SBATCH -N 5
#SBATCH --ntasks-per-node=48
#SBATCH -A nnandak1


#### load and unload modules you may need

ml pytorch
ml python/2.7-anaconda
pip install --user torch
ml cuda/9.2
curr_dir=$PWD
python /home/nnandak1/scratch4-avenka14/Dynamic_EZ_localization/transformer_gcn.py $test_index $lr $epoch $BATCH_SIZE $class_0 $class_1 $Alpha
cd $curr_dir

