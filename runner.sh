
lr=0.005
epoch=140
BATCH_SIZE=1
class_0=0.3
Alpha=0.0015
for class_1 in 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.2
do
        for test_index in 1 2 3 4 5 6 7 8 9 10 11 12 13 14
        do
            export test_index lr epoch BATCH_SIZE class_0 class_1 Alpha
            sbatch -c 4 wrapper.sh
        done
done


