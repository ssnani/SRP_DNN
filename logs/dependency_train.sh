RES=$(sbatch train_batch_script.txt)

for i in `seq 1 3`
do
    echo $i, $RES

    RES=$(sbatch --dependency=afterany:${RES##* } train_batch_script.txt)

#RES=$(sbatch --dependency=afterok:${RES##* } b_train.txt)  
done