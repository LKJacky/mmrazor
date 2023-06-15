export EXP_NAME=dms_ablation_epoch_dimension
export CPUS_PER_TASK=15

export JOB_NAME_=${EXP_NAME}/dms_r50_prune_e10_e_bf
export PTH_NAME=epoch_10
sh ./tools/slurm_train.sh eval test projects/configs/dms/ablation/dimension/dms_r50_prune_e10_e_bf.py work_dirs/${JOB_NAME_}
sh ./tools/slurm_train.sh eval test projects/configs/dms/ablation/dms_r50_finetune.py work_dirs/${JOB_NAME_}_finetune
