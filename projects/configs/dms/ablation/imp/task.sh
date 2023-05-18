export EXP_NAME=dms_ablation_epoch
export CPUS_PER_TASK=20

export JOB_NAME=${EXP_NAME}/dms_r50_prune_e1
export PTH_NAME=epoch_5
sh ./tools/slurm_train.sh eval test projects/configs/dms/ablation/imp/dms_r50_prune_e5_index.py work_dirs/${JOB_NAME}
sh ./tools/slurm_train.sh eval test projects/configs/dms/ablation/dms_r50_finetune.py work_dirs/${JOB_NAME}_finetune
