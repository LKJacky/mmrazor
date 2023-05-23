export EXP_NAME=dms_ablation_super_size
# export CPUS_PER_TASK=20
export SRUN_ARGS="--quotatype=auto"

export GPUS=2
export GPUS_PER_NODE=2

export JOB_NAME_=${EXP_NAME}/dms_r50_prune_e10_e_R34
export PTH_NAME=epoch_10
sh ./tools/slurm_train.sh eval test projects/configs/dms/ablation/super_size/dms_r50_prune_e10_e_R34.py work_dirs/${JOB_NAME_}
sh ./tools/slurm_train.sh eval test projects/configs/dms/ablation/super_size/dms_r50_prune_e10_e_R34_finetune.py work_dirs/${JOB_NAME_}_finetune


export JOB_NAME_=${EXP_NAME}/dms_r50_prune_e10_e_R101
export PTH_NAME=epoch_10
sh ./tools/slurm_train.sh eval test projects/configs/dms/ablation/super_size/dms_r50_prune_e10_e_R101.py work_dirs/${JOB_NAME_}
sh ./tools/slurm_train.sh eval test projects/configs/dms/ablation/super_size/dms_r50_prune_e10_e_R101_finetune.py work_dirs/${JOB_NAME_}_finetune

export JOB_NAME_=${EXP_NAME}/dms_r50_prune_e10_e_R152
export PTH_NAME=epoch_10
sh ./tools/slurm_train.sh eval test projects/configs/dms/ablation/super_size/dms_r50_prune_e10_e_R152.py work_dirs/${JOB_NAME_}
sh ./tools/slurm_train.sh eval test projects/configs/dms/ablation/super_size/dms_r50_prune_e10_e_R152_finetune.py work_dirs/${JOB_NAME_}_finetune
