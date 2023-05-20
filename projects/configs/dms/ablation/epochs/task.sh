export EXP_NAME=dms_ablation_epoch
# export CPUS_PER_TASK=20
export SRUN_ARGS="--quotatype=auto"

# export GPUS=2
# export GPUS_PER_NODE=2

export JOB_NAME=${EXP_NAME}/dms_r50_prune_e1
export PTH_NAME=epoch_1
sh ./tools/slurm_train.sh eval test projects/configs/dms/ablation/epochs/dms_r50_prune_e1.py work_dirs/${JOB_NAME}
sh ./tools/slurm_train.sh eval test projects/configs/dms/ablation/dms_r50_finetune.py work_dirs/${JOB_NAME}_finetune

export JOB_NAME=${EXP_NAME}/dms_r50_prune_e3
export PTH_NAME=epoch_3
sh ./tools/slurm_train.sh eval test projects/configs/dms/ablation/epochs/dms_r50_prune_e3.py work_dirs/${JOB_NAME}
sh ./tools/slurm_train.sh eval test projects/configs/dms/ablation/dms_r50_finetune.py work_dirs/${JOB_NAME}_finetune

export JOB_NAME=${EXP_NAME}/dms_r50_prune_e5
export PTH_NAME=epoch_5
sh ./tools/slurm_train.sh eval test projects/configs/dms/ablation/epochs/dms_r50_prune_e5.py work_dirs/${JOB_NAME}
sh ./tools/slurm_train.sh eval test projects/configs/dms/ablation/dms_r50_finetune.py work_dirs/${JOB_NAME}_finetune

export JOB_NAME=${EXP_NAME}/dms_r50_prune_e10
export PTH_NAME=epoch_10
sh ./tools/slurm_train.sh eval test projects/configs/dms/ablation/epochs/dms_r50_prune_e10.py work_dirs/${JOB_NAME}
sh ./tools/slurm_train.sh eval test projects/configs/dms/ablation/dms_r50_finetune.py work_dirs/${JOB_NAME}_finetune

export JOB_NAME=${EXP_NAME}/dms_r50_prune_e20
export PTH_NAME=epoch_20
sh ./tools/slurm_train.sh eval test projects/configs/dms/ablation/epochs/dms_r50_prune_e20.py work_dirs/${JOB_NAME}
sh ./tools/slurm_train.sh eval test projects/configs/dms/ablation/dms_r50_finetune.py work_dirs/${JOB_NAME}_finetune

export JOB_NAME=${EXP_NAME}/dms_r50_prune_e30
export PTH_NAME=epoch_30
sh ./tools/slurm_train.sh eval test projects/configs/dms/ablation/epochs/dms_r50_prune_e30.py work_dirs/${JOB_NAME}
sh ./tools/slurm_train.sh eval test projects/configs/dms/ablation/dms_r50_finetune.py work_dirs/${JOB_NAME}_finetune
