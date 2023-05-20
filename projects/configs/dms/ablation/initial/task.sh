export EXP_NAME=dms_ablation_init
# export CPUS_PER_TASK=20
export SRUN_ARGS="--quotatype=auto"

# export GPUS=2
# export GPUS_PER_NODE=2


export JOB_NAME_=${EXP_NAME}/dms_r50_prune_e10_pre
export PTH_NAME=epoch_5
sh ./tools/slurm_train.sh eval test projects/configs/dms/ablation/initial/dms_r50_prune_e10_pre.py work_dirs/${JOB_NAME_}
sh ./tools/slurm_train.sh eval test projects/configs/dms/ablation/dms_r50_finetune_reset.py work_dirs/${JOB_NAME_}_finetune_reset

sh ./tools/slurm_train.sh eval test projects/configs/dms/ablation/dms_r50_finetune.py work_dirs/${JOB_NAME_}_finetune
