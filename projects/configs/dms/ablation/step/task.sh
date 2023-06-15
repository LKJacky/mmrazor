export EXP_NAME=dms_ablation_step
export CPUS_PER_TASK=15

export JOB_NAME_=${EXP_NAME}/dms_r50_prune_e10_s200
export PTH_NAME=epoch_10
sh ./tools/slurm_train.sh eval test projects/configs/dms/ablation/step/dms_r50_prune_e10_s200.py work_dirs/${JOB_NAME_}
sh ./tools/slurm_train.sh eval test projects/configs/dms/ablation/dms_r50_finetune.py work_dirs/${JOB_NAME_}_finetune

export JOB_NAME_=${EXP_NAME}/dms_r50_prune_e10_s500
export PTH_NAME=epoch_10
sh ./tools/slurm_train.sh eval test projects/configs/dms/ablation/step/dms_r50_prune_e10_s500.py work_dirs/${JOB_NAME_}
sh ./tools/slurm_train.sh eval test projects/configs/dms/ablation/dms_r50_finetune.py work_dirs/${JOB_NAME_}_finetune

export JOB_NAME_=${EXP_NAME}/dms_r50_prune_e10_s1000
export PTH_NAME=epoch_10
sh ./tools/slurm_train.sh eval test projects/configs/dms/ablation/step/dms_r50_prune_e10_s1000.py work_dirs/${JOB_NAME_}
sh ./tools/slurm_train.sh eval test projects/configs/dms/ablation/dms_r50_finetune.py work_dirs/${JOB_NAME_}_finetune

export JOB_NAME_=${EXP_NAME}/dms_r50_prune_e10_s2500
export PTH_NAME=epoch_10
sh ./tools/slurm_train.sh eval test projects/configs/dms/ablation/step/dms_r50_prune_e10_s2500.py work_dirs/${JOB_NAME_}
sh ./tools/slurm_train.sh eval test projects/configs/dms/ablation/dms_r50_finetune.py work_dirs/${JOB_NAME_}_finetune
