export EXP_NAME=dms_ablation

export JOB_NAME_=${EXP_NAME}/dms_r50_prune
export PTH_NAME=epoch_10
bash ./tools/dist_train.sh projects/configs/dms/ablation/dms_r50_prune.py 8 --work-dir work_dirs/${JOB_NAME_}
bash ./tools/dist_train.sh projects/configs/dms/ablation/dms_r50_finetune.py 8 --work-dir work_dirs/${JOB_NAME_}_finetune
