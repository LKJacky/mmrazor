export EXP_NAME=dms_ablation_epoch

export JOB_NAME=${EXP_NAME}/dms_r50_prune_e1
export PTH_NAME=epoch_1
bash ./tools/dist_train.sh projects/configs/dms/ablation/epochs/dms_r50_prune_e1.py 8 --work-dir work_dirs/${JOB_NAME}
bash ./tools/dist_train.sh projects/configs/dms/ablation/dms_r50_finetune.py 8 --work-dir work_dirs/${JOB_NAME}_finetune

export JOB_NAME=${EXP_NAME}/dms_r50_prune_e3
export PTH_NAME=epoch_3
bash ./tools/dist_train.sh projects/configs/dms/ablation/epochs/dms_r50_prune_e3.py 8 --work-dir work_dirs/${JOB_NAME}
bash ./tools/dist_train.sh projects/configs/dms/ablation/dms_r50_finetune.py 8 --work-dir work_dirs/${JOB_NAME}_finetune

export JOB_NAME=${EXP_NAME}/dms_r50_prune_e5
export PTH_NAME=epoch_5
bash ./tools/dist_train.sh projects/configs/dms/ablation/epochs/dms_r50_prune_e5.py 8 --work-dir work_dirs/${JOB_NAME}
bash ./tools/dist_train.sh projects/configs/dms/ablation/dms_r50_finetune.py 8 --work-dir work_dirs/${JOB_NAME}_finetune

export JOB_NAME=${EXP_NAME}/dms_r50_prune_e10
export PTH_NAME=epoch_10
bash ./tools/dist_train.sh projects/configs/dms/ablation/epochs/dms_r50_prune_e10.py 8 --work-dir work_dirs/${JOB_NAME}
bash ./tools/dist_train.sh projects/configs/dms/ablation/dms_r50_finetune.py 8 --work-dir work_dirs/${JOB_NAME}_finetune
