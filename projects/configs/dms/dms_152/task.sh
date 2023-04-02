export JOB_NAME=dms_t_r152_prune_17_cls
export PTH_NAME=epoch_10
bash ./tools/dist_train.sh projects/configs/dms/dms_152/dms_t_r152_prune_17_cls.py 8
bash ./tools/dist_train.sh projects/configs/dms/dms_152/dms_t_r152_finetune_cls.py 8 --work-dir work_dirs/${JOB_NAME}_finetune

export JOB_NAME=dtp_t_r152_prune_17_cls
export PTH_NAME=epoch_10
bash ./tools/dist_train.sh projects/configs/dms/dms_152/dtp_t_r152_prune_17_cls.py 8
bash ./tools/dist_train.sh projects/configs/dms/dms_152/dms_t_r152_finetune_cls.py 8 --work-dir work_dirs/${JOB_NAME}_finetune
