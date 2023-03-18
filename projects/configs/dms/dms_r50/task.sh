export JOB_NAME=dms_t_r50_prune_50_cls
export PTH_NAME=epoch_10
bash ./tools/dist_train.sh projects/configs/dms/dms_r50/dms_t_r50_prune_50_cls.py 8
bash ./tools/dist_train.sh projects/configs/dms/dms_r50/dms_t_r50_finetune_50_cls.py 8 --work-dir work_dirs/${JOB_NAME}_finetune
