
export JOB_NAME=dms_t_mnet_prune_31_cls
export PTH_NAME=epoch_30
bash ./tools/dist_train.sh projects/configs/dms/dms_mobilent/dms_t_ment_prune_31_cls.py 8 --work-dir work_dirs/${JOB_NAME}
bash ./tools/dist_train.sh projects/configs/dms/dms_mobilent/dms_t_ment_finetune_31_cls.py 8 --work-dir work_dirs/${JOB_NAME}_finetune
