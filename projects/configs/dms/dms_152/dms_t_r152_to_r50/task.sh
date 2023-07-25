export EXP_NAME=dms_t_r152_to_r50

export JOB_NAME=${EXP_NAME}/dms_t_r152_prune_35_cls
export PTH_NAME=epoch_10
bash ./tools/dist_train.sh projects/configs/dms/dms_152/dms_t_r152_to_r50/dms_t_r152_prune_35_cls.py 8 --work-dir work_dirs/${JOB_NAME}
bash ./tools/dist_train.sh projects/configs/dms/dms_152/dms_t_r152_to_r50/dms_t_r152_finetune_cls.py 8 --work-dir work_dirs/${JOB_NAME}_finetune
bash ./tools/dist_train.sh projects/configs/dms/dms_152/dms_t_r152_to_r50/dms_t_r152_finetune_cls_e600.py 8 --work-dir work_dirs/${JOB_NAME}_finetune_600