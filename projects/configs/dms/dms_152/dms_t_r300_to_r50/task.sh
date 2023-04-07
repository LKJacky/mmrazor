export EXP_NAME=dms_t_r152_to_r50

export JOB_NAME=${EXP_NAME}/dms_t_r300_prune_r50_cls
export PTH_NAME=epoch_10
bash ./tools/dist_train.sh projects/configs/dms/dms_152/dms_t_r300_to_r50/dms_t_r300_prune_r50_cls.py 8 --work-dir work_dirs/${JOB_NAME}
bash ./tools/dist_train.sh projects/configs/dms/dms_152/dms_t_r300_to_r50/dms_t_r300_finetune_cls.py 8 --work-dir work_dirs/${JOB_NAME}_finetune
