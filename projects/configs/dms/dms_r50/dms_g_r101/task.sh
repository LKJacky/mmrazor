export JOB_NAME=dms_g_t_r101_prune_25_cls
export PTH_NAME=epoch_10
bash ./tools/dist_train.sh projects/configs/dms/dms_r50/dms_g_r101/dms_g_t_r101_prune_25_cls.py 8
bash ./tools/dist_train.sh projects/configs/dms/dms_r50/dms_t_r101_finetune_cls.py 8 --work-dir work_dirs/${JOB_NAME}_finetune

export JOB_NAME=dms_gs_t_r101_prune_25_cls
export PTH_NAME=epoch_10
bash ./tools/dist_train.sh projects/configs/dms/dms_r50/dms_g_r101/dms_gs_t_r101_prune_25_cls.py 8
bash ./tools/dist_train.sh projects/configs/dms/dms_r50/dms_t_r101_finetune_cls.py 8 --work-dir work_dirs/${JOB_NAME}_finetune

export JOB_NAME=dms_gs_ed_r101_prune_25_cls
export PTH_NAME=epoch_10
bash ./tools/dist_train.sh projects/configs/dms/dms_r50/dms_g_r101/dms_gs_ed_r101_prune_25_cls.py 8
bash ./tools/dist_train.sh projects/configs/dms/dms_r50/dms_t_r101_finetune_cls.py 8 --work-dir work_dirs/${JOB_NAME}_finetune
