export EXP_NAME=dms_t_swin

export JOB_NAME=${EXP_NAME}/dms_t_swin_ex_prune_22_cls
export PTH_NAME=epoch_30
bash ./tools/dist_train.sh projects/configs/dms/dms_swin/dms_t_swin/dms_t_swin_ex_prune_22_cls.py 8 --work-dir work_dirs/${JOB_NAME} --amp
bash ./tools/dist_train.sh projects/configs/dms/dms_swin/dms_t_swin/dms_t_swin_ex_finetune_22_cls.py 8 --work-dir work_dirs/${JOB_NAME}_finetune --amp
