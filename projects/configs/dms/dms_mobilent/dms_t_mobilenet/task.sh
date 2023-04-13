export EXP_NAME=dms_t_mobilenet

export JOB_NAME=${EXP_NAME}/dms_t_mobile_ex_prune_22_cls
export PTH_NAME=epoch_30
bash ./tools/dist_train.sh projects/configs/dms/dms_mobilent/dms_t_mobilenet/dms_t_mobile_ex_prune_22_cls.py 8 --work-dir work_dirs/${JOB_NAME}
bash ./tools/dist_train.sh projects/configs/dms/dms_mobilent/dms_t_mobilenet/dms_t_mobile_ex_finetune_22_cls.py 8 --work-dir work_dirs/${JOB_NAME}_finetune
