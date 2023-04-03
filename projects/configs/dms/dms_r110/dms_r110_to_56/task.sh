
export EXP_NAME=dms_r110_to_56

export JOB_NAME=${EXP_NAME}/dms_r110_fix_l2_49
export PTH_NAME=epoch_100
python ./tools/train.py projects/configs/dms/dms_r110/dms_r110_to_56/dms_r110_fix_l2_49.py --work-dir work_dirs/${JOB_NAME}
python ./tools/train.py projects/configs/dms/dms_r110/dms_r110_1/dms_r110_finetune.py --work-dir work_dirs/${JOB_NAME}_finetune

export JOB_NAME=${EXP_NAME}/dms_r110super_fix_l2_37
export PTH_NAME=epoch_100
python ./tools/train.py projects/configs/dms/dms_r110/dms_r110_to_56/dms_r110super_fix_l2_37.py --work-dir work_dirs/${JOB_NAME}
python ./tools/train.py projects/configs/dms/dms_r110/dms_r110_to_56/dms_r110super_finetune.py --work-dir work_dirs/${JOB_NAME}_finetune
