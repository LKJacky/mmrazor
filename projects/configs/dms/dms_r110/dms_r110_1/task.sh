
export EXP_NAME=dms_r110_ex1

export JOB_NAME=${EXP_NAME}/dms_r110_fix_l2
export PTH_NAME=epoch_100
python ./tools/train.py projects/configs/dms/dms_r110/dms_r110_1/dms_r110_fix_l2.py --work-dir work_dirs/${JOB_NAME}
python ./tools/train.py projects/configs/dms/dms_r110/dms_r110_1/dms_r110_finetune.py --work-dir work_dirs/${JOB_NAME}_finetune

export JOB_NAME=${EXP_NAME}/dms_r110_fix_l2+
export PTH_NAME=epoch_100
python ./tools/train.py projects/configs/dms/dms_r110/dms_r110_1/dms_r110_fix_l2+.py --work-dir work_dirs/${JOB_NAME}
python ./tools/train.py projects/configs/dms/dms_r110/dms_r110_1/dms_r110_finetune.py --work-dir work_dirs/${JOB_NAME}_finetune

export JOB_NAME=${EXP_NAME}/dms_r110_fix_l2+pre
export PTH_NAME=epoch_100
python ./tools/train.py projects/configs/dms/dms_r110/dms_r110_1/dms_r110_fix_l2+pre.py --work-dir work_dirs/${JOB_NAME}
python ./tools/train.py projects/configs/dms/dms_r110/dms_r110_1/dms_r110_finetune.py --work-dir work_dirs/${JOB_NAME}_finetune

export JOB_NAME=${EXP_NAME}/dms_r110_cos_l2
export PTH_NAME=epoch_100
python ./tools/train.py projects/configs/dms/dms_r110/dms_r110_1/dms_r110_cos_l2.py --work-dir work_dirs/${JOB_NAME}
python ./tools/train.py projects/configs/dms/dms_r110/dms_r110_1/dms_r110_finetune.py --work-dir work_dirs/${JOB_NAME}_finetune
