
export EXP_NAME=dms_r110_to_56_influence

export JOB_NAME=${EXP_NAME}/dms_r110_fix_l2_49_pretrain
export PTH_NAME=epoch_100
python ./tools/train.py projects/configs/dms/dms_r110/dms_r110_to_56_influence/dms_r110_fix_l2_49_pretrain.py --work-dir work_dirs/${JOB_NAME}
python ./tools/train.py projects/configs/dms/dms_r110/dms_r110_to_56_influence/dms_r110_finetune.py --work-dir work_dirs/${JOB_NAME}_finetune

export JOB_NAME=${EXP_NAME}/dms_r110_fix_l2_49
export PTH_NAME=epoch_100
python ./tools/train.py projects/configs/dms/dms_r110/dms_r110_to_56_influence/dms_r110_fix_l2_49.py --work-dir work_dirs/${JOB_NAME}
python ./tools/train.py projects/configs/dms/dms_r110/dms_r110_to_56_influence/dms_r110_finetune.py --work-dir work_dirs/${JOB_NAME}_finetune
python ./tools/train.py projects/configs/dms/dms_r110/dms_r110_to_56_influence/dms_r110_finetune_reset.py --work-dir work_dirs/${JOB_NAME}_finetune_reset
