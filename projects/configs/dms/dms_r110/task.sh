export JOB_NAME=dms_r110_prune
export PTH_NAME=epoch_100
python ./tools/train.py projects/configs/dms/dms_r110/dms_r110_prune.py
python ./tools/train.py projects/configs/dms/dms_r110/dms_r110_finetune.py --work-dir work_dirs/${JOB_NAME}_finetune
