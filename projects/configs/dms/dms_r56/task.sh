export JOB_NAME=dms_r56_prune
export PTH_NAME=epoch_300
python ./tools/train.py projects/configs/dms/dms_r56/dms_r56_prune.py
python ./tools/train.py projects/configs/dms/dms_r56/dms_r56_finetune.py --work-dir work_dirs/${JOB_NAME}_finetune
