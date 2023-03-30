export JOB_NAME=dms_r101_prune
export PTH_NAME=epoch_100
python ./tools/train.py projects/configs/dms/dms_r101/dms_r101_prune.py
python ./tools/train.py projects/configs/dms/dms_r101/dms_r101_finetune.py --work-dir work_dirs/${JOB_NAME}_finetune
