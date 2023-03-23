export JOB_NAME=dms_c_r56super_prune
export PTH_NAME=epoch_300
python ./tools/train.py projects/configs/dms/dms_r56/dms_channel/dms_c_r56super_prune.py
python ./tools/train.py projects/configs/dms/dms_r56/dms_r56super_finetune.py --work-dir work_dirs/${JOB_NAME}_finetune
