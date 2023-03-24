export JOB_NAME=dms_sq_r56super_prune_e
export PTH_NAME=epoch_100
python ./tools/train.py projects/configs/dms/dms_r56/dms_imp/dms_sq_r56super_prune_e.py
python ./tools/train.py projects/configs/dms/dms_r56/dms_r56super_finetune.py --work-dir work_dirs/${JOB_NAME}_finetune
