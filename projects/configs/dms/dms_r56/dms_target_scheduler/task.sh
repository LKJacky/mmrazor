export JOB_NAME=dms_r56super_prune_tscos
export PTH_NAME=epoch_100
python ./tools/train.py projects/configs/dms/dms_r56/dms_target_scheduler/dms_r56super_prune_tscos.py
python ./tools/train.py projects/configs/dms/dms_r56/dms_r56super_finetune.py --work-dir work_dirs/${JOB_NAME}_finetune
