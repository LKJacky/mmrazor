export JOB_NAME=dms_r56super_prune_ep100
export PTH_NAME=epoch_100
python ./tools/train.py projects/configs/dms/dms_r56/dms_epoch_num/dms_r56super_prune_ep100.py
python ./tools/train.py projects/configs/dms/dms_r56/dms_r56super_finetune.py --work-dir work_dirs/${JOB_NAME}_finetune

export JOB_NAME=dms_r56super_prune_ep200
export PTH_NAME=epoch_200
python ./tools/train.py projects/configs/dms/dms_r56/dms_epoch_num/dms_r56super_prune_ep200.py
python ./tools/train.py projects/configs/dms/dms_r56/dms_r56super_finetune.py --work-dir work_dirs/${JOB_NAME}_finetune
