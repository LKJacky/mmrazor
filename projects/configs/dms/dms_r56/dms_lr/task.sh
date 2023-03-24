export JOB_NAME=dms_r56super_prune_e_lr2
export PTH_NAME=epoch_100
python ./tools/train.py projects/configs/dms/dms_r56/dms_lr/dms_r56super_prune_e_lr2.py
python ./tools/train.py projects/configs/dms/dms_r56/dms_r56super_finetune.py --work-dir work_dirs/${JOB_NAME}_finetune

export JOB_NAME=dms_r56super_prune_e_lr3
export PTH_NAME=epoch_100
python ./tools/train.py projects/configs/dms/dms_r56/dms_lr/dms_r56super_prune_e_lr3.py
python ./tools/train.py projects/configs/dms/dms_r56/dms_r56super_finetune.py --work-dir work_dirs/${JOB_NAME}_finetune


export JOB_NAME=dms_r56super_prune_e_lrc
export PTH_NAME=epoch_100
python ./tools/train.py projects/configs/dms/dms_r56/dms_lr/dms_r56super_prune_e_lrc.py
python ./tools/train.py projects/configs/dms/dms_r56/dms_r56super_finetune.py --work-dir work_dirs/${JOB_NAME}_finetune

export JOB_NAME=dms_r56super_prune_e_lrs
export PTH_NAME=epoch_100
python ./tools/train.py projects/configs/dms/dms_r56/dms_lr/dms_r56super_prune_e_lrs.py
python ./tools/train.py projects/configs/dms/dms_r56/dms_r56super_finetune.py --work-dir work_dirs/${JOB_NAME}_finetune
