export JOB_NAME=dtp_leap_r56_prune_34_e300
export PTH_NAME=epoch_300
python ./tools/train.py projects/configs/dtp/dtp_r56/dtp_leap/dtp_leap_r56_prune_34_e300.py
python ./tools/train.py projects/configs/dtp/dtp_r56/dtp_r56_finetune.py --work-dir work_dirs/${JOB_NAME}_finetune
