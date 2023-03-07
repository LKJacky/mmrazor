export JOB_NAME=dtp_r56_prune
export PTH_NAME=epoch_30
python ./tools/train.py projects/configs/dtp/dtp_r56/dtp_r56_prune.py
python ./tools/train.py projects/configs/dtp/dtp_r56/dtp_r56_finetune.py
