export JOB_NAME=dtp_fn_r56_prune_34
export PTH_NAME=epoch_30
python ./tools/train.py projects/configs/dtp/dtp_r56/dtp_fn.py/dtp_fn_r56_prune_34.py
python ./tools/train.py projects/configs/dtp/dtp_r56/dtp_r56_finetune.py --work-dir work_dirs/${JOB_NAME}_finetune
