export JOB_NAME=dtp_r50_prune
export PTH_NAME=epoch_14
bash ./tools/dist_train.sh projects/configs/dtp/dtp_r50/dtp_r50_prune.py 8
bash ./tools/dist_train.sh projects/configs/dtp/dtp_r50/dtp_r50_finetune.py 8 --work-dir work_dirs/${JOB_NAME}_finetune
