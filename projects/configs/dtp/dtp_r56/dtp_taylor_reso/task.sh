export JOB_NAME=dtp_tr_r56_prune_34
export PTH_NAME=epoch_300
python ./tools/train.py projects/configs/dtp/dtp_r56/dtp_taylor_reso/dtp_tr_r56_prune_34.py
python ./tools/train.py projects/configs/dtp/dtp_r56/dtp_r56_finetune.py --work-dir work_dirs/${JOB_NAME}_finetune
python ./tools/train.py projects/configs/dtp/dtp_r56/dtp_r56_finetune_cos.py --work-dir work_dirs/${JOB_NAME}_finetune_cos
