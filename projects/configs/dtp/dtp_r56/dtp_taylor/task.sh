export JOB_NAME=dtp_t_r56_prune_34
export PTH_NAME=epoch_30
python ./tools/train.py projects/configs/dtp/dtp_r56/dtp_taylor/dtp_t_r56_prune_34.py
python ./tools/train.py projects/configs/dtp/dtp_r56/dtp_r56_finetune.py --work-dir work_dirs/${JOB_NAME}_finetune
python ./tools/train.py projects/configs/dtp/dtp_r56/dtp_r56_finetune_cos.py --work-dir work_dirs/${JOB_NAME}_finetune_cos


export JOB_NAME=dtp_t_r56_prune_34_e3
export PTH_NAME=epoch_3
python ./tools/train.py projects/configs/dtp/dtp_r56/dtp_taylor/dtp_t_r56_prune_34_e3.py
python ./tools/train.py projects/configs/dtp/dtp_r56/dtp_r56_finetune.py --work-dir work_dirs/${JOB_NAME}_finetune

export JOB_NAME=dtp_t_r56_prune_34_e150
export PTH_NAME=epoch_150
python ./tools/train.py projects/configs/dtp/dtp_r56/dtp_taylor/dtp_t_r56_prune_34_e150.py
python ./tools/train.py projects/configs/dtp/dtp_r56/dtp_r56_finetune.py --work-dir work_dirs/${JOB_NAME}_finetune
