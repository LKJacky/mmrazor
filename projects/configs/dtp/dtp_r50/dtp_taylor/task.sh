export JOB_NAME=dtp_t_r50_prune_50
export PTH_NAME=epoch_14
bash ./tools/dist_train.sh projects/configs/dtp/dtp_r50/dtp_taylor/dtp_t_r50_prune_50.py 8
bash ./tools/dist_train.sh projects/configs/dtp/dtp_r50/dtp_r50_finetune.py 8 --work-dir work_dirs/${JOB_NAME}_finetune

export JOB_NAME=dtp_t_r50_prune_25
export PTH_NAME=epoch_14
bash ./tools/dist_train.sh projects/configs/dtp/dtp_r50/dtp_taylor/dtp_t_r50_prune_25.py 8
bash ./tools/dist_train.sh projects/configs/dtp/dtp_r50/dtp_r50_finetune.py 8 --work-dir work_dirs/${JOB_NAME}_finetune


export JOB_NAME=dtp_t_r50_prune_50_cls
export PTH_NAME=epoch_9
bash ./tools/dist_train.sh projects/configs/dtp/dtp_r50/dtp_taylor/dtp_t_r50_prune_50_cls.py 8
bash ./tools/dist_train.sh projects/configs/dtp/dtp_r50/dtp_r50_finetune_cls.py 8 --work-dir work_dirs/${JOB_NAME}_finetune

export JOB_NAME=dtp_t_r50_prune_50_cls_90
export PTH_NAME=epoch_90
bash ./tools/dist_train.sh projects/configs/dtp/dtp_r50/dtp_taylor/dtp_t_r50_prune_50_cls_90.py 8
bash ./tools/dist_train.sh projects/configs/dtp/dtp_r50/dtp_r50_finetune_cls.py 8 --work-dir work_dirs/${JOB_NAME}_finetune

export JOB_NAME=dtp_t_r50_prune_25_cls_10
export PTH_NAME=epoch_10
bash ./tools/dist_train.sh projects/configs/dtp/dtp_r50/dtp_taylor/dtp_t_r50_prune_25_cls_10.py 8
bash ./tools/dist_train.sh projects/configs/dtp/dtp_r50/dtp_r50_finetune_cls.py 8 --work-dir work_dirs/${JOB_NAME}_finetune

export JOB_NAME=dtp_t_r50_prune_25_cls_20
export PTH_NAME=epoch_20
bash ./tools/dist_train.sh projects/configs/dtp/dtp_r50/dtp_taylor/dtp_t_r50_prune_25_cls_20.py 8
bash ./tools/dist_train.sh projects/configs/dtp/dtp_r50/dtp_r50_finetune_cls.py 8 --work-dir work_dirs/${JOB_NAME}_finetune

export JOB_NAME=dtp_t_r50_prune_25_cls_10_lr3
export PTH_NAME=epoch_10
bash ./tools/dist_train.sh projects/configs/dtp/dtp_r50/dtp_taylor/dtp_t_r50_prune_25_cls_10_lr3.py 8
bash ./tools/dist_train.sh projects/configs/dtp/dtp_r50/dtp_r50_finetune_cls.py 8 --work-dir work_dirs/${JOB_NAME}_finetune
