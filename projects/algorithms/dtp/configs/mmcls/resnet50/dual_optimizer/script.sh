export JOB_NAME=dtp_prune_r50_lr0
bash ./projects/algorithms/dtp/configs/mmcls/resnet50/dual_optimizer/{JOB_NAME}.py 8
bash ./tools/dist_train.sh projects/algorithms/dtp/configs/mmcls/resnet50/dtp_finetune_r50.py 8 --work-dir ./work_dirs/${JOB_NAME}_finetune

export JOB_NAME=dtp_prune_r50_lr2
bash ./projects/algorithms/dtp/configs/mmcls/resnet50/dual_optimizer/{JOB_NAME}.py 8
bash ./tools/dist_train.sh projects/algorithms/dtp/configs/mmcls/resnet50/dtp_finetune_r50.py 8 --work-dir ./work_dirs/${JOB_NAME}_finetune

export JOB_NAME=dtp_prune_r50_lr3
bash ./projects/algorithms/dtp/configs/mmcls/resnet50/dual_optimizer/{JOB_NAME}.py 8
bash ./tools/dist_train.sh projects/algorithms/dtp/configs/mmcls/resnet50/dtp_finetune_r50.py 8 --work-dir ./work_dirs/${JOB_NAME}_finetune
