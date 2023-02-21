export JOB_NAME=dtp_prune_r50
bash ./tools/dist_train.sh projects/algorithms/dtp/configs/mmcls/resnet50/${JOB_NAME}.py 8
bash ./tools/dist_train.sh projects/algorithms/dtp/configs/mmcls/resnet50/dtp_finetune_r50.py 8 --work-dir ./work_dirs/$JOB_NAME
