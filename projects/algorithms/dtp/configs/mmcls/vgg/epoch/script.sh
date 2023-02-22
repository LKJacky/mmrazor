export JOB_NAME=dtp_prune_vgg_e20
python ./tools/train.py projects/algorithms/dtp/configs/mmcls/vgg/epoch/${JOB_NAME}.py
python ./tools/train.py projects/algorithms/dtp/configs/mmcls/vgg/dtp_finetune_vgg_epoch.py --work-dir ./work_dirs/${JOB_NAME}_finetune
