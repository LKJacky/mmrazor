export JOB_NAME=dtp_prune_vgg_revert
python ./tools/train.py projects/algorithms/dtp/configs/mmcls/vgg/imp/${JOB_NAME}.py
python ./tools/train.py projects/algorithms/dtp/configs/mmcls/vgg/dtp_finetune_vgg.py --work-dir ./work_dirs/$JOB_NAME
