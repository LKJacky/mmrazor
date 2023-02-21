export JOB_NAME=dtp_prune_vgg_revert
python ./tools/train.py projects/algorithms/dtp/configs/mmcls/vgg/prune_iter_ratio/${JOB_NAME}.py
python ./tools/train.py projects/algorithms/dtp/configs/mmcls/vgg/prune_iter_ratio/dtp_finetune_vgg.py --work-dir ./work_dirs/$JOB_NAME
