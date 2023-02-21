export JOB_NAME=dtp_prune_vgg_r9_fix
python ./tools/train.py projects/algorithms/dtp/configs/mmcls/vgg/prune_schedule//${JOB_NAME}.py
python ./tools/train.py projects/algorithms/dtp/configs/mmcls/vgg/dtp_finetune_vgg.py --work-dir ./work_dirs/$JOB_NAME
