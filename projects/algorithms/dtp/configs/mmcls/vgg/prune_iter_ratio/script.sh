export JOB_NAME=dtp_prune_vgg_r01
python ./tools/train.py projects/algorithms/dtp/configs/mmcls/vgg/prune_iter_ratio/${JOB_NAME}.py
python ./tools/train.py projects/algorithms/dtp/configs/mmcls/vgg/prune_iter_ratio/dtp_finetune_vgg.py --work-dir ./work_dirs/$JOB_NAME

export JOB_NAME=dtp_prune_vgg_r05
python ./tools/train.py projects/algorithms/dtp/configs/mmcls/vgg/prune_iter_ratio/${JOB_NAME}.py
python ./tools/train.py projects/algorithms/dtp/configs/mmcls/vgg/prune_iter_ratio/dtp_finetune_vgg.py --work-dir ./work_dirs/$JOB_NAME

export JOB_NAME=dtp_prune_vgg_r3
python ./tools/train.py projects/algorithms/dtp/configs/mmcls/vgg/prune_iter_ratio/${JOB_NAME}.py
python ./tools/train.py projects/algorithms/dtp/configs/mmcls/vgg/prune_iter_ratio/dtp_finetune_vgg.py --work-dir ./work_dirs/$JOB_NAME

export JOB_NAME=dtp_prune_vgg_r6
python ./tools/train.py projects/algorithms/dtp/configs/mmcls/vgg/prune_iter_ratio/${JOB_NAME}.py
python ./tools/train.py projects/algorithms/dtp/configs/mmcls/vgg/prune_iter_ratio/dtp_finetune_vgg.py --work-dir ./work_dirs/$JOB_NAME

export JOB_NAME=dtp_prune_vgg_r9
python ./tools/train.py projects/algorithms/dtp/configs/mmcls/vgg/prune_iter_ratio/${JOB_NAME}.py
python ./tools/train.py projects/algorithms/dtp/configs/mmcls/vgg/prune_iter_ratio/dtp_finetune_vgg.py --work-dir ./work_dirs/$JOB_NAME
