export JOB_NAME=dtp_vgg_prune
export PTH_NAME=epoch_15
python ./tools/train.py projects/configs/dtp/dtp_vgg/dtp_vgg_prune.py
python ./tools/train.py projects/configs/dtp/dtp_vgg/dtp_vgg_finetune.py

export JOB_NAME=dtp_vgg_prune_30
export PTH_NAME=epoch_15
python ./tools/train.py projects/configs/dtp/dtp_vgg/dtp_vgg_prune_30.py
python ./tools/train.py projects/configs/dtp/dtp_vgg/dtp_vgg_finetune.py
