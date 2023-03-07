export JOB_NAME=dtp_chip_vgg_prune
export PTH_NAME=epoch_15

python ./tools/train.py projects/configs/chip/chip_vgg.py
python ./tools/train.py projects/configs/chip/dtp_chip_vgg_prune.py
python ./tools/train.py projects/configs/chip/dtp_chip_vgg_finetune.py
