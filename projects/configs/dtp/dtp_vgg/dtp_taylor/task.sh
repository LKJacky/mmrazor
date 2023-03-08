export JOB_NAME=dtp_a_vgg_prune_20
export PTH_NAME=epoch_15
python ./tools/train.py projects/configs/dtp/dtp_vgg/dtp_taylor/dtp_t_vgg_prune_20.py
python ./tools/train.py projects/configs/dtp/dtp_vgg/dtp_vgg_finetune.py --work-dir work_dirs/${JOB_NAME}_finetune
