export JOB_NAME=dtp_fn_vgg_prune_20
export PTH_NAME=epoch_15
python ./tools/train.py projects/configs/dtp/dtp_vgg/dtp_fn.py/dtp_fn_vgg_prune_20.py
python ./tools/train.py projects/configs/dtp/dtp_vgg/dtp_vgg_finetune.py --work-dir ${JOB_NAME}_finetune
