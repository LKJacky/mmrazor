

export JOB_NAME=dtp_prune_r50_lre2_e140
export PTH_NAME=epoch_140
bash ./tools/dist_train.sh ./projects/algorithms/dtp/configs/mmcls/resnet50/dual_optimizer/${JOB_NAME}.py 8
bash ./tools/dist_train.sh projects/algorithms/dtp/configs/mmcls/resnet50/dtp_finetune_r50.py 8 --work-dir ./work_dirs/${JOB_NAME}_finetune

bash ./tools/dist_train.sh work_dirs/res_pruning.py 8
