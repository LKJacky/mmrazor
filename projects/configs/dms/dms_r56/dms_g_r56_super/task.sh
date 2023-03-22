export JOB_NAME=dms_g_r56super_prune
export PTH_NAME=epoch_300
python ./tools/train.py projects/configs/dms/dms_r56/dms_g_r56_super/dms_g_r56super_prune.py
python ./tools/train.py projects/configs/dms/dms_r56/dms_r56super_finetune.py --work-dir work_dirs/${JOB_NAME}_finetune

export JOB_NAME=dms_gs_r56super_prune
export PTH_NAME=epoch_300
python ./tools/train.py projects/configs/dms/dms_r56/dms_g_r56_super/dms_gs_r56super_prune.py
python ./tools/train.py projects/configs/dms/dms_r56/dms_r56super_finetune.py --work-dir work_dirs/${JOB_NAME}_finetune


export JOB_NAME=dms_gs_e_r56super_prune
export PTH_NAME=epoch_300
python ./tools/train.py projects/configs/dms/dms_r56/dms_g_r56_super/dms_gs_e_r56super_prune.py
python ./tools/train.py projects/configs/dms/dms_r56/dms_r56super_finetune.py --work-dir work_dirs/${JOB_NAME}_finetune



export JOB_NAME=dms_gs_ed_r56super_prune
export PTH_NAME=epoch_300
python ./tools/train.py projects/configs/dms/dms_r56/dms_g_r56_super/dms_gs_ed_r56super_prune.py
python ./tools/train.py projects/configs/dms/dms_r56/dms_r56super_finetune.py --work-dir work_dirs/${JOB_NAME}_finetune
