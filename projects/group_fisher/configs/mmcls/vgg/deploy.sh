python mmdeploy/tools/deploy.py \
    mmdeploy/configs/mmcls/classification_openvino_dynamic-224x224.py \
    projects/group_fisher/configs/mmcls/vgg/vgg_group_fisher_finetune.py \
    ./work_dirs/vgg_group_fisher_finetune/best_accuracy/top1_epoch_142.pth \
    ./mmdeploy/demo/resources/face.png  \
    --work-dir work_dirs/mmdeploy_model/ \
    --device cpu \
    --dump-info

python mmdeploy/tools/test.py \
    mmdeploy/configs/mmcls/classification_openvino_dynamic-224x224.py \
    projects/group_fisher/configs/mmcls/vgg/vgg_group_fisher_finetune.py \
    --model ./work_dirs/mmdeploy_model/end2end.xml \
    --batch-size=1 \
    --device cpu \
    --speed-test
