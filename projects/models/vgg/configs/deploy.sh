python mmdeploy/tools/deploy.py \
    mmdeploy/configs/mmcls/classification_openvino_dynamic-224x224.py \
   ./projects/models/vgg/configs/vgg_pretrain.py \
   ./work_dirs/pretrained/vgg_pretrained.pth \
    ./mmdeploy/demo/resources/face.png  \
    --work-dir work_dirs/mmdeploy_model/ \
    --device cpu \
    --dump-info

python mmdeploy/tools/test.py \
    mmdeploy/configs/mmcls/classification_openvino_dynamic-224x224.py \
   ./projects/models/vgg/configs/vgg_pretrain.py \
    --model ./work_dirs/mmdeploy_model/end2end.xml \
    --batch-size=1 \
    --device cpu \
    --speed-test
