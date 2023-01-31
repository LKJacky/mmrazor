python mmdeploy/tools/deploy.py \
    mmdeploy/configs/mmcls/classification_onnxruntime_dynamic.py \
   ./projects/models/vgg/configs/vgg_pretrain.py \
   ./work_dirs/pretrained/vgg_pretrained.pth \
    ./mmdeploy/demo/resources/face.png  \
    --work-dir work_dirs/mmdeploy_model/vgg \
    --device cpu \
    --dump-info

python mmdeploy/tools/test.py \
    mmdeploy/configs/mmcls/classification_onnxruntime_dynamic.py \
   ./projects/models/vgg/configs/vgg_pretrain.py \
    --model ./work_dirs/mmdeploy_model/vgg/end2end.onnx \
    --show-dir ./work_dirs/mmdeploy_model/vgg/
    --device cpu \
    --speed-test
