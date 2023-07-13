export SRUN_ARGS="--async"
export USE_CEPH=false
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29111
export GPUS=4
export GPUS_PER_NODE=4


# total batch size=384*2=768=96*8

sh timm_distributed_train.sh 8 timm_pruning.py ../data/imagenet_torch --model efficientnet_b4 -b 96 --sched step --epochs 30 \
--decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 \
--drop 0.4 --drop-path 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .048 \
--experiment 10_to_39 --pin-mem --resume output/train/10_to_39/last.pth.tar  --input-size 3 224 224 \
--target 0.25 --mutator_lr  0.0048