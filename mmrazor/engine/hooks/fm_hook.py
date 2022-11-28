# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import HOOKS
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine import dist


@HOOKS.register_module()
class FmHook(Hook):


    def before_train(self, runner: Runner) -> None:
        max_iter = 100
        iter = 0
        if dist.is_distributed():
            model = runner.model.module.architecture
        else:
            model = runner.model.architecture

        runner.train_dataloader
        model.eval()
        for _, data_batch in enumerate(runner.train_dataloader):
            data = model.data_preprocessor(data_batch, True)
            model._run_forward(data, mode='tensor')  # type: ignore
            iter += 1
            if iter > max_iter:
                break

        return super().before_run(runner)
