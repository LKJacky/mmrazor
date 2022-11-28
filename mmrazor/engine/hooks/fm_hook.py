# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import HOOKS
from mmengine.hooks import Hook
from mmengine.runner import Runner


@HOOKS.register_module()
class FmHook(Hook):

    def before_run(self, runner: Runner) -> None:
        max_iter = 100
        iter = 0
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
