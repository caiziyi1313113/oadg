from jdet.runner.runner import Runner as JDetRunner
import jittor as jt
import time
import datetime
from jdet.utils.general import parse_losses, sync, check_interval


class Runner(JDetRunner):
    """Runner with optional iter-based checkpointing."""
    def __init__(self):
        super().__init__()
        # If resuming, keep the config's max_iter (not the checkpoint's).
        if self.max_epoch is None and self.cfg.max_iter is not None:
            self.max_iter = self.cfg.max_iter
            self.total_iter = self.max_iter

    def train(self):
        self.model.train()

        start_time = time.time()
        for batch_idx, (images, targets) in enumerate(self.train_dataset):
            losses = self.model(images, targets)
            all_loss, losses = parse_losses(losses)
            self.optimizer.step(all_loss)

            by_epoch = self.max_epoch is not None
            self.scheduler.step(self.iter, self.epoch, by_epoch=by_epoch)

            if check_interval(self.iter, self.log_interval) and self.iter > 0:
                batch_size = len(images) * jt.world_size
                ptime = time.time() - start_time
                fps = batch_size * (batch_idx + 1) / ptime
                eta_time = (self.total_iter - self.iter) * ptime / (batch_idx + 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_time)))
                data = dict(
                    name=self.cfg.name,
                    lr=self.optimizer.cur_lr(),
                    iter=self.iter,
                    epoch=self.epoch,
                    batch_idx=batch_idx,
                    batch_size=batch_size,
                    total_loss=all_loss,
                    fps=fps,
                    eta=eta_str,
                )
                data.update(losses)
                data = sync(data)
                if jt.rank == 0:
                    self.logger.log(data)

            eval_iter = getattr(self.cfg, "eval_interval_iter", None)
            if check_interval(self.iter, eval_iter) and self.iter > 0:
                self.val()
                # val() sets eval mode; switch back to train for next iter
                self.model.train()

            ckpt_iter = getattr(self.cfg, "checkpoint_interval_iter", None)
            if check_interval(self.iter, ckpt_iter) and self.iter > 0:
                self.save()

            self.iter += 1
            if self.finish:
                break

        self.epoch += 1
