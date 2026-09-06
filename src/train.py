from pytorch_lightning.utilities import rank_zero_only
import torch
from dataloader.FireSpreadDataModule import FireSpreadDataModule
from pytorch_lightning.cli import LightningCLI
from models import SMPModel, BaseModel, ConvLSTMLightning, LogisticRegression  # noqa
from models import BaseModel
import wandb
import os
import subprocess

from dataloader.FireSpreadDataset import FireSpreadDataset
from dataloader.utils import get_means_stds_missing_values

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
torch.set_float32_matmul_precision('high')


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("trainer.default_root_dir",
                              "trainer.logger.init_args.save_dir")
        # NOTE(reproduction): the upstream link below forces every run's wandb
        # display name to "models.SMPModel", which collides across a 12-fold sweep.
        # Disabled so each fold can pass its own --trainer.logger.init_args.name.
        # parser.link_arguments("model.class_path",
        #                       "trainer.logger.init_args.name")
        parser.add_argument("--do_train", type=bool,
                            help="If True: skip training the model.")
        parser.add_argument("--do_predict", type=bool,
                            help="If True: compute predictions.")
        parser.add_argument("--do_test", type=bool,
                            help="If True: compute test metrics.")
        parser.add_argument("--do_validate", type=bool,
                            default=False, help="If True: compute val metrics.")
        parser.add_argument("--ckpt_path", type=str, default=None,
                            help="Path to checkpoint to load for resuming training, for testing and predicting.")

    def before_instantiate_classes(self):
        # The number of features is only known inside the data module, but we need that info to instantiate the model.
        # Since datamodule and model are instantiated at the same time with LightningCLI, we need to set the number of features here.
        n_features = FireSpreadDataset.get_n_features(
            self.config.data.n_leading_observations,
            self.config.data.features_to_keep,
            self.config.data.remove_duplicate_features)
        # The centroid channel count is DERIVED from the same table the dataset emits from
        # (FireSpreadDataset.CENTROID_CHANNEL_SPEC), never hardcoded here. With four independent
        # flags there are 16 valid layouts; a second hardcoded count would eventually disagree
        # with the tensor the dataloader actually produces, surfacing as an in_channels error at
        # the first forward pass -- or, worse, as a run labelled as one arm while training on
        # another's channels. Attribute access is direct rather than getattr-with-default on
        # purpose: a renamed flag must fail loudly instead of silently resolving to "no centroid
        # channels" and quietly training the wrong arm.
        centroid_names = FireSpreadDataset.centroid_channel_names(
            self.config.data.use_centroid_position,
            self.config.data.use_centroid_velocity,
            self.config.data.use_centroid_position_validity,
            self.config.data.use_centroid_velocity_validity,
            # One channel per enabled family PER model-visible frame. n_timesteps is the model's
            # own window; the peek-back frame the dataset loads for velocity is never emitted.
            n_timesteps=self.config.data.n_leading_observations)
        n_features += len(centroid_names)
        print(f"[centroid] {len(centroid_names)} extra channel(s): "
              f"{centroid_names if centroid_names else 'none'} "
              f"-> model n_channels = {n_features}")
        self.config.model.init_args.n_channels = n_features

        # The exact positive class weight changes with the data fold in the data module, but the weight is needed to instantiate the model.
        # Non-fire pixels are marked as missing values in the active fire feature, so we simply use that to compute the positive class weight.
        train_years, _, _ = FireSpreadDataModule.split_fires(
            self.config.data.data_fold_id, self.config.data.additional_data)
        _, _, missing_values_rates = get_means_stds_missing_values(train_years)
        fire_rate = 1 - missing_values_rates[-1]
        pos_class_weight = float(1 / fire_rate)

        self.config.model.init_args.pos_class_weight = pos_class_weight

    def before_fit(self):
        self.wandb_setup()

    def before_test(self):
        self.wandb_setup()

    def before_validate(self):
        self.wandb_setup()

    @rank_zero_only
    def wandb_setup(self):
        """
        Save the config used by LightningCLI to disk, then save that file to wandb.
        Using wandb.config adds some strange formating that means we'd have to do some 
        processing to be able to use it again as CLI input.

        Also define min and max metrics in wandb, because otherwise it just reports the 
        last known values, which is not what we want.
        """
        config_file_name = os.path.join(wandb.run.dir, "cli_config.yaml")

        cfg_string = self.parser.dump(self.config, skip_none=False)
        with open(config_file_name, "w") as f:
            f.write(cfg_string)
        wandb.save(config_file_name, policy="now", base_path=wandb.run.dir)
        wandb.define_metric("train_loss_epoch", summary="min")
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("train_f1_epoch", summary="max")
        wandb.define_metric("val_f1", summary="max")
        wandb.define_metric("val_avg_precision", summary="max")
        self.log_run_metadata()

    @rank_zero_only
    def log_run_metadata(self):
        """Tag the run with everything needed to find and group it again later.

        Written here rather than in the launch script so it holds for ANY invocation --
        a hand-typed one-off is tagged identically to a 12-fold sweep. EXPERIMENTS.md
        records earlier centroid runs arriving with no arm/fold/config tags at all,
        which made them unattributable after the fact; that is what this prevents.

        Every value except arm_id is DERIVED from the resolved config, so a tag cannot
        disagree with what the run actually trained on. arm_id is the one thing not
        recoverable from config (it names which YAML was layered on), so it comes from
        the environment and defaults to "unspecified" rather than guessing.
        """
        d = self.config.data
        centroid_names = FireSpreadDataset.centroid_channel_names(
            d.use_centroid_position, d.use_centroid_velocity,
            d.use_centroid_position_validity, d.use_centroid_velocity_validity,
            n_timesteps=d.n_leading_observations)

        arm_id = os.environ.get("ARM_ID", "unspecified")
        tags = [f"arm_{arm_id}",
                f"fold_{d.data_fold_id}",
                f"seed_{self.config.seed_everything}",
                f"nlead_{d.n_leading_observations}"]
        for flag, short in (("use_centroid_position", "pos"),
                            ("use_centroid_velocity", "vel"),
                            ("use_centroid_position_validity", "posvalid"),
                            ("use_centroid_velocity_validity", "velvalid")):
            if getattr(d, flag):
                tags.append(f"cent_{short}")
        if not centroid_names:
            tags.append("cent_none")

        # dict.fromkeys de-duplicates while preserving order: wandb_setup is called from
        # main() and again from before_fit/before_test, so this runs more than once.
        wandb.run.tags = tuple(dict.fromkeys(list(wandb.run.tags) + tags))

        # Which code produced this number. Without it, a sweep spanning a dataloader edit
        # is indistinguishable from one that did not -- and the centroid work is editing
        # the dataloader.
        try:
            sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                          stderr=subprocess.DEVNULL).decode().strip()
            dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "src", "cfgs"],
                                                 stderr=subprocess.DEVNULL).decode().strip())
        except Exception:
            sha, dirty = "unknown", None

        wandb.config.update({
            "arm_id": arm_id,
            "fold": d.data_fold_id,
            "seed": self.config.seed_everything,
            "n_lead": d.n_leading_observations,
            "centroid_channels": centroid_names,
            "n_centroid_channels": len(centroid_names),
            "n_channels": self.config.model.init_args.n_channels,
            "git_commit": sha,
            "git_dirty_src_or_cfgs": dirty,
        }, allow_val_change=True)
        print(f"[wandb] tags={list(wandb.run.tags)} git={sha}"
              f"{' (DIRTY)' if dirty else ''}")


def main():

    # LightningCLI automatically creates an argparse parser with required arguments and types,
    # and instantiates the model and datamodule. For this, it's important to import the model and datamodule classes above.
    cli = MyLightningCLI(BaseModel, FireSpreadDataModule, subclass_mode_model=True, save_config_kwargs={
        "overwrite": True}, parser_kwargs={"parser_mode": "yaml"}, run=False)
    cli.wandb_setup()

    if cli.config.do_train:
        cli.trainer.fit(cli.model, cli.datamodule,
                        ckpt_path=cli.config.ckpt_path)

    # If we have trained a model, use the best checkpoint for testing and predicting.
    # Without this, the model's state at the end of the training would be used, which is not necessarily the best.
    ckpt = cli.config.ckpt_path
    if cli.config.do_train:
        ckpt = "best"

    if cli.config.do_validate:
        cli.trainer.validate(cli.model, cli.datamodule, ckpt_path=ckpt)

    if cli.config.do_test:
        # torchmetrics' binned PrecisionRecallCurve / ConfusionMatrix / JaccardIndex
        # fall back to an `arange(n_bins).repeat(n_pixels, 1)` path whenever torch's
        # deterministic-algorithms flag is set (that includes `--trainer.deterministic
        # warn`, the config default). That fallback allocates one large tensor per
        # batch and OOMs a 12 GB card in the first test batches. Test runs over frozen
        # weights, so disabling determinism here changes no metric value -- it only
        # lets `--do_train=true --do_test=true` complete in a single process instead
        # of forcing a separate deterministic=false test sweep. Training above is
        # unaffected; it already ran under whatever --trainer.deterministic was given.
        torch.use_deterministic_algorithms(False)
        cli.trainer.test(cli.model, cli.datamodule, ckpt_path=ckpt)

    if cli.config.do_predict:
        print(f"Loading checkpoint from: {ckpt}")

        # Produce predictions, save them in a single file, including ground truth fire targets and input fire masks.
        prediction_output = cli.trainer.predict(
            cli.model, cli.datamodule, ckpt_path=ckpt)
        #torch.save(prediction_output, "prediction_output.pt")
        x_af = torch.cat(
            list(map(lambda tup: tup[0][:, -1, :, :].squeeze(), prediction_output)), axis=0)
        y = torch.cat(list(map(lambda tup: tup[1], prediction_output)), axis=0)
        y_hat = torch.cat(
            list(map(lambda tup: tup[2], prediction_output)), axis=0)
        fire_masks_combined = torch.cat(
            [x_af.unsqueeze(0), y_hat.unsqueeze(0), y.unsqueeze(0)], axis=0)

        predictions_file_name = os.path.join(
            cli.config.trainer.default_root_dir, f"predictions_{wandb.run.id}.pt")
        torch.save(fire_masks_combined, predictions_file_name)


if __name__ == "__main__":
    main()
