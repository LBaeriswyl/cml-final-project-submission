import ast
import importlib
import argparse
import gc
import math
import os
import sys
import random
import time
import json
from multiprocessing import Value
from pathlib import Path
from typing import BinaryIO, Union

import bitsandbytes as bnb
import toml
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import torch
from torchvision import transforms
import sys, humanize, psutil, GPUtil
from transformers import CLIPTextModel, CLIPTokenizer

from huggingface_hub import HfApi

# from torchinfo import summary
try:
    import intel_extension_for_pytorch as ipex

    if torch.xpu.is_available():
        from library.ipex import ipex_init

        ipex_init()
except Exception:
    pass
from accelerate.utils import set_seed
from diffusers import DDPMScheduler

from library import train_util
import library.config_util as config_util
from library.custom_train_functions import (
    add_custom_train_arguments,
    fix_noise_scheduler_betas_for_zero_terminal_snr,
    apply_snr_weight,
    get_weighted_text_embeddings,
    prepare_scheduler_for_custom_training,
    scale_v_prediction_loss_like_noise_prediction,
    add_v_prediction_like_loss,
    apply_debiased_estimation,
)

DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


def mem_report():
    print("CPU RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available))
    GPUs = GPUtil.getGPUs()
    for i, gpu in enumerate(GPUs):
        print(
            "GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%".format(
                i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil * 100
            )
        )


class HuggingFaceRepo:
    def __init__(self, repo_id: str, repo_type: str, hf_token: str):
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.token = hf_token
        self.private = args.huggingface_repo_visibility is not "public"
        self.api = HfApi(token=self.token)

    def exists_repo(self, revision: str = "main"):
        try:
            self.api.repo_info(
                repo_id=self.repo_id, revision=revision, repo_type=self.repo_type
            )
            return True
        except:
            return False

    def upload(
        self,
        args: argparse.Namespace,
        src: Union[str, Path, bytes, BinaryIO],
        dest_suffix: str = "",
    ):
        path_in_repo = (
            args.huggingface_path_in_repo + dest_suffix
            if args.huggingface_path_in_repo is not None
            else None
        )

        if not self.exists_repo():
            try:
                self.api.create_repo(
                    repo_id=self.repo_id, repo_type=self.repo_type, private=self.private
                )
            except Exception as e:
                print("===========================================")
                print(f"HF repo does not exist and failed to create {e}")
                print("===========================================")

        is_folder = (type(src) == str and os.path.isdir(src)) or (
            isinstance(src, Path) and src.is_dir()
        )

        try:
            if is_folder:
                self.api.upload_folder(
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    folder_path=src,
                    path_in_repo=path_in_repo,
                )
            else:
                self.api.upload_file(
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    path_or_fileobj=src,
                    path_in_repo=path_in_repo,
                )
        except Exception as e:
            print("===========================================")
            print(f"failed to upload to HuggingFace  {e}")
            print("===========================================")


class NetworkTrainer:
    def __init__(self, args):
        self.vae_scale_factor = 0.18215
        self.is_sdxl = False

        self.text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=None
        )
        # function for saving/removing
        self.hf_repo = HuggingFaceRepo(
            repo_id=args.huggingface_repo_id,
            repo_type=args.huggingface_repo_type,
            hf_token=args.huggingface_token,
        )

    def save_model(
        self,
        ckpt_name,
        unwrapped_nw,
        save_dtype,
    ):
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_file = os.path.join(args.output_dir, ckpt_name)

        self.accelerator.print(f"\nsaving checkpoint: {ckpt_file}")

        unwrapped_nw.save_weights(ckpt_file, save_dtype, None)
        if args.huggingface_repo_id is not None:
            self.hf_repo.upload(
                args,
                ckpt_file,
                "/" + ckpt_name,
            )

    def remove_model(self, old_ckpt_name):
        old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
        if os.path.exists(old_ckpt_file):
            self.accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
            os.remove(old_ckpt_file)

    def generate_step_logs(
        self,
        args: argparse.Namespace,
        current_loss,
        avr_loss,
        lr_scheduler,
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
    ):
        logs = {"loss/current": current_loss, "loss/average": avr_loss}

        if keys_scaled is not None:
            logs["max_norm/keys_scaled"] = keys_scaled
            logs["max_norm/average_key_norm"] = mean_norm
            logs["max_norm/max_key_norm"] = maximum_norm

        lrs = lr_scheduler.get_last_lr()

        if (
            args.network_train_text_encoder_only or len(lrs) <= 2
        ):  # not block lr (or single block)
            if args.network_train_unet_only:
                logs["lr/unet"] = float(lrs[0])
            elif args.network_train_text_encoder_only:
                logs["lr/textencoder"] = float(lrs[0])
            else:
                logs["lr/textencoder"] = float(lrs[0])
                logs["lr/unet"] = float(lrs[-1])  # may be same to textencoder

            if (
                args.optimizer_type.lower().startswith("DAdapt".lower())
                or args.optimizer_type.lower() == "Prodigy".lower()
            ):  # tracking d*lr value of unet.
                logs["lr/d*lr"] = (
                    lr_scheduler.optimizers[-1].param_groups[0]["d"]
                    * lr_scheduler.optimizers[-1].param_groups[0]["lr"]
                )
        else:
            idx = 0
            if not args.network_train_unet_only:
                logs["lr/textencoder"] = float(lrs[0])
                idx = 1

            for i in range(idx, len(lrs)):
                logs[f"lr/group{i}"] = float(lrs[i])
                if (
                    args.optimizer_type.lower().startswith("DAdapt".lower())
                    or args.optimizer_type.lower() == "Prodigy".lower()
                ):
                    logs[f"lr/d*lr/group{i}"] = (
                        lr_scheduler.optimizers[-1].param_groups[i]["d"]
                        * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
                    )

        return logs

    def assert_extra_args(self, args, train_dataset_group):
        pass

    def clean_optimizer_args(self, optimizer_args):
        optimizer_kwargs = {}
        if optimizer_args is not None and len(optimizer_args) > 0:
            for arg in optimizer_args:
                key, value = arg.split("=")
                value = ast.literal_eval(value)
                optimizer_kwargs[key] = value
        return optimizer_kwargs

    def get_optimizer(self, args, trainable_params):
        optimizer_type = args.optimizer_type
        if args.use_8bit_adam:
            assert (
                not args.use_lion_optimizer
            ), "both option use_8bit_adam and use_lion_optimizer are specified"
            assert (
                optimizer_type is None or optimizer_type == ""
            ), "both option use_8bit_adam and optimizer_type are specified"
            optimizer_type = "AdamW8bit"

        optimizer_type = optimizer_type.lower()
        optimizer_kwargs = self.clean_optimizer_args(args.optimizer_args)
        lr = args.learning_rate

        optimizer_dict = {
            "adamw": torch.optim.AdamW,
            "adamw8bit": bnb.optim.AdamW8bit,
            "sgdnesterov8bit": bnb.optim.SGD8bit,
            "sgdnesterov": torch.optim.SGD,
            "pagedadamw8bit": bnb.optim.PagedAdamW8bit,
            "pagedadamw32bit": bnb.optim.PagedAdamW32bit,
        }
        if optimizer_type not in optimizer_dict:
            print("Optimizer type not recognized, defaulted to AdamW. ")
            print("Available options: {}".format(optimizer_dict.keys()))
            optimizer_type = "adamw"

        if "nesterov" in optimizer_type:
            optimizer_kwargs["nesterov"] = True
            if "momentum" in optimizer_kwargs:
                optimizer_kwargs["momentum"] = 0.9

        optimizer_class = optimizer_dict[optimizer_type]
        optimizer = optimizer_class(
            trainable_params, lr=lr, **optimizer_kwargs
        )

        optimizer_name = optimizer_class.__module__ + "." + optimizer_class.__name__
        optimizer_args = ",".join([f"{k}={v}" for k, v in optimizer_kwargs.items()])

        return optimizer_name, optimizer_args, optimizer


    def load_target_model(self, args, weight_dtype, accelerator):
        text_encoder, vae, unet, _ = train_util.load_target_model(
            args, weight_dtype, accelerator
        )
        return (
            "sd_v1",
            text_encoder,
            vae,
            unet,
        )

    def load_tokenizer(self, args):
        tokenizer = train_util.load_tokenizer(args)
        return tokenizer

    def is_text_encoder_outputs_cached(self, args):
        return False

    def is_train_text_encoder(self, args):
        return (
            not args.network_train_unet_only
            and not self.is_text_encoder_outputs_cached(args)
        )

    def cache_text_encoder_outputs_if_needed(
        self,
        args,
        accelerator,
        unet,
        vae,
        tokenizers,
        text_encoders,
        data_loader,
        weight_dtype,
    ):
        for t_enc in text_encoders:
            t_enc.to(accelerator.device)

    def get_text_cond(
        self, args, accelerator, batch, tokenizers, text_encoders, weight_dtype
    ):
        encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]
        return encoder_hidden_states

    def call_unet(
        self,
        args,
        accelerator,
        unet,
        noisy_latents,
        timesteps,
        text_conds,
        batch,
        weight_dtype,
    ):
        noise_pred = unet(noisy_latents, timesteps, text_conds).sample
        return noise_pred

    def enable_gradient_checkpointing(
        self, gradient_checkpointing, unet, text_encoders, network
    ):
        if gradient_checkpointing:
            unet.enable_gradient_checkpointing()
            for t_enc in text_encoders:
                t_enc.gradient_checkpointing_enable()
            del t_enc
            network.enable_gradient_checkpointing()  # may have no effect

    def get_network(self, args, vae, text_encoder, unet):
        # prepare network
        network_module = importlib.import_module(args.network_module)
        net_kwargs = {}
        if args.network_args is not None:
            for net_arg in args.network_args:
                key, value = net_arg.split("=")
                net_kwargs[key] = value

        if args.dim_from_weights:
            network, _ = network_module.create_network_from_weights(
                1, args.network_weights, vae, text_encoder, unet, **net_kwargs
            )
        else:
            if "dropout" not in net_kwargs:
                net_kwargs["dropout"] = args.network_dropout
            network = network_module.create_network(
                1.0,
                args.network_dim,
                args.network_alpha,
                vae,
                text_encoder,
                unet,
                neuron_dropout=args.network_dropout,
                **net_kwargs,
            )
        if network is None:
            return
        if hasattr(network, "prepare_network"):
            network.prepare_network(args)
        if args.scale_weight_norms and not hasattr(
            network, "apply_max_norm_regularization"
        ):
            print(
                "warning: scale_weight_norms is specified but the network does not support it / scale_weight_normsが指定されていますが、ネットワークが対応していません"
            )
            args.scale_weight_norms = False

        return network, net_kwargs

    def train(self, args):
        session_id = random.randint(0, 2**32)
        training_started_at = time.time()
        train_util.verify_training_args(args)
        train_util.prepare_dataset_args(args, True)

        cache_latents = args.cache_latents
        if args.seed is None:
            args.seed = random.randint(0, 2**32)
        set_seed(args.seed)

        tokenizer = self.load_tokenizer(args)
        tokenizers = tokenizer if isinstance(tokenizer, list) else [tokenizer]

        train_transforms = transforms.Compose(
            [
                transforms.Resize(
                    args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(args.resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        dataset = load_dataset(
            args.dataset_name,
            None,
            cache_dir=None,
            data_dir=None,
        ).with_format("torch")

        current_epoch = Value("i", 0)
        current_step = Value("i", 0)

        column_names = dataset["train"].column_names

        dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
        image_column = (
            dataset_columns[0] if dataset_columns is not None else column_names[0]
        )

        print("preparing accelerator")
        accelerator = train_util.prepare_accelerator(args)
        self.accelerator = accelerator
        is_main_process = accelerator.is_main_process
        caption_column = (
            dataset_columns[1] if dataset_columns is not None else column_names[1]
        )
        weight_dtype = torch.float32
        # if accelerator.mixed_precision == "fp16":
        #     weight_dtype = torch.float16
        # elif accelerator.mixed_precision == "bf16":
        #     weight_dtype = torch.bfloat16
        self.text_encoder.to(accelerator.device, dtype=weight_dtype)

        def preprocess_train(examples):
            images = [image.convert("RGB") for image in examples[image_column]]
            examples["pixel_values"] = [train_transforms(image) for image in images]
            examples["input_ids"] = tokenize_captions(examples)
            return examples

        def tokenize_captions(examples, is_train=True):
            captions = []
            for caption in examples[caption_column]:
                if isinstance(caption, str):
                    captions.append(caption)
                elif isinstance(caption, (list, np.ndarray)):
                    # take a random caption if there are multiple
                    captions.append(random.choice(caption) if is_train else caption[0])
                else:
                    raise ValueError(
                        f"Caption column `{caption_column}` should contain either strings or lists of strings."
                    )
            inputs = tokenizer(
                captions,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return inputs.input_ids

        with accelerator.main_process_first():
            if args.max_train_samples is not None:
                dataset["train"] = (
                    dataset["train"]
                    .shuffle(seed=args.seed)
                    .select(range(args.max_train_samples))
                )
            # Set the training transforms
            train_dataset_group = dataset["train"].with_transform(preprocess_train)

        if args.debug_dataset:
            train_util.debug_dataset(train_dataset_group)
            return
        if len(train_dataset_group) == 0:
            print(
                "No data found. Please verify arguments (train_data_dir must be the parent of folders with images)"
            )
            return

        def collate_fn(examples):
            pixel_values = torch.stack(
                [example["pixel_values"] for example in examples]
            )
            pixel_values = pixel_values.to(
                memory_format=torch.contiguous_format
            ).float()
            input_ids = torch.stack([example["input_ids"] for example in examples])
            return {"pixel_values": pixel_values, "input_ids": input_ids}

        collator = collate_fn

        weight_dtype, save_dtype = train_util.prepare_dtype(args)
        vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

        model_version, text_encoder, vae, unet = self.load_target_model(
            args, weight_dtype, accelerator
        )
        text_encoders = (
            text_encoder if isinstance(text_encoder, list) else [text_encoder]
        )
        train_util.replace_unet_modules(
            unet, args.mem_eff_attn, args.xformers, args.sdpa
        )

        sys.path.append(os.path.dirname(__file__))

        self.cache_text_encoder_outputs_if_needed(
            args,
            accelerator,
            unet,
            vae,
            tokenizers,
            text_encoders,
            train_dataset_group,
            weight_dtype,
        )

        network, net_kwargs = self.get_network(args, vae, text_encoder, unet)
        if network is None:
            return

        mem_report()

        train_unet = not args.network_train_text_encoder_only
        train_text_encoder = self.is_train_text_encoder(args)
        network.apply_to(text_encoder, unet, train_text_encoder, train_unet)

        if args.network_weights is not None:
            info = network.load_weights(args.network_weights)
            accelerator.print(
                f"load network weights from {args.network_weights}: {info}"
            )
        self.enable_gradient_checkpointing(
            args.gradient_checkpointing, unet, text_encoders, network
        )

        accelerator.print("prepare optimizer, data loader etc.")
        try:
            trainable_params = network.prepare_optimizer_params(
                args.text_encoder_lr, args.unet_lr, args.learning_rate
            )
        except TypeError:
            accelerator.print(
                "Deprecated: use prepare_optimizer_params(text_encoder_lr, unet_lr, learning_rate) instead of prepare_optimizer_params(text_encoder_lr, unet_lr)"
            )
            trainable_params = network.prepare_optimizer_params(
                args.text_encoder_lr, args.unet_lr
            )

        optimizer_name, optimizer_args, optimizer = self.get_optimizer(
            args, trainable_params
        )

        n_workers = min(args.max_data_loader_n_workers, os.cpu_count() - 1)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            batch_size=1,
            shuffle=True,
            collate_fn=collator,
            num_workers=n_workers,
            persistent_workers=args.persistent_data_loader_workers,
        )

        if args.max_train_epochs is not None:
            args.max_train_steps = args.max_train_epochs * math.ceil(
                len(train_dataloader)
                / accelerator.num_processes
                / args.gradient_accumulation_steps
            )
            accelerator.print(
                f"override steps. steps for {args.max_train_epochs} epochs is {args.max_train_steps}"
            )

        lr_scheduler = train_util.get_scheduler_fix(
            args, optimizer, accelerator.num_processes
        )

        unet.requires_grad_(False)
        unet.to(dtype=weight_dtype)
        for t_enc in text_encoders:
            t_enc.requires_grad_(False)

        network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            network, optimizer, train_dataloader, lr_scheduler
        )
        if train_unet:
            unet = accelerator.prepare(unet)
            for t_enc in text_encoders:
                t_enc.to(accelerator.device, dtype=weight_dtype)
        if train_text_encoder:
            if len(text_encoders) > 1:
                t_enc1, t_enc2 = accelerator.prepare(
                    text_encoders[0],
                    text_encoders[1],
                )
                text_encoder = text_encoders = [t_enc1, t_enc2]
                del t_enc1, t_enc2
            else:
                text_encoder = accelerator.prepare(text_encoder)
                text_encoders = [text_encoder]
            unet.to(accelerator.device, dtype=weight_dtype)

        # transform DDP after prepare (train_network here only)
        text_encoders = train_util.transform_models_if_DDP(text_encoders)
        unet, network = train_util.transform_models_if_DDP([unet, network])

        if args.gradient_checkpointing:
            # according to TI example in Diffusers, train is required
            unet.train()
            for t_enc in text_encoders:
                t_enc.train()
                if train_text_encoder:
                    t_enc.text_model.embeddings.requires_grad_(True)
            if not train_text_encoder:  # train U-Net only
                unet.parameters().__next__().requires_grad_(True)
        else:
            unet.eval()
            for t_enc in text_encoders:
                t_enc.eval()

        del t_enc

        network.prepare_grad_etc(text_encoder, unet)

        if not cache_latents:
            vae.requires_grad_(False)
            vae.eval()
            vae.to(accelerator.device, dtype=vae_dtype)

        if args.full_fp16:
            train_util.patch_accelerator_for_fp16_training(accelerator)

        train_util.resume_from_local_or_hf_if_specified(accelerator, args)

        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps
        )
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
            args.save_every_n_epochs = (
                math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1
            )

        self.print_train_info(args, num_train_epochs, train_dataloader)

        minimum_metadata = {}


        progress_bar = tqdm(
            range(args.max_train_steps),
            smoothing=0,
            disable=not accelerator.is_local_main_process,
            desc="steps",
        )
        global_step = 0

        noise_scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
        )
        prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)
        if args.zero_terminal_snr:
            fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)

        if accelerator.is_main_process:
            init_kwargs = {}
            if args.log_tracker_config is not None:
                init_kwargs = toml.load(args.log_tracker_config)
            accelerator.init_trackers(
                "network_train"
                if args.log_tracker_name is None
                else args.log_tracker_name,
                init_kwargs=init_kwargs,
            )

        loss_recorder = train_util.LossRecorder()
        del train_dataset_group

        # callback for step start
        if hasattr(network, "on_step_start"):
            on_step_start = network.on_step_start
        else:
            on_step_start = lambda *args, **kwargs: None

        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=vae_dtype)
        # training loop
        mem_report()
        for epoch in range(num_train_epochs):
            accelerator.print(f"\nepoch {epoch + 1}/{num_train_epochs}")
            current_epoch.value = epoch + 1


            network.on_epoch_start(text_encoder, unet)

            for step, batch in enumerate(train_dataloader):
                current_step.value = global_step
                with accelerator.accumulate(network):
                    on_step_start(text_encoder, unet)

                    with torch.no_grad():
                        latents = vae.encode(
                            batch["pixel_values"]
                            .to(accelerator.device)
                            .to(dtype=vae_dtype)
                        ).latent_dist.sample()

                        latents = latents * self.vae_scale_factor

                    with torch.set_grad_enabled(
                        train_text_encoder
                    ), accelerator.autocast():
                        # Get the text embedding for conditioning
                        if args.weighted_captions:
                            text_encoder_conds = get_weighted_text_embeddings(
                                tokenizer,
                                text_encoder,
                                batch["captions"],
                                accelerator.device,
                                args.max_token_length // 75
                                if args.max_token_length
                                else 1,
                                clip_skip=args.clip_skip,
                            )
                        else:
                            text_encoder_conds = self.get_text_cond(
                                args,
                                accelerator,
                                batch,
                                tokenizers,
                                text_encoders,
                                weight_dtype,
                            )

                    # Sample noise, sample a random timestep for each image, and add noise to the latents,
                    # with noise offset and/or multires noise if specified
                    (
                        noise,
                        noisy_latents,
                        timesteps,
                    ) = train_util.get_noise_noisy_latents_and_timesteps(
                        args, noise_scheduler, latents
                    )

                    # Predict the noise residual
                    with accelerator.autocast():
                        noise_pred = self.call_unet(
                            args,
                            accelerator,
                            unet,
                            noisy_latents,
                            timesteps,
                            text_encoder_conds,
                            batch,
                            weight_dtype,
                        )

                    if args.v_parameterization:
                        # v-parameterization training
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        target = noise

                    loss = torch.nn.functional.mse_loss(
                        noise_pred.float(), target.float(), reduction="none"
                    )
                    loss = loss.mean([1, 2, 3])

                    if args.min_snr_gamma:
                        loss = apply_snr_weight(
                            loss,
                            timesteps,
                            noise_scheduler,
                            args.min_snr_gamma,
                            args.v_parameterization,
                        )
                    if args.scale_v_pred_loss_like_noise_pred:
                        loss = scale_v_prediction_loss_like_noise_prediction(
                            loss, timesteps, noise_scheduler
                        )
                    if args.v_pred_like_loss:
                        loss = add_v_prediction_like_loss(
                            loss, timesteps, noise_scheduler, args.v_pred_like_loss
                        )
                    if args.debiased_estimation_loss:
                        loss = apply_debiased_estimation(
                            loss, timesteps, noise_scheduler
                        )

                    loss = loss.mean()

                    accelerator.backward(loss)
                    if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                        params_to_clip = network.get_trainable_params()
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if args.scale_weight_norms:
                    (
                        keys_scaled,
                        mean_norm,
                        maximum_norm,
                    ) = network.apply_max_norm_regularization(
                        args.scale_weight_norms, accelerator.device
                    )
                    max_mean_logs = {
                        "Keys Scaled": keys_scaled,
                        "Average key norm": mean_norm,
                    }
                else:
                    keys_scaled, mean_norm, maximum_norm = None, None, None

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    train_util.sample_images(
                        accelerator,
                        args,
                        None,
                        global_step,
                        accelerator.device,
                        vae,
                        tokenizer,
                        text_encoder,
                        unet,
                    )

                    if (
                        args.save_every_n_steps is not None
                        and global_step % args.save_every_n_steps == 0
                    ):
                        self.checkpointing(
                            args,
                            global_step,
                            epoch,
                            network,
                            minimum_metadata,
                            {},
                            save_dtype,
                        )

                current_loss = loss.detach().item()
                loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
                avr_loss: float = loss_recorder.moving_average
                logs = {"avr_loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if args.scale_weight_norms:
                    progress_bar.set_postfix(**{**max_mean_logs, **logs})

                if args.logging_dir is not None:
                    logs = self.generate_step_logs(
                        args,
                        current_loss,
                        avr_loss,
                        lr_scheduler,
                        keys_scaled,
                        mean_norm,
                        maximum_norm,
                    )
                    accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break
                mem_report()

            if args.logging_dir is not None:
                logs = {"loss/epoch": loss_recorder.moving_average}
                accelerator.log(logs, step=epoch + 1)

            accelerator.wait_for_everyone()

            if args.save_every_n_epochs is not None:
                saving = (epoch + 1) % args.save_every_n_epochs == 0 and (
                    epoch + 1
                ) < num_train_epochs
                if is_main_process and saving:
                    ckpt_name = train_util.get_epoch_ckpt_name(
                        args, "." + args.save_model_as, epoch + 1
                    )
                    self.save_model(
                        ckpt_name,
                        accelerator.unwrap_model(network),
                        save_dtype,
                    )

                    remove_epoch_no = train_util.get_remove_epoch_no(args, epoch + 1)
                    if remove_epoch_no is not None:
                        remove_ckpt_name = train_util.get_epoch_ckpt_name(
                            args, "." + args.save_model_as, remove_epoch_no
                        )
                        self.remove_model(remove_ckpt_name)

                    if args.save_state:
                        train_util.save_and_remove_state_on_epoch_end(
                            args, accelerator, epoch + 1
                        )

            train_util.sample_images(
                accelerator,
                args,
                epoch + 1,
                global_step,
                accelerator.device,
                vae,
                tokenizer,
                text_encoder,
                unet,
            )

            # end of epoch


        if is_main_process:
            network = accelerator.unwrap_model(network)

        accelerator.end_training()

        if is_main_process and args.save_state:
            train_util.save_state_on_train_end(args, accelerator)

        if is_main_process:
            self.save_model(
                train_util.get_last_ckpt_name(args, "." + args.save_model_as),
                network,
                save_dtype,
            )

            print("model saved.")

    def print_train_info(self, args, num_train_epochs, train_dataloader):
        self.accelerator.print("running training / 学習開始")
        self.accelerator.print(
            f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}"
        )
        self.accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
        self.accelerator.print(
            f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}"
        )
        self.accelerator.print(
            f"  total optimization steps / 学習ステップ数: {args.max_train_steps}"
        )

    def checkpointing(
        self, args, global_step, epoch, network, minimum_metadata, metadata, save_dtype
    ):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            ckpt_name = train_util.get_step_ckpt_name(
                args, "." + args.save_model_as, global_step
            )
            self.save_model(
                ckpt_name,
                self.accelerator.unwrap_model(network),
                save_dtype,
            )

            if args.save_state:
                train_util.save_and_remove_state_stepwise(
                    args, self.accelerator, global_step
                )

            remove_step_no = train_util.get_remove_step_no(args, global_step)
            if remove_step_no is not None:
                self.remove_model(
                    train_util.get_step_ckpt_name(
                        args, "." + args.save_model_as, remove_step_no
                    )
                )


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, True)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    add_custom_train_arguments(parser)

    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="do not save metadata in output model / メタデータを出力先モデルに保存しない",
    )
    parser.add_argument(
        "--save_model_as",
        type=str,
        default="safetensors",
        choices=[None, "ckpt", "pt", "safetensors"],
        help="format to save the model (default is .safetensors) / モデル保存時の形式（デフォルトはsafetensors）",
    )

    parser.add_argument(
        "--unet_lr",
        type=float,
        default=None,
        help="learning rate for U-Net / U-Netの学習率",
    )
    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=None,
        help="learning rate for Text Encoder / Text Encoderの学習率",
    )

    parser.add_argument(
        "--network_weights",
        type=str,
        default=None,
        help="pretrained weights for network / 学習するネットワークの初期重み",
    )
    parser.add_argument(
        "--network_module",
        type=str,
        default=None,
        help="network module to train / 学習対象のネットワークのモジュール",
    )
    parser.add_argument(
        "--network_dim",
        type=int,
        default=None,
        help="network dimensions (depends on each network) / モジュールの次元数（ネットワークにより定義は異なります）",
    )
    parser.add_argument(
        "--network_alpha",
        type=float,
        default=1,
        help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version) / LoRaの重み調整のalpha値、デフォルト1（旧バージョンと同じ動作をするにはnetwork_dimと同じ値を指定）",
    )
    parser.add_argument(
        "--network_dropout",
        type=float,
        default=None,
        help="Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons) / 訓練時に毎ステップでニューロンをdropする（0またはNoneはdropoutなし、1は全ニューロンをdropout）",
    )
    parser.add_argument(
        "--network_args",
        type=str,
        default=None,
        nargs="*",
        help="additional arguments for network (key=value) / ネットワークへの追加の引数",
    )
    parser.add_argument(
        "--network_train_unet_only",
        action="store_true",
        help="only training U-Net part / U-Net関連部分のみ学習する",
    )
    parser.add_argument(
        "--network_train_text_encoder_only",
        action="store_true",
        help="only training Text Encoder part / Text Encoder関連部分のみ学習する",
    )
    parser.add_argument(
        "--training_comment",
        type=str,
        default=None,
        help="arbitrary comment string stored in metadata / メタデータに記録する任意のコメント文字列",
    )
    parser.add_argument(
        "--dim_from_weights",
        action="store_true",
        help="automatically determine dim (rank) from network_weights / dim (rank)をnetwork_weightsで指定した重みから自動で決定する",
    )
    parser.add_argument(
        "--scale_weight_norms",
        type=float,
        default=None,
        help="Scale the weight of each key pair to help prevent overtraing via exploding gradients. (1 is a good starting point) / 重みの値をスケーリングして勾配爆発を防ぐ（1が初期値としては適当）",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--base_weights_multiplier",
        type=float,
        default=None,
        nargs="*",
        help="multiplier for network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みの倍率",
    )
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)

    trainer = NetworkTrainer(args)
    trainer.train(args)
