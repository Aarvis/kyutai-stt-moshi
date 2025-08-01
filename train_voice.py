import collections
import dataclasses
import logging
import os
import pprint
import re
import shutil
from contextlib import ExitStack
from pathlib import Path

import fire
import torch.cuda
import torch.distributed as dist
from torch.optim import AdamW, lr_scheduler

# from torch.profiler import ProfilerActivity, profile

from finetune.args import TrainArgs
from finetune.checkpointing import Checkpointer
from finetune.data.data_loader import build_data_loader
from finetune.data.interleaver import InterleavedTokenizer, Interleaver
from finetune.distributed import (
    BACKEND,
    avg_aggregate,
    get_rank,
    get_world_size,
    is_torchrun,
    set_device,
)
from finetune.eval import evaluate
from finetune.loss import compute_loss_with_mask
from finetune.mixed_precision import (
    downcast_mixed_precision,
    prepare_mixed_precision,
    upcast_mixed_precision,
)
from finetune.monitoring.metrics_logger import (
    MetricsLogger,
    eval_log_msg,
    get_eval_logs,
    get_train_logs,
    train_log_msg,
)
from finetune.monitoring.utils import set_logger
from finetune.utils import TrainState, logged_closing, set_random_seed
from finetune.wrapped_model import get_fsdp_model
from moshi.models import loaders
from moshi.models.tts import TTSModel, TokenIds, StateMachine, script_to_entries
from moshi.conditioners.base import TensorCondition, ConditionType

from safetensors.torch import save_file

logger = logging.getLogger("train")


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


def train(config: str):
    args: TrainArgs = TrainArgs.load(config, drop_extra_fields=False)
    set_logger(logging.INFO)

    with ExitStack() as exit_stack:
        _train(args, exit_stack)
    logger.info("Closed everything!")


# Note: this is slow (e.g. takes ~1s per batch of 16 samples with length of 30s).
# It depends only on the text input, but it's currently unnecessarily re-executed
# for each batch supplied by the data loader during the training loop.
# A more efficient approach would be to pretokenize the text/audio and premux
# the text just once, cache the result somewhere, and train multiple steps
# over such prepared dataset.
#
def apply_lookahead_mux(codes, spm, lookahead, frame_rate, token_ids):
    """
    Performs the "lookahead token muxing" on the text token dimension (0)
    of codes representing the ground truth. This emulates the work of
    moshi.models.tts.StateMachine performed at inference time: at any point
    the TTS model uses a combination of two tokens for the next text token
    prediction, one input token is taken from the current word and another
    from the lookahead word (except at the end of sequence).
    """
    T = codes.shape[-1]
    text_tokens = codes[:, 0, :].cpu().detach().numpy()
    texts = [
        spm.decode([token for token in _.tolist() if token != token_ids.new_word])
        for _ in text_tokens
    ]

    for bi, text in enumerate(texts):
        entries = script_to_entries(
            spm,
            token_ids,
            frame_rate,
            [text],
            multi_speaker=True,
            padding_between=1,
        )

        machine = StateMachine(
            token_ids=token_ids,
            second_stream_ahead=lookahead,
            # from TTSModel.from_checkpoint_info:
            max_padding=8,
            initial_padding=2,
        )

        sm_state = machine.new_state(entries)
        muxed_tokens = torch.full(
            (T,), machine.token_ids.pad, dtype=torch.long, device=codes.device
        )
        muxed_tokens[0] = codes[bi, 0, 0]

        mux_offset = 0  # Pointer for the output muxed_tokens tensor
        inp_offset = 0  # Pointer for the input codes ground truth

        while (
            inp_offset < T
            and mux_offset < T - 1
            and (sm_state.end_step is None or mux_offset <= sm_state.end_step)
        ):
            muxed_tokens[mux_offset + 1], consumed_new_word = machine.process(
                mux_offset, sm_state, codes[bi, 0, inp_offset]
            )
            mux_offset += 1

            if consumed_new_word:
                while inp_offset < T and codes[bi, 0, inp_offset] != token_ids.new_word:
                    inp_offset += 1
                inp_offset += 1  # also skip over special new_word token

        codes[bi, 0, :] = muxed_tokens

    return codes


def _train(args: TrainArgs, exit_stack: ExitStack):
    # 1. Initial setup and checks
    set_random_seed(args.seed)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Init NCCL
    if "LOCAL_RANK" in os.environ:
        set_device()
        logger.info("Going to init comms...")

        dist.init_process_group(backend=BACKEND)
    else:
        logger.error(
            "PyTorch environment is not correctly initialized. This message should only be displayed when testing."
        )

    # 2. Init run dir
    main_logger_info(f"Run dir: {args.run_dir}")
    run_dir = Path(args.run_dir)

    if is_torchrun():
        if run_dir.exists() and not args.overwrite_run_dir:
            raise RuntimeError(
                f"Run dir {run_dir} already exists. Make sure to either rename `run_dir` or remove {run_dir}."
            )
        elif run_dir.exists():
            main_logger_info(f"Reusing run dir {run_dir}...")
            # shutil.rmtree(run_dir)

    if args.full_finetuning:
        assert not args.lora.enable, "LoRA should not be enabled for full finetuning."
    else:
        assert args.lora.enable, "LoRA should be enabled for partial finetuning"

    dist.barrier()
    run_dir.mkdir(exist_ok=True, parents=True)

    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        args.save(args_path)

    main_logger_info(f"TrainArgs: {pprint.pformat(dataclasses.asdict(args))}")

    # 3. Get loggers
    metrics_logger: MetricsLogger = MetricsLogger(
        run_dir,
        tag="train",
        is_master=get_rank() == 0,
        wandb_args=args.wandb,
        config=dataclasses.asdict(args),
    )
    exit_stack.enter_context(logged_closing(metrics_logger, "metrics_logger"))

    eval_logger: MetricsLogger = MetricsLogger(
        run_dir,
        tag="eval",
        is_master=get_rank() == 0,
        wandb_args=args.wandb,
        config=dataclasses.asdict(args),
    )
    exit_stack.enter_context(logged_closing(eval_logger, "eval_logger"))

    # 4.1 Load function calling audio encoder and tokenizer
    main_logger_info("Loading Mimi and Moshi...")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        hf_repo=args.moshi_paths.hf_repo_id,
        moshi_weights=args.moshi_paths.moshi_path,
        mimi_weights=args.moshi_paths.mimi_path,
        tokenizer=args.moshi_paths.tokenizer_path,
        config_path=args.moshi_paths.config_path,
    )

    lm_config = (
        loaders._lm_kwargs
        if checkpoint_info.raw_config is None
        else checkpoint_info.raw_config
    )
    lm_config["lora"] = args.lora.enable
    lm_config["lora_rank"] = args.lora.rank
    lm_config["lora_scaling"] = args.lora.scaling

    mimi = checkpoint_info.get_mimi(device="cuda")
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad = False

    # 4.2 Load and shard model, prepare interleaver for audio/text tokens.
    model = get_fsdp_model(args, checkpoint_info)

    # 4.3 Continue from last checkpoint found in run_dir
    def get_last_checkpoint(directory):
        pattern = re.compile(r"^.+@(\d+)\.safetensors$")
        max_index = -1
        last_checkpoint = None

        for filename in os.listdir(run_dir):
            match = pattern.match(filename)
            if match:
                index = int(match.group(1))
                if index > max_index:
                    max_index = index
                    last_checkpoint = filename

        return (max_index, last_checkpoint)

    (init_step, init_checkpoint) = get_last_checkpoint(run_dir)
    assert (
        not init_checkpoint is None
    ), f"need initial voice embedding in {run_dir}/speaker.wav@0.safetensors"

    voices = [f"{run_dir}/{init_checkpoint}"]

    main_logger_info(f"Resuming training from checkpoint {voices[0]}...")

    # fake_tts shim to reuse TTSModel.make_condition_attributes without copying the code or creating a full instance
    # Note: hard-coded cfg_coef and max_speakers in this PoC, taken from moshi.models.tts.TTSModel and tts_pytorch.py
    max_speakers = 5
    cfg_coef = 2.0
    fake_tts = collections.namedtuple(
        "FakeTTSModel", ["max_speakers", "valid_cfg_conditionings"]
    )(max_speakers, [cfg_coef])
    condition_attributes = TTSModel.make_condition_attributes(
        fake_tts, voices, cfg_coef=cfg_coef
    )

    initial_prepared = model.condition_provider.prepare([condition_attributes])
    initial_prepared_tensor = initial_prepared["speaker_wavs"].tensor.detach().clone()
    all_speaker_wavs = torch.nn.Parameter(initial_prepared_tensor, requires_grad=True)

    # Train only the "voice embedding" parameter.
    for p in model.parameters():
        p.requires_grad = False
    for p in model.condition_provider.parameters():
        p.requires_grad = True

    spm = checkpoint_info.get_text_tokenizer()
    audio_delay = checkpoint_info.tts_config.get("audio_delay", 0.0)
    main_logger_info(f"TTS audio_delay: {audio_delay}s")

    interleaver = Interleaver(
        spm,
        mimi.frame_rate,
        model.text_padding_token_id,
        model.end_of_text_padding_id,
        model.zero_token_id,
        keep_main_only=True,
        audio_delay=audio_delay,
    )
    interleaved_tokenizer = InterleavedTokenizer(
        mimi, interleaver, duration_sec=args.duration_sec
    )

    # 5. Load data loaders
    data_loader = build_data_loader(
        instruct_tokenizer=interleaved_tokenizer,
        args=args.data,
        batch_size=args.batch_size,
        seed=args.seed,
        rank=get_rank(),  # DDP rank
        world_size=get_world_size(),  # DDP world_size
        is_eval=False,
    )

    # Debug code to display show relative positions of text/audio tokens from a batch
    #
    # for i in range(1):
    #     batch = next(data_loader)
    #     batch.codes = add_tts_delay_steps(batch.codes, delay_steps, model.text_padding_token_id, model.zero_token_id)
    #     tokens = batch.codes[0, 0, :].flatten().cpu().detach().numpy().tolist()
    #     audio_tokens = batch.codes[0, 1, :].flatten().cpu().detach().numpy().tolist()
    #     words = [spm.decode([_]) for _ in tokens]
    #     logger.info(spm.decode(tokens))
    #     logger.info([_ for _ in zip(tokens, words, audio_tokens)])
    # quit()

    if args.do_eval:
        eval_data_loader = build_data_loader(
            instruct_tokenizer=interleaved_tokenizer,
            args=args.data,
            batch_size=args.batch_size,
            seed=None,
            rank=get_rank(),  # DDP rank
            world_size=get_world_size(),  # DDP world_size
            is_eval=True,
        )

    # 6. Load model
    # Define mixed precision
    param_dtype = getattr(torch, args.param_dtype)
    optim_dtype = torch.float32

    assert args.lora is not None, "`args.lora` should be set to a valid value."

    # 7. Load optimizer
    # Note: Lion seems to converge faster than AdamW, YMMV
    from lion_pytorch import Lion

    # Note: hard-coded rule-of-thumb scaling constant 0.1 for Lion's weight decay
    optimizer = Lion(
        [all_speaker_wavs],
        lr=args.optim.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1 * args.optim.weight_decay,
    )

    # optimizer = AdamW(
    #     [all_speaker_wavs],
    #     lr=args.optim.lr,
    #     betas=(0.9, 0.95),
    #     eps=1e-08,
    #     weight_decay=args.optim.weight_decay,
    # )

    # scheduler = lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=args.optim.lr,
    #     total_steps=args.max_steps,
    #     pct_start=args.optim.pct_start,
    # )

    # Constant LR seems to work ok:
    scheduler = lr_scheduler.ConstantLR(optimizer)

    state = TrainState(args.max_steps)
    state.step = init_step

    # 8. Initialize checkpointer
    if args.do_ckpt:
        checkpointer = Checkpointer(
            model=model,
            state=state,
            config=lm_config,
            run_dir=run_dir,
            optimizer=optimizer,
            num_ckpt_keep=args.num_ckpt_keep,
            full_finetuning=args.full_finetuning,
        )
    # 9. Prepare mixed precision
    prepare_mixed_precision(
        model.parameters(), param_dtype=param_dtype, optim_dtype=optim_dtype
    )

    # 11. train!
    model.train()
    torch.cuda.empty_cache()

    token_ids = TokenIds(model.text_card + 1)

    while state.step < args.max_steps:
        state.start_step()
        is_last_step = state.step == args.max_steps

        optimizer.zero_grad()

        loss = torch.tensor([0.0], device="cuda")
        n_batch_tokens: int = 0
        n_real_tokens: int = 0

        for i in range(args.num_microbatches):
            batch = next(data_loader)
            codes = batch.codes

            # Note: training of a voice embedding also seems to work "ok" without apply_lookahead_mux,
            # which may slow it down considerably depending on the training dataset's size.
            # However, it is included here to get a closer match to the inputs that the model works
            # with during inference.
            codes = apply_lookahead_mux(
                codes,
                spm,
                checkpoint_info.tts_config["second_stream_ahead"],
                mimi.frame_rate,
                token_ids,
            )

            prepared = initial_prepared.copy()
            prepared["speaker_wavs"] = TensorCondition(
                all_speaker_wavs.to(dtype=param_dtype),
                initial_prepared["speaker_wavs"].mask.detach().clone(),
            )

            condition_tensors = model.condition_provider(prepared)

            # forward / backward
            output = model(codes=codes, condition_tensors=condition_tensors)
            # text_loss = compute_loss_with_mask(
            #     output.text_logits,
            #     codes[:, : model.audio_offset],
            #     output.text_mask,
            #     mode="text",
            #     text_padding_weight=args.text_padding_weight,
            #     text_padding_ids={
            #         model.text_padding_token_id,
            #         model.end_of_text_padding_id,
            #     },
            # )
            audio_loss = compute_loss_with_mask(
                output.logits,
                codes[:, model.audio_offset : model.audio_offset + model.dep_q],
                output.mask,
                mode="audio",
                first_codebook_weight_multiplier=args.first_codebook_weight_multiplier,
            )

            # mb_loss = text_loss + audio_loss
            mb_loss = audio_loss
            mb_loss.backward()

            loss += mb_loss.detach()
            # n_batch_tokens += output.text_mask.numel() + output.mask.numel()
            n_batch_tokens += output.text_mask.numel()
            n_real_tokens += torch.sum(
                output.text_mask
            ).item()  # + torch.sum(output.mask).item()

            if i < args.num_microbatches - 1:
                # synchronize CUDA to re-run backward
                assert args.num_microbatches > 1  # should not happen
                torch.cuda.synchronize()

        if args.num_microbatches > 1:
            loss /= args.num_microbatches
            for p in model.parameters():
                if p.requires_grad:
                    assert p.grad is not None
                    p.grad.div_(args.num_microbatches)

        # upcast params for optimizer update
        upcast_mixed_precision(model.parameters(), optim_dtype=optim_dtype)

        # clip grad norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

        # optimizer step
        optimizer.step()

        # downcast params for forward & backward
        downcast_mixed_precision(model.parameters(), param_dtype=param_dtype)

        last_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # Host sync
        loss_item = loss.item()
        avg_loss = avg_aggregate(loss_item)

        if args.do_eval and (
            (args.eval_freq > 0 and state.step % args.eval_freq == 0) or is_last_step
        ):
            # write perplexity to state
            evaluate(model, eval_data_loader, state, args)

            eval_logs = get_eval_logs(
                state.step,
                avg_loss,
                state.this_eval_perplexity,
                state.this_eval_loss,
            )

            main_logger_info(eval_log_msg(eval_logs))
            eval_logger.log(eval_logs, step=state.step)

        # Timing
        state.end_step(n_batch_tokens)

        if state.step % args.log_freq == 0:
            train_logs = get_train_logs(
                state,
                avg_loss,
                n_real_tokens,
                last_lr,
                torch.cuda.max_memory_allocated(),
                torch.cuda.memory_allocated(),
                args,
            )
            main_logger_info(train_log_msg(state, logs=train_logs, loss=avg_loss))
            metrics_logger.log(train_logs, step=state.step)

        if args.do_ckpt and (
            (args.ckpt_freq > 0 and state.step % args.ckpt_freq == 0) or is_last_step
        ):
            # note: for now assume there's just one speaker in the training data (hence speaker_idx = 0)
            speaker_idx = 0
            speaker_dim = all_speaker_wavs.shape[1] // max_speakers
            speaker_offset = speaker_idx * speaker_dim
            speaker_wavs = (
                all_speaker_wavs[:, speaker_offset : speaker_offset + speaker_dim, :]
                .clone()
                .detach()
                .transpose(1, 2)
                .contiguous()
            )
            save_file(
                {"speaker_wavs": speaker_wavs},
                f"{run_dir}/speaker.wav@{state.step}.safetensors",
            )

            # note: Checkpointer doesn't currently support dumping speaker_wavs, so we use the code above instead
            # checkpointer.save_checkpoint(
            #     save_only_lora=not args.full_finetuning and args.save_adapters,
            #     dtype=param_dtype,
            # )

    main_logger_info("done!")


if __name__ == "__main__":
    """See README.md for usage."""
    fire.Fire(train)
