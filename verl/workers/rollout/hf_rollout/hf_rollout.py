# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single
GPU model. Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model
to perform generation.
"""

import contextlib

import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig

from verl import DataProto
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.torch_functional import get_response_mask

from .base import BaseRollout

__all__ = ["HFRollout", "ServerAdapter"]


class HFRollout(BaseRollout):
    def __init__(self, module: nn.Module, config):
        super().__init__()
        self.config = config
        self.module = module

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        batch_size = prompts.batch.batch_size[0]
        num_chunks = max(batch_size // self.config.get("micro_batch_size", batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
        return output

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        # make sampling args can be overridden by inputs
        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        is_validate = prompts.meta_info.get("validate", False)

        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = max(0, prompts.meta_info.get("top_k", self.config.get("top_k", 0)))  # to be compatible with vllm

        if not do_sample:
            # do_sample==False -> greedy decoding
            kwargs = {
                "do_sample": False,
                "num_beams": 1,
            }
        elif is_validate:
            # do validate and do sample -> use val_kwargs
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_k": max(0, self.config.val_kwargs.top_k),  # to be compatible with vllm
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "num_return_sequences": 1,  # if validate, already repeat in ray_trainer
            }
        else:
            # do_sample -> use rollout config
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                # already repeat in ray_trainer
                # https://github.com/verl-project/verl/blob/2fdfbdcba6f2e076f64bc47922d8fe6cf7dc7da5/verl/trainer/ppo/ray_trainer.py#L1117
                "num_return_sequences": 1,
            }

        # make config according to generate mode
        generation_config = GenerationConfig(**kwargs)

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        prompt_length = idx.size(1)
        attention_mask = prompts.batch["attention_mask"]  # left-padded attention_mask
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]

        self.module.eval()
        param_ctx = contextlib.nullcontext()

        if isinstance(self.module, FSDP):
            # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        with param_ctx, torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            output = self.module.generate(
                input_ids=idx,
                attention_mask=attention_mask,
                position_ids=position_ids,
                do_sample=do_sample,
                max_new_tokens=response_length,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                generation_config=generation_config,
                output_scores=False,  # this is potentially very large
                return_dict_in_generate=True,
                use_cache=True,
            )

        # TODO: filter out the seq with no answers like ds-chat
        seq = output.sequences
        generated_batch_size = seq.size(0)  # bs * num_return_sequences

        # huggingface generate will stop generating when all the batch reaches [EOS].
        # We have to pad to response_length
        sequence_length = prompt_length + self.config.response_length
        delta_length = sequence_length - seq.shape[1]

        if delta_length > 0:
            delta_tokens = torch.ones(size=(generated_batch_size, delta_length), device=seq.device, dtype=seq.dtype)
            delta_tokens = pad_token_id * delta_tokens
            seq = torch.cat((seq, delta_tokens), dim=1)
        assert seq.shape[1] == sequence_length

        # make necessary reputations if num_return_sequences > 1
        num_return_sequences = kwargs.get("num_return_sequences", 1)
        if num_return_sequences > 1:
            position_ids = position_ids.repeat_interleave(num_return_sequences, dim=0)
            attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)

        prompt = seq[:, :prompt_length]  # (generated_batch_size, prompt_length)
        response = seq[:, prompt_length:]  # (generated_batch_size, response_length)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(generated_batch_size, 1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": prompt,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=generated_batch_size,
        )

        # empty cache before compute old_log_prob
        get_torch_device().empty_cache()

        self.module.train()
        return DataProto(batch=batch)


class ServerAdapter(BaseRollout):
    """HF rollout adapter for verl's async rollout mode.

    vLLM/SGLang/TRT-LLM front a separate inference server, so their async
    ``ServerAdapter`` classes are clients that talk to that server. The HF
    rollout instead runs ``transformers`` generation in-process, so this
    adapter holds the (single-device) model and implements the weight-sync
    contract the async trainer calls (``resume`` / ``update_weights`` /
    ``release``) by loading the named tensors straight into the held model.

    This resolves ``Rollout hf with mode async not found`` on the weight-sync
    side. Generating in async mode still requires an HF server behind the
    LLM-server client (see ``verl/workers/rollout/llm_server.py``); this
    adapter covers the weight-sync contract.
    """

    def __init__(self, config, model_config, device_mesh, *args, **kwargs):
        super().__init__(config, model_config, device_mesh)
        self.device_name = get_device_name()
        self.device = torch.device(self.device_name) if self.device_name != "cpu" else torch.device("cpu")
        self.module, self.tokenizer, self.processor = self._load_model()
        self.module.eval()

    def _load_model(self):
        """Load a single-device HF model (+ LoRA) for in-process generation."""
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoProcessor,
            AutoTokenizer,
        )

        cfg = AutoConfig.from_pretrained(
            self.model_config.local_path, trust_remote_code=self.model_config.trust_remote_code
        )
        is_vlm = getattr(cfg, "vision_config", None) is not None
        auto_cls = AutoModelForImageTextToText if is_vlm else AutoModelForCausalLM
        model = auto_cls.from_pretrained(
            self.model_config.local_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=self.model_config.trust_remote_code,
        )

        if getattr(self.model_config, "lora_rank", 0) > 0:
            from peft import LoraConfig, TaskType, get_peft_model

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.model_config.lora_rank,
                lora_alpha=self.model_config.lora_alpha,
                target_modules=self.model_config.target_modules,
                exclude_modules=self.model_config.exclude_modules,
                bias="none",
            )
            model = get_peft_model(model, lora_config)

        model = model.to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.local_path, trust_remote_code=self.model_config.trust_remote_code
        )
        processor = (
            AutoProcessor.from_pretrained(
                self.model_config.local_path, trust_remote_code=self.model_config.trust_remote_code
            )
            if is_vlm
            else None
        )
        return model, tokenizer, processor

    async def resume(self, tags: list[str]):
        # In-process model: nothing to wake. Keep the cache warm.
        return

    async def release(self):
        # In-process model: drop the cache so the next stage can reuse device memory.
        try:
            get_torch_device().empty_cache()
        except Exception:
            pass

    async def update_weights(
        self,
        weights,
        global_steps=None,
        wire_format="named_tensors",
        **kwargs,
    ):
        """Load the trainer's named tensors into the held model."""
        assert wire_format == "named_tensors", (
            f"HF rollout supports wire_format='named_tensors'; got {wire_format!r}"
        )
        state_dict = {}
        for name, tensor in weights:
            state_dict[name] = tensor.to(self.device, dtype=self.module.dtype)

        missing, _ = self.module.load_state_dict(state_dict, strict=False)
        # The actor may emit "model.<name>" depending on how it wraps the module.
        if missing:
            stripped = {
                (k[len("model.") :] if k.startswith("model.") else k): v for k, v in state_dict.items()
            }
            self.module.load_state_dict(stripped, strict=False)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Synchronous generation is handled by ``HFRollout``, not this adapter.

        In async mode, generation goes through the LLM-server client
        (``verl/workers/rollout/llm_server.py``) talking to an HF server; this
        adapter only implements the weight-sync contract.
        """
        raise NotImplementedError(
            "HFRollout ServerAdapter does not support synchronous generate_sequences. "
            "Use HFRollout for sync generation, or the async server interface for async generation."
        )
