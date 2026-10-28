# Copyright 2026 HolboxAI
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
"""Async HF rollout server for verl.

verl's async rollout is client-server: ``LLMServerManager`` spawns ``RolloutReplica``
actors, ``ServerAdapter`` syncs weights into them, and the agent loop generates
against them through ``LLMServerClient.generate``. vLLM/SGLang/TRT-LLM front an
inference engine; this module implements the same server contract with an
in-process ``transformers`` model so ``rollout.name=hf`` works in async mode.

Only a single-node, single-process server is supported (``tensor_model_parallel_size``
/ ``data_parallel_size`` / ``pipeline_model_parallel_size`` must all be 1). Multi-GPU
HF serving is out of scope; use vLLM/SGLang for that.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import ray
import torch

from verl.plugin.platform import get_platform
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_name, get_torch_device
from verl.workers.config.model import HFModelConfig
from verl.workers.rollout.replica import RolloutReplica, TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel("WARN")


class HFHttpServer:
    """Ray actor that holds a single-device HF model and generates in-process.

    Implements the subset of ``vLLMHttpServer``'s contract that the async
    trainer calls: ``generate`` for token generation, ``update_weights_from_ipc``
    for weight sync, and the lifecycle no-ops (``wake_up``/``sleep``/
    ``clear_kv_cache``/``abort_all_requests``/``resume_generation``).
    """

    def __init__(
        self,
        config,
        model_config,
        rollout_mode,
        workers,
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
        cuda_visible_devices: Optional[str] = None,
    ):
        self.config = config
        self.model_config = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
        self.rollout_mode = rollout_mode
        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.device_name = get_device_name()
        self.device = torch.device(self.device_name) if self.device_name != "cpu" else torch.device("cpu")
        self.module, self.tokenizer, self.processor = self._load_model()
        self.module.eval()

    def _load_model(self):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

        cfg = AutoConfig.from_pretrained(self.model_config.local_path, trust_remote_code=self.model_config.trust_remote_code)
        is_vlm = getattr(cfg, "vision_config", None) is not None
        auto_cls = AutoModelForImageTextToText if is_vlm else AutoModelForCausalLM
        model = auto_cls.from_pretrained(
            self.model_config.local_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=self.model_config.trust_remote_code,
        )
        if getattr(self.model_config, "lora_rank", 0) > 0:
            from peft import LoraConfig, TaskType, get_peft_model

            model = get_peft_model(
                model,
                LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=self.model_config.lora_rank,
                    lora_alpha=self.model_config.lora_alpha,
                    target_modules=self.model_config.target_modules,
                    exclude_modules=self.model_config.exclude_modules,
                    bias="none",
                ),
            )
        model = model.to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_config.local_path, trust_remote_code=self.model_config.trust_remote_code)
        processor = AutoProcessor.from_pretrained(self.model_config.local_path, trust_remote_code=self.model_config.trust_remote_code) if is_vlm else None
        return model, tokenizer, processor

    # ------------------------------------------------------------------ launch
    async def get_master_address(self):
        import socket

        return socket.gethostbyname(socket.gethostname()), 0, 0

    async def launch_server(self, master_address: str = None, master_port: int = None, dp_rpc_port: int = None):
        # Single-node in-process server: nothing to bind.
        return

    async def get_server_address(self):
        import socket

        return socket.gethostbyname(socket.gethostname()), 0

    # -------------------------------------------------------------- generation
    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        priority: int = 0,
        kv_transfer_params: Optional[dict] = None,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> TokenOutput:
        input_ids = torch.tensor([prompt_ids], device=self.device)
        attention_mask = torch.ones_like(input_ids)

        gen_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": int(sampling_params.get("max_tokens", 256)),
            "do_sample": float(sampling_params.get("temperature", 0.0)) > 0.0,
            "temperature": float(sampling_params.get("temperature", 1.0)),
            "top_p": float(sampling_params.get("top_p", 1.0)),
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if sampling_params.get("top_k") is not None and int(sampling_params.get("top_k", -1)) > 0:
            gen_kwargs["top_k"] = int(sampling_params["top_k"])

        # Multimodal (vision) inputs, if the model and request carry images.
        if image_data is not None and self.processor is not None:
            pixel_values, image_grid_thw = self._process_images(image_data)
            if pixel_values is not None:
                gen_kwargs["pixel_values"] = pixel_values
                gen_kwargs["image_grid_thw"] = image_grid_thw

        with torch.no_grad():
            output = self.module.generate(**gen_kwargs)

        seq = output[0]
        prompt_len = len(prompt_ids)
        response_ids = seq[prompt_len:].tolist()
        stop_reason = "completed" if self.tokenizer.eos_token_id in response_ids else None
        # Truncate at the first EOS, like vLLM does.
        if self.tokenizer.eos_token_id in response_ids:
            response_ids = response_ids[: response_ids.index(self.tokenizer.eos_token_id)]

        return TokenOutput(token_ids=response_ids, stop_reason=stop_reason)

    def _process_images(self, image_data):
        # image_data is a list of images (PIL/numpy/bytes). Let the checkpoint's
        # processor build the vision tensors; fall back to None on failure.
        try:
            import numpy as np
            from PIL import Image

            images = []
            for img in image_data:
                if isinstance(img, Image.Image):
                    images.append(img)
                elif isinstance(img, np.ndarray):
                    images.append(Image.fromarray(img))
                else:
                    images.append(img)
            if not images:
                return None, None
            proc = self.processor
            out = proc(images=images, return_tensors="pt")
            pixel_values = out.get("pixel_values")
            image_grid_thw = out.get("image_grid_thw")
            if pixel_values is not None:
                pixel_values = pixel_values.to(self.device, dtype=torch.bfloat16)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(self.device)
            return pixel_values, image_grid_thw
        except Exception as e:  # noqa: BLE001
            logger.warning("HF server image processing failed, running text-only: %s", e)
            return None, None

    # -------------------------------------------------------------- weight sync
    async def update_weights(self, weights: dict[str, torch.Tensor], **kwargs):
        """Load a materialized state_dict into the held model."""
        state_dict = {name: tensor.to(self.device, dtype=self.module.dtype) for name, tensor in weights.items()}
        missing, _ = self.module.load_state_dict(state_dict, strict=False)
        if missing:
            stripped = {(k[len("model.") :] if k.startswith("model.") else k): v for k, v in state_dict.items()}
            self.module.load_state_dict(stripped, strict=False)

    # ---------------------------------------------------------------- lifecycle
    async def wake_up(self, tags: list[str] | None = None):
        return

    async def sleep(self):
        return

    async def clear_kv_cache(self):
        try:
            get_torch_device().empty_cache()
        except Exception:
            pass

    async def abort_all_requests(self, reject_request: bool = False):
        return

    async def resume_generation(self):
        return


class HFReplica(RolloutReplica):
    """Single-node HF rollout replica.

    HF generation is single-device and in-process, so only world_size == 1 is
    supported (no tensor/data/pipeline parallel).
    """

    def __init__(
        self,
        replica_rank: int,
        config,
        model_config,
        gpus_per_node: int = 1,
        is_reward_model: bool = False,
        is_teacher_model: bool = False,
        name_suffix: str = "",
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model, is_teacher_model, name_suffix)
        assert self.world_size == 1, "HF rollout supports a single device only (world_size must be 1)"
        self.server_class = ray.remote(HFHttpServer)

    async def launch_servers(self):
        assert len(self.workers) >= 1, "HF replica requires at least one worker"

        worker_infos = await asyncio.gather(
            *[worker.__ray_call__.remote(lambda self: ray.get_runtime_context().get_node_id()) for worker in self.workers]
        )
        node_id = worker_infos[0]
        prefix = self._get_server_name_prefix()
        name = f"{prefix}server_{self.replica_rank}_0{self.name_suffix}"
        env_vars = {
            **{var: "1" for var in get_platform().ray_noset_envvars()},
            **get_platform().rollout_env_vars(),
        }

        server = self.server_class.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
            runtime_env={"env_vars": env_vars},
            name=name,
            max_concurrency=self.config.ray_actor_max_concurrency,
        ).remote(
            config=self.config,
            model_config=self.model_config,
            rollout_mode=self.rollout_mode,
            workers=self.workers,
            replica_rank=self.replica_rank,
            node_rank=0,
            gpus_per_node=1,
            nnodes=1,
        )
        self.servers.append(server)

        server_address, _ = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = f"[{server_address}]:0"

    def _get_server_name_prefix(self) -> str:
        return "hf_"
