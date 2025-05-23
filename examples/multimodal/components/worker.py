# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import os
import signal
import base64
from typing import Optional

import torch
import httpx
import requests
from PIL import Image
from io import BytesIO
from components.disagg_router import PyDisaggregatedRouter
from components.prefill_worker import PrefillWorker
from transformers import AutoImageProcessor, LlavaForConditionalGeneration
from utils.logging import check_required_workers
from utils.nixl import NixlMetadataStore
from utils.prefill_queue import PrefillQueue
from utils.protocol import (
    EncodeRequest,
    EncodeResponse,
    MyRequestOutput,
    vLLMMultimodalRequest,
)
from utils.vllm import parse_vllm_args
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.inputs.data import TokensPrompt
from vllm.remote_prefill import RemotePrefillParams, RemotePrefillRequest
from vllm.sampling_params import RequestOutputKind

from dynamo.sdk import async_on_start, depends, dynamo_context, endpoint, service

logger = logging.getLogger(__name__)


@service(
    dynamo={
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class VllmWorker:
    # For disaggregated serving, we need to link the prefill worker to the vllm worker
    prefill_worker = depends(PrefillWorker)

    def __init__(self):
        self._http_client: Optional[httpx.AsyncClient] = None
        self._http_timeout = 30.0
        self.client = None
        self.min_workers = 1
        self.disaggregated_router: Optional[PyDisaggregatedRouter] = None
        class_name = self.__class__.__name__
        self.engine_args = parse_vllm_args(class_name, "")
        self.do_remote_prefill = self.engine_args.remote_prefill
        self.model_name = (
            self.engine_args.served_model_name
            if self.engine_args.served_model_name is not None
            else "vllm"
        )
        self._prefill_queue_nats_server = os.getenv(
            "NATS_SERVER", "nats://localhost:4222"
        )
        self._prefill_queue_stream_name = self.model_name
        logger.info(
            f"Prefill queue: {self._prefill_queue_nats_server}:{self._prefill_queue_stream_name}"
        )

        if self.engine_args.remote_prefill:
            if self.engine_args.enable_chunked_prefill is not False:
                logger.info("Chunked prefill is not supported yet, setting to False")
                self.engine_args.enable_chunked_prefill = False

            if self.engine_args.preemption_mode != "swap":
                logger.info("Preemption mode is not supported yet, setting to swap")
                self.engine_args.preemption_mode = "swap"

            if self.engine_args.pipeline_parallel_size != 1:
                logger.info("Pipeline parallel size is not supported yet, setting to 1")
                self.engine_args.pipeline_parallel_size = 1

        if self.engine_args.router == "kv":
            raise NotImplementedError(
                "Multimodal requests are not supported for kv router mode"
            )

        signal.signal(signal.SIGTERM, self.shutdown_vllm_engine)
        signal.signal(signal.SIGINT, self.shutdown_vllm_engine)

    @async_on_start
    async def async_init(self):
        self._engine_context = build_async_engine_client_from_engine_args(
            self.engine_args
        )
        if self._engine_context is not None:
            self.engine_client = await self._engine_context.__aenter__()
        else:
            raise RuntimeError("Failed to initialize engine client")

        if self.engine_args.router == "kv":
            raise NotImplementedError(
                "Multimodal requests are not supported for kv router mode"
            )

        runtime = dynamo_context["runtime"]

        if self.do_remote_prefill:
            metadata = self.engine_client.nixl_metadata
            metadata_store = NixlMetadataStore("dynamo", runtime)
            await metadata_store.put(metadata.engine_id, metadata)

            if self.engine_args.conditional_disagg:
                self.disaggregated_router = PyDisaggregatedRouter(
                    runtime,
                    self.model_name,
                    max_local_prefill_length=self.engine_args.max_local_prefill_length,
                    max_prefill_queue_size=self.engine_args.max_prefill_queue_size,
                )
                await self.disaggregated_router.async_init()
            else:
                self.disaggregated_router = None

        else:
            self.disaggregated_router = None
            # For encoding
            self.MODEL_ID = self.engine_args.model

            self.image_processor = AutoImageProcessor.from_pretrained(
                self.MODEL_ID, trust_remote_code=True
            )

            self.vision_model = LlavaForConditionalGeneration.from_pretrained(
                self.MODEL_ID, device_map="auto", torch_dtype=torch.float16
            ).eval()

        # Initialize HTTP client with default limits
        self._http_client = httpx.AsyncClient(timeout=self._http_timeout)

        logger.info("VllmWorker has been initialized")

    def shutdown_vllm_engine(self, signum, frame):
        """Shutdown the background loop"""
        logger.info(f"Received signal {signum}, shutting down")
        loop = asyncio.get_event_loop()
        try:
            if self._http_client:
                loop.run_until_complete(self._http_client.aclose())
            self.engine_client.close()
            logger.info("VllmWorker shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            loop.stop()

    def get_remote_prefill_request_callback(self):
        async def callback(request: RemotePrefillRequest):
            async with PrefillQueue.get_instance(
                nats_server=self._prefill_queue_nats_server,
                stream_name=self._prefill_queue_stream_name,
            ) as prefill_queue:
                await prefill_queue.enqueue_prefill_request(request)

        return callback
    
    async def load_image(self, image_url: str) -> Image.Image:
        try:
            if image_url.startswith("data:image/"):
                # Remove the data URL prefix to get just the base64 string
                base64_data = image_url.split(",", 1)[1]
                try:
                    image_bytes = base64.b64decode(base64_data)
                    image_data = BytesIO(image_bytes)
                except base64.binascii.Error as e:
                    raise ValueError(f"Invalid base64 encoding: {e}")
            elif image_url.startswith(("http://", "https://")):
                if not self._http_client:
                    raise RuntimeError("HTTP client not initialized")
                    
                response = await self._http_client.get(image_url)
                response.raise_for_status()

                if not response.content:
                    raise ValueError("Empty response content from image URL")
                    
                image_data = BytesIO(response.content)
            else:
                raise ValueError(f"Invalid image source: {image_url}")

            # PIL is sync, so offload to a thread to avoid blocking the event loop
            image = await asyncio.to_thread(Image.open, image_data)
            
            # Validate image format and convert to RGB
            if image.format not in ('JPEG', 'PNG', 'WEBP'):
                raise ValueError(f"Unsupported image format: {image.format}")
                
            image = image.convert("RGB")
                
            return image
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error loading image: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise ValueError(f"Failed to load image: {e}")

    @endpoint()
    async def generate(self, request: vLLMMultimodalRequest):
        if request.image_url is None:
            raise ValueError("Image URL is required for multimodal requests")

        if self.do_remote_prefill:
            if self.disaggregated_router is not None:
                async with PrefillQueue.get_instance(
                    nats_server=self._prefill_queue_nats_server,
                    stream_name=self._prefill_queue_stream_name,
                ) as prefill_queue:
                    prefill_queue_size = await prefill_queue.get_queue_size()
                disagg_router_decision = await self.disaggregated_router.prefill_remote(
                    len(request.engine_prompt["prompt_token_ids"]),
                    request.prefix_hit_rate,
                    prefill_queue_size,
                )
            else:
                # always prefill remotely if no disaggregated router is provided
                disagg_router_decision = True

            if self.do_remote_prefill and disagg_router_decision:
                remote_prefill_params = RemotePrefillParams(
                    is_remote_prefill=True,
                    remote_prefill_request_callback=self.get_remote_prefill_request_callback(),
                    # Pass the image url as part of the RemotePrefillParams, which will be passed to the prefill worker via RemotePrefillRequest
                    multimodal_data_source={
                        "image_url": request.image_url,
                    },
                )
                logger.info(
                    f"Prefilling remotely for request {request.request_id} with length {len(request.engine_prompt['prompt_token_ids'])}"
                )
            else:
                remote_prefill_params = None
                logger.info(
                    f"Prefilling locally for request {request.request_id} with length {len(request.engine_prompt['prompt_token_ids'])}"
                )

            # The decode worker will pre-allocate the memory based on the prompt token length for the prefill worker to transfer the kv cache.
            # As a workaround, here we manually insert some placeholder dummy tokens based on the embedding size
            # so that decode worker can pre-allocate the memory with the correct size.
            # The structure of the prompt will be like: "\nUSER: <image> <dummy_tokens>\n<user_prompt>\nASSISTANT:".
            # Since the "<image>" token is included in the prompt, only need to insert (embedding_size - 1) dummy tokens after the image token.
            IMAGE_TOKEN_ID = 32000
            DUMMY_TOKEN_ID = 0
            # Find the index of the image token in the prompt token ids
            image_token_index = request.engine_prompt["prompt_token_ids"].index(
                IMAGE_TOKEN_ID
            )
            dummy_token_index = image_token_index + 1
            prompt_ids = (
                request.engine_prompt["prompt_token_ids"][:dummy_token_index]
                + [DUMMY_TOKEN_ID] * (self.embedding_size - 1)
                + request.engine_prompt["prompt_token_ids"][dummy_token_index:]
            )
        else:
            remote_prefill_params = None

        # rust HTTP requires Delta streaming
        request.sampling_params.output_kind = RequestOutputKind.DELTA

        # If remote prefill is not used, we need to get the image data and pass it to the engine
        if remote_prefill_params is None:
            image_url = request.image_url
            try:
                image_data = await self.load_image(request.image_url)

                image_embeds = self.image_processor(images=image_data, return_tensors="pt")

                with torch.no_grad():
                    logger.debug(f"Vision model device: {self.vision_model.device}")
                    vision_outputs = self.vision_model.vision_tower(
                        image_embeds["pixel_values"].to(self.vision_model.device)
                    )

                    image_features = vision_outputs.last_hidden_state
                    image_features = self.vision_model.multi_modal_projector(image_features)

            except Exception as e:
                logger.error(f"Failed to process image: {e}")
                raise ValueError(f"Failed to process image: {e}")

            multi_modal_data = {"image": image_features}
            logger.info(
                f"Prefilling locally for request {request.request_id} with length {len(request.engine_prompt['prompt_token_ids'])}"
            )
            prompt_ids = request.engine_prompt["prompt_token_ids"]
        else:
            multi_modal_data = None

        async for response in self.engine_client.generate(
            prompt=TokensPrompt(
                prompt_token_ids=prompt_ids,
                multi_modal_data=multi_modal_data,
            ),
            sampling_params=request.sampling_params,
            request_id=request.request_id,
            remote_prefill_params=remote_prefill_params,
        ):
            yield MyRequestOutput(
                request_id=response.request_id,
                prompt=response.prompt,
                prompt_token_ids=response.prompt_token_ids,
                prompt_logprobs=response.prompt_logprobs,
                outputs=response.outputs,
                finished=response.finished,
            ).model_dump_json()
