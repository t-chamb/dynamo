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
import uuid

from common.base_engine import BaseTensorrtLLMEngine
from common.parser import parse_tensorrt_llm_args
from common.protocol import (
    PreprocessedRequest,
    Tokens,
    TRTLLMWorkerRequest,
    TRTLLMWorkerResponse,
)
from common.utils import ServerType
from components.prefill_worker import TensorRTLLMPrefillWorker

from dynamo.llm import ModelType, register_llm
from dynamo.sdk import async_on_start, depends, dynamo_context, dynamo_endpoint, service
from dynamo.sdk.lib.config import ServiceConfig

logger = logging.getLogger(__name__)


def convert_to_trtllm_worker_request(request: PreprocessedRequest, model_name: str):
    # Convert to TRTLLMWorkerRequest
    sampling_params = {
        "max_tokens": request.stop_conditions.max_tokens,
        "stop": request.stop_conditions.stop,
        "stop_token_ids": request.stop_conditions.stop_token_ids_hidden,
        "temperature": request.sampling_options.temperature,
        "top_p": request.sampling_options.top_p,
        "top_k": request.sampling_options.top_k,
        "repetition_penalty": request.sampling_options.repetition_penalty,
        "presence_penalty": request.sampling_options.presence_penalty,
        "frequency_penalty": request.sampling_options.frequency_penalty,
        "min_p": request.sampling_options.min_p,
        "seed": request.sampling_options.seed,
        "ignore_eos": request.stop_conditions.ignore_eos,
        "use_beam_search": request.sampling_options.use_beam_search,
        "length_penalty": request.sampling_options.length_penalty,
        "min_tokens": request.stop_conditions.min_tokens,
    }

    # Remove None values
    sampling_params = {k: v for k, v in sampling_params.items() if v is not None}

    return TRTLLMWorkerRequest(
        # TODO: served model name?
        # model=self.engine_config.model_name,
        model=model_name,
        id=str(uuid.uuid4()),
        sampling_params=sampling_params,
        streaming=True,
        tokens=Tokens(tokens=request.token_ids),
    )


@service(
    dynamo={
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class TensorRTLLMWorker(BaseTensorrtLLMEngine):
    prefill_worker = depends(TensorRTLLMPrefillWorker)

    def __init__(self):
        logger.info("Initializing TensorRT-LLM Worker")
        class_name = self.__class__.__name__
        config = ServiceConfig.get_instance()
        config_args = config.as_args(class_name, prefix="")
        args, self.engine_config = parse_tensorrt_llm_args(config_args)
        worker_id = dynamo_context["endpoints"][0].lease_id()
        self._min_prefill_workers = args.min_prefill_workers
        super().__init__(
            namespace_str="dynamo",
            component_str=class_name,
            worker_id=worker_id,
            engine_config=self.engine_config,
            remote_prefill=args.remote_prefill,
            min_workers=args.min_workers,
            disagg_config_file=args.llmapi_disaggregated_config,
            block_size=args.block_size,
            router=args.router,
            server_type=ServerType.GEN,
        )

    @async_on_start
    async def async_init(self):
        logger.info("Async Initializing TensorRT-LLM Worker engine...")
        self._init_engine()

        runtime = dynamo_context["runtime"]
        comp_ns, comp_name = TensorRTLLMWorker.dynamo_address()  # type: ignore

        # Register an endpoint that accepts pre-processed requests for integration
        # with dynamo-run in=http out=dyn://{endpoint}
        endpoint_str = "generate_preprocessed"
        endpoint = (
            runtime.namespace(comp_ns).component(comp_name).endpoint(endpoint_str)
        )
        logger.info(
            f"Registering LLM for discovery at endpoint: {comp_ns}/{comp_name}/{endpoint_str} ..."
        )
        await register_llm(
            ModelType.Backend,
            endpoint,
            self.engine_config.model_name,
            # TODO: served model name?
            self.engine_config.model_name,
        )
        logger.info(
            f"Registered LLM for discovery at endpoint: {comp_ns}/{comp_name}/{endpoint_str} for model_name: {self.engine_config.model_name} ..."
        )

        if self._remote_prefill:
            prefill_comp_ns, prefill_comp_name = TensorRTLLMPrefillWorker.dynamo_address()  # type: ignore
            # Wait for prefill workers
            self._prefill_client = (
                await runtime.namespace(prefill_comp_ns)
                .component(prefill_comp_name)
                .endpoint("generate")
                .client()
            )
            while len(self._prefill_client.endpoint_ids()) < self._min_prefill_workers:
                logger.info(
                    f"Waiting for prefill workers to be ready.\n"
                    f" Current: {len(self._prefill_client.endpoint_ids())},"
                    f" Required: {self._min_prefill_workers}"
                )
                await asyncio.sleep(10)

        if self._kv_metrics_publisher is not None:
            task = asyncio.create_task(self.create_metrics_publisher_endpoint())
            task.add_done_callback(
                lambda _: logger.info("metrics publisher endpoint created")
            )

        logger.info("TensorRT-LLM Worker initialized")

    async def create_metrics_publisher_endpoint(self):
        component = dynamo_context["component"]
        await self._kv_metrics_publisher.create_endpoint(component)

    @dynamo_endpoint()
    async def generate_preprocessed(self, request: PreprocessedRequest):
        trtllm_request = convert_to_trtllm_worker_request(
            request, self.engine_config.model_name
        )

        # Call generate on internal engine
        async for response_str in super().generate(trtllm_request):
            trt_response = TRTLLMWorkerResponse.model_validate_json(response_str)
            # Every chunk returns all of the output tokens seen so far, iteratively
            # appending new tokens to the end of the returned token_ids. For
            # efficiency, just return the net new tokens (delta).
            outputs = trt_response.outputs[0]
            new_token_start_idx = outputs["_last_token_ids_len"]
            new_tokens = outputs["token_ids"][new_token_start_idx:]
            response = {"token_ids": new_tokens}
            finish_reason = outputs.get("finish_reason")
            if finish_reason:
                response["finish_reason"] = finish_reason
            yield response

    @dynamo_endpoint()
    async def generate(self, request: TRTLLMWorkerRequest):
        async for response in super().generate(request):
            yield response
