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

import logging
import time
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.sdk import (
    DYNAMO_IMAGE,
    DynamoServiceInterface,
    abstract_dynamo_endpoint,
    depends,
    dynamo_endpoint,
    service,
)
from dynamo.sdk.lib.config import ServiceConfig

from .types import (
    ChatRequest, ChatResponse,
    RouteRequest, RouteResponse,
    GenerateRequest, GenerateResponse,
)

logger = logging.getLogger(__name__)

"""
Pipeline Architecture:

Users/Clients (HTTP)
      │
      ▼
┌─────────────┐
│  Frontend   │  HTTP API endpoint (/v1/chat/completions)
└─────────────┘
      │ dynamo/runtime
      ▼
┌─────────────┐
│   Router    │  Routes requests to appropriate worker
└─────────────┘
      │ dynamo/runtime
      ▼
┌─────────────┐
│   Worker    │  Generates text using LLM
└─────────────┘
"""


class WorkerInterface(DynamoServiceInterface):
    """Interface for LLM workers."""
    @abstract_dynamo_endpoint
    async def generate(self, request: GenerateRequest):
        pass


class RouterInterface(DynamoServiceInterface):
    """Interface for request routers."""
    @abstract_dynamo_endpoint
    async def route(self, request: RouteRequest):
        pass


@service(dynamo={"enabled": True}, image=DYNAMO_IMAGE)
class VllmWorker(WorkerInterface):
    @dynamo_endpoint()
    async def generate(self, request: GenerateRequest):
        for token in request.text.split():
            yield GenerateResponse(text=f"VLLM: {token}")


@service(dynamo={"enabled": True}, image=DYNAMO_IMAGE)
class TRTLLMWorker(WorkerInterface):
    @dynamo_endpoint()
    async def generate(self, request: GenerateRequest):
        for token in request.text.split():
            yield GenerateResponse(text=f"TRT-LLM: {token}")


@service(dynamo={"enabled": True}, image=DYNAMO_IMAGE)
class SlowRouter(RouterInterface):
    worker = depends(WorkerInterface)  # Will be overridden by link()
    
    @dynamo_endpoint()
    async def route(self, request: RouteRequest):
        print("Routing slow")
        yield RouteResponse(text="slow")
        async for response in self.worker.generate(request.model_dump_json()):
            yield response


@service(dynamo={"enabled": True}, image=DYNAMO_IMAGE)
class FastRouter(RouterInterface):
    worker = depends(WorkerInterface)  # Will be overridden by link()
    
    @dynamo_endpoint()
    async def route(self, request: RouteRequest):
        print("Routing fast")
        yield RouteResponse(text="fast")
        async for response in self.worker.generate(request.model_dump_json()):
            yield response


app = FastAPI()


@service(dynamo={"enabled": True}, image=DYNAMO_IMAGE, app=app)
class Frontend:
    router = depends(RouterInterface)  # Will be overridden by link()
    
    @dynamo_endpoint(is_api=True)
    async def chat_completions(self, request: ChatRequest):
        text = " ".join(msg["content"] for msg in request.messages)
        route_request = RouteRequest(model=request.model, text=text)
        async for response in self.router.route(route_request.model_dump_json()):
            yield ChatResponse(text=response.text)


# Mix and match pipelines
fast_pipeline = Frontend.link(FastRouter).link(TRTLLMWorker)
# slow_pipeline = Frontend.link(SlowRouter).link(VllmWorker)
# mixed_pipeline = Frontend.link(FastRouter).link(VllmWorker)

"""
Example usage:

fast_pipeline = Frontend.link(FastRouter).link(TRTLLMWorker)
# slow_pipeline = Frontend.link(SlowRouter).link(VllmWorker)
# mixed_pipeline = Frontend.link(FastRouter).link(VllmWorker)


# Basic setup with VLLM worker and slow router
The interface-based design allows for:
1. Easy swapping of implementations (VLLM vs TRT-LLM)
2. Different routing strategies (slow vs fast)
3. Type safety through interface contracts
""" 