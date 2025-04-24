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
from pathlib import Path
from typing import AsyncGenerator

from components.processor import Processor
from components.worker import VllmWorker
from pydantic import BaseModel
from utils.protocol import MultiModalRequest

from dynamo import sdk
from dynamo.sdk import api, depends, service
from dynamo.sdk.lib.image import DYNAMO_IMAGE

logger = logging.getLogger(__name__)


def get_http_binary_path():
    sdk_path = Path(sdk.__file__)
    binary_path = sdk_path.parent / "cli/bin/http"
    if not binary_path.exists():
        return "http"
    else:
        return str(binary_path)


class FrontendConfig(BaseModel):
    served_model_name: str
    endpoint: str
    port: int = 8080


@service(
    resources={"cpu": "10", "memory": "20Gi"},
    workers=1,
    image=DYNAMO_IMAGE,
)
class Frontend:
    worker = depends(VllmWorker)
    processor = depends(Processor)

    @api
    async def generate(
        self,
        model: str,
        image: str,
        max_tokens: int = 300,
        prompt: str = "Describe the image in detail.",
    ) -> AsyncGenerator[str, None]:
        request = MultiModalRequest(
            model=model, image=image, max_tokens=max_tokens, prompt=prompt
        )
        async for response in self.processor.generate(request.model_dump_json()):
            yield response
