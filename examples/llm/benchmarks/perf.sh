#!/bin/bash
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

# dynamo + trtllm
model=nvidia/DeepSeek-R1-FP4
tokenizer="${model}"

host="localhost"

# TODO: Why does genai-perf with isl=8192 actually send 8195 tokens?
# RuntimeError: [TensorRT-LLM][ERROR] Assertion failed: The number of context tokens (8195) exceeds the limit value (8192)
# FIXME: Use isl=8000 as a WAR to avoid exceeding 8192 max seq_len for prefill
isl=8000
osl=256

# NOTE: putting ignore_eos inside nvext is currently required when benchmarking
# dynamo -- the dynamo OpenAI frontend won't propagate ignore_eos outside of
# of the nvext json field.

# Concurrency levels to test
for concurrency in 1 2 4 8 16 32 64 128 256; do
  genai-perf profile \
    --model ${model} \
    --tokenizer ${tokenizer} \
    --service-kind openai \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url ${host}:8000 \
    --synthetic-input-tokens-mean ${isl} \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean ${osl} \
    --output-tokens-stddev 0 \
    --extra-inputs max_tokens:${osl} \
    --extra-inputs min_tokens:${osl} \
    --extra-inputs ignore_eos:true \
    --extra-inputs "{\"nvext\":{\"ignore_eos\":true}}" \
    --concurrency ${concurrency} \
    --request-count $(($concurrency*10)) \
    --warmup-request-count $(($concurrency*2)) \
    --num-dataset-entries $(($concurrency*12)) \
    --random-seed 100 \
    -- \
    -v \
    --max-threads 256 \
    -H 'Authorization: Bearer NOT USED' \
    -H 'Accept: text/event-stream'

done

