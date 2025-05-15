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

This directory is currently used for generate synthetic data based on the mooncake dataset, but should be easily extendible to any request datasets with (prefix) hash ids, with a current cavaet. The synthesizer is designed to work for jsonl files in the "mooncake" trace file format, meaning that the input are increasing integers of block hashes. For now, new block hashes must be the next consecutive integer, otherwise will not work.

## Quickstart

For instance, you can run from the project root:
```
python -m benchmark.data_synth.synthesizer \
--input-file mooncake_trace.jsonl \
--num-requests 500 \
--depth-multiplier 4 \
--width-multiplier 4 \
--prompt-len-multiplier 0.1
```
where `num-requests` sets the number of total synthetic requests generated, `speedup-ratio` tunes the rate at which the requests are sent, `depth-multiplier` tunes the lengths of the request prefixes (higher multiplier will then yield longer ISLs), and `width-multiplier` controls the branching factor of the synthetic requests (higher multiplier will generate more diverse request patterns).

## How it works

The generation algorithm, simplified, is as follows

- Store the hash ids in a directed tree structure (prefix tree)
- Each directed edge `weight` indicates how many times the edge is traversed, which is needed to compute transition probabilities.
- Contract unary paths (chains) in the tree so that it is in a radix-tree form, meaning every node that is the only child will be contracted with the parent. As a consequence, each node need to store an attribute `length` to indicate the compressed length (1 if no compression). The depth multiplier scales this compressed length (rounded to the nearest integer), effectively increasing the length of each radix node.
- Identify every leaf node that is visited only once, and prune them from the tree, as they are highly likely not part of the core radix tree. In other words, we do not need to store nodes that are part of the actual user prompts.
- At this stage, each node will have (possibly zero) transition probabilities to a child prefix node, to a "user prompt" node, and to a "termination" node. Use these probabilities to sample a path in the core radix tree, the append the path with new hash ids corresponding to a user prompt of length sampled from the dataset. The width multiplier effectively duplicates the entire radix tree the specified number of times, each with a new set of hash ids, creating more diverse request patterns.

## Testing

To test for "correctness", or faithfulness to the original mooncake statistics, one can run
```
python mooncake_synth.py --num-requests 500000
```
and compare the synthetic ISL statistics (mean, median, std) to the original ISL statistics. I find this to be the most "robust" end-to-end test.
