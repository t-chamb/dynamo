// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Block Manager Dynamo Integration Tests
//!
//! This module both the integration components in the `llm_kvbm` module
//! and the tests for the `llm_kvbm` module.
//!
//! The intent is to move [llm_kvbm] to a separate crate in the future.

pub mod llm_kvbm {
    // alias for the kvbm module to make the refactor to standalone crate easier
    use dynamo_llm::block_manager as kvbm;

    // kvbm specific imports
    use kvbm::{block::registry::RegistrationHandle, events::*, config::NixlOptions};

    // std imports
    use std::sync::Arc;

    use anyhow::Result;
    use derive_builder::Builder;
    use derive_getters::Dissolve;
    use dynamo_runtime::DistributedRuntime;
    use dynamo_llm::tokens::SequenceHash;
    use dynamo_runtime::component::Namespace;

    #[derive(Debug, PartialEq, Eq)]
    pub enum EventType {
        Register(SequenceHash),
        Remove(SequenceHash),
    }


    /// Translate the Dynamo [`DistributedRuntime`] to the [`kvbm::config::KvManagerRuntimeConfig`]
    #[derive(Clone, Builder, Dissolve)]
    #[builder(pattern = "owned", build_fn(private, name = "build_internal"))]
    pub struct DynamoKvbmRuntimeConfig {
        pub runtime: DistributedRuntime,
        pub nixl: kvbm::config::NixlOptions,
    }

    impl DynamoKvbmRuntimeConfig {
        pub fn builder() -> DynamoKvbmRuntimeConfigBuilder {
            DynamoKvbmRuntimeConfigBuilder::default()
        }
    }

    impl DynamoKvbmRuntimeConfigBuilder {
        pub fn build(self) -> Result<kvbm::config::KvManagerRuntimeConfig> {
            let (runtime, nixl) = self.build_internal()?.dissolve();
            Ok(kvbm::config::KvManagerRuntimeConfig::builder()
                .worker_id(runtime.primary_lease().unwrap().id() as u64)
                .cancellation_token(runtime.primary_token().child_token())
                .nixl(nixl)
                .build()?)
        }
    }

    /// Implementation of the [`kvbm::events::EventManager`] for the Dynamo Runtime Event Plane + the
    /// Dynamo LLM KV router message protocol.
    pub struct DynamoEventManager {
        // TODO: Implement the Dynamo Event Manager
        runtime: DistributedRuntime,
        event_channel: tokio::sync::mpsc::UnboundedSender<Vec<EventType>>,
    }

    impl DynamoEventManager {
        pub fn new(runtime: DistributedRuntime, namespace_name: &str) -> (
            Arc<Self>,
            tokio::sync::mpsc::UnboundedReceiver<Vec<EventType>>, //todo change for a proper receiver
        ) {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            let namespace = runtime.namespace(namespace_name);

            (
                Arc::new(Self {
                    namespace,
                    event_channel: tx,
                }),
                rx,
            )
        }

        pub fn publisher(self: &Arc<Self>) -> Publisher {
            Publisher::new(self.clone())
        }
    }

    impl kvbm::events::EventManager for DynamoEventManager {}

    #[async_trait]
    impl kvbm::events::EventPublisher for DynamoEventManager {
        fn publish(&self, handles: Vec<Arc<RegistrationHandle>>) {
            let events = handles
                .into_iter()
                .map(|handle| EventType::Register(handle.sequence_hash()))
                .collect::<Vec<_>>();

            if self.namespace
                .publish("block_events", &events)
                .await
                .is_err()
            {
                tracing::warn!("Failed to publish RegisterBlock events");
            }
        }
    }

    #[async_trait]
    impl kvbm::events::EventReleaseManager for DynamoEventManager {
        async fn block_release(&self, registration_handle: &RegistrationHandle) {
            let event = EventType::Remove(registration_handle.sequence_hash());

            if self.namespace
                .publish("block_events", &vec![event])
                .await
                .is_err()
            {
                tracing::warn!("Failed to publish ReleaseBlock event");
            }
        }
    }
}

#[allow(unused_imports)]
use llm_kvbm::*;

use dynamo_runtime::{DistributedRuntime, Runtime};
use dynamo_llm::block_manager::NixlOptions;
use dynamo_llm::block_manager::storage::DeviceAllocator;
use dynamo_llm::block_manager::KvManagerLayoutConfig;

// fn create_dynamo_block_manager() -> ReferenceBlockManager {
//     let worker_id = WORKER_ID.fetch_add(1, Ordering::SeqCst);
//     let config = KvBlockManagerConfig::builder()
//         .runtime(
//             KvManagerRuntimeConfig::builder()
//                 .worker_id(worker_id)
//                 .build()
//                 .unwrap(),
//         )
//         .model(
//             KvManagerModelConfig::builder()
//                 .num_layers(3)
//                 .page_size(4)
//                 .inner_dim(16)
//                 .build()
//                 .unwrap(),
//         )
//         .host_layout(
//             KvManagerLayoutConfig::builder()
//                 .num_blocks(16)
//                 .allocator(storage::PinnedAllocator::default())
//                 .build()
//                 .unwrap(),
//         )
//         .device_layout(
//             KvManagerLayoutConfig::builder()
//                 .num_blocks(8)
//                 .allocator(storage::DeviceAllocator::new(0).unwrap())
//                 .build()
//                 .unwrap(),
//         )
//         .build()
//         .unwrap();

//     ReferenceBlockManager::new(config).unwrap()
// }

#[tokio::test]
async fn test_dynamo_kvbm_runtime_config_builder() {
    let rt = Runtime::from_current().unwrap();
    let runtime = DistributedRuntime::from_settings(rt.clone()).await.unwrap();
    let nixl = NixlOptions::Enabled;

    let config = DynamoKvbmRuntimeConfig::builder()
        .runtime(runtime.clone())
        .nixl(nixl)
        .build()
        .unwrap();

    assert_eq!(config.worker_id, runtime.primary_lease().unwrap().id() as u64);
   //assert!(config.cancellation_token().is_child_of(runtime.primary_token()));
    assert!(matches!(config.nixl, NixlOptions::Enabled));
}

#[tokio::test]
async fn test_nixl() {
    let rt = Runtime::from_current().unwrap();
    let runtime = DistributedRuntime::from_settings(rt.clone()).await.unwrap();
    let nixl = NixlOptions::Enabled;

    let config = KvManagerLayoutConfig::builder()
        .num_blocks(8)
        .allocator(DeviceAllocator::new(0).unwrap())
        .build()
        .unwrap();


}

#[test]
fn test_dynamo_block_manager_blocking() {
    // let event_manager = DynamoEventManager::new();
}

#[tokio::test]
async fn test_dynamo_block_manager_async() {
    // let event_manager = DynamoEventManager::new();
}