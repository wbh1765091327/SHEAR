use serde::{Deserialize, Serialize};

use super::params::PiPNNGpuSubtreePolicy;

pub(crate) const GPU_HARDWARE_ADAPTATION_SCHEMA: &str = "shear-gpu-hardware-adaptation-v1";

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct GpuDeviceCaps {
    pub(crate) device_id: usize,
    pub(crate) device_uuid: String,
    pub(crate) device_name: String,
    pub(crate) compute_capability_major: usize,
    pub(crate) compute_capability_minor: usize,
    pub(crate) sm_count: usize,
    pub(crate) warp_size: usize,
    pub(crate) max_threads_per_block: usize,
    pub(crate) max_threads_per_sm: usize,
    pub(crate) max_blocks_per_sm: usize,
    pub(crate) max_shared_bytes_per_block: usize,
    pub(crate) max_shared_bytes_per_sm: usize,
    pub(crate) total_hbm_bytes: usize,
    pub(crate) driver_version: i32,
    pub(crate) runtime_version: i32,
}

impl GpuDeviceCaps {
    pub(crate) fn supports_persistent_wmma(&self) -> bool {
        self.compute_capability_major >= 8
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct GpuMemorySnapshot {
    pub(crate) label: String,
    pub(crate) free_hbm_bytes: usize,
    pub(crate) total_hbm_bytes: usize,
    pub(crate) captured_at_unix_ms: u128,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct GpuHardwareMemoryProfile {
    pub(crate) snapshots: Vec<GpuMemorySnapshot>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct GpuKernelCaps {
    pub(crate) entrypoint: String,
    pub(crate) kernel_source_hash: String,
    pub(crate) nvrtc_arch: String,
    pub(crate) compile_specialization: String,
    pub(crate) block_threads: usize,
    pub(crate) warps_per_cta: usize,
    pub(crate) registers_per_thread: usize,
    pub(crate) static_shared_bytes: usize,
    pub(crate) dynamic_shared_bytes: usize,
    pub(crate) active_ctas_per_sm: usize,
    pub(crate) resident_cta_cap: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct GpuPersistentLaunchPlan {
    pub(crate) entrypoint: String,
    pub(crate) block_threads: usize,
    pub(crate) warps_per_cta: usize,
    pub(crate) total_worker_ctas: usize,
    pub(crate) partition_worker_ctas: usize,
    pub(crate) topk_worker_ctas: usize,
    pub(crate) resident_cta_cap: usize,
    pub(crate) work_demand_ctas: usize,
    pub(crate) plan_source: String,
    pub(crate) calibration_key: Option<String>,
    pub(crate) decision_reason: String,
}

impl Default for GpuPersistentLaunchPlan {
    fn default() -> Self {
        Self {
            entrypoint: "gpu_persistent_subtree_topk".to_string(),
            block_threads: 256,
            warps_per_cta: 8,
            total_worker_ctas: 0,
            partition_worker_ctas: 0,
            topk_worker_ctas: 0,
            resident_cta_cap: 0,
            work_demand_ctas: 0,
            plan_source: "legacy".to_string(),
            calibration_key: None,
            decision_reason: "legacy_observation_only".to_string(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct GpuHardwareDecision {
    pub(crate) legacy_plan: GpuPersistentLaunchPlan,
    pub(crate) recommended_plan: Option<GpuPersistentLaunchPlan>,
    pub(crate) applied_plan: GpuPersistentLaunchPlan,
    pub(crate) applied_source: String,
    pub(crate) reason: String,
}

impl Default for GpuHardwareDecision {
    fn default() -> Self {
        Self {
            legacy_plan: GpuPersistentLaunchPlan::default(),
            recommended_plan: None,
            applied_plan: GpuPersistentLaunchPlan::default(),
            applied_source: "legacy".to_string(),
            reason: "legacy_observation_only".to_string(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct GpuHardwareAdaptationProfile {
    pub(crate) schema_version: String,
    pub(crate) launch_policy: PiPNNGpuSubtreePolicy,
    pub(crate) hbm_policy: PiPNNGpuSubtreePolicy,
    pub(crate) device: GpuDeviceCaps,
    pub(crate) memory: GpuHardwareMemoryProfile,
    pub(crate) kernel_caps: Vec<GpuKernelCaps>,
    pub(crate) decision: GpuHardwareDecision,
}

impl GpuHardwareAdaptationProfile {
    pub(crate) fn new(
        launch_policy: PiPNNGpuSubtreePolicy,
        hbm_policy: PiPNNGpuSubtreePolicy,
        device: GpuDeviceCaps,
    ) -> Self {
        Self {
            schema_version: GPU_HARDWARE_ADAPTATION_SCHEMA.to_string(),
            launch_policy,
            hbm_policy,
            device,
            memory: GpuHardwareMemoryProfile::default(),
            kernel_caps: Vec::new(),
            decision: GpuHardwareDecision::default(),
        }
    }

    pub(crate) fn upsert_kernel_caps(&mut self, caps: GpuKernelCaps) {
        let existing_position = self.kernel_caps.iter().position(|existing| {
            existing.entrypoint == caps.entrypoint
                && existing.kernel_source_hash == caps.kernel_source_hash
                && existing.nvrtc_arch == caps.nvrtc_arch
                && existing.compile_specialization == caps.compile_specialization
                && existing.block_threads == caps.block_threads
                && existing.dynamic_shared_bytes == caps.dynamic_shared_bytes
        });
        if let Some(position) = existing_position {
            self.kernel_caps[position] = caps;
        } else {
            self.kernel_caps.push(caps);
        }
    }
}

pub(crate) fn resident_cta_cap(sm_count: usize, active_ctas_per_sm: usize) -> usize {
    sm_count.saturating_mul(active_ctas_per_sm)
}

#[cfg(test)]
mod tests {
    use super::{
        GpuDeviceCaps, GpuHardwareAdaptationProfile, GpuHardwareDecision, GpuKernelCaps,
        GpuMemorySnapshot, GpuPersistentLaunchPlan, resident_cta_cap,
    };
    use crate::algorithm::pipnn::params::PiPNNGpuSubtreePolicy;

    fn test_profile() -> GpuHardwareAdaptationProfile {
        GpuHardwareAdaptationProfile::new(
            PiPNNGpuSubtreePolicy::Report,
            PiPNNGpuSubtreePolicy::Report,
            GpuDeviceCaps::default(),
        )
    }

    fn kernel_caps() -> GpuKernelCaps {
        GpuKernelCaps {
            entrypoint: "gpu_persistent_subtree_topk".to_string(),
            kernel_source_hash: "test-source-hash".to_string(),
            nvrtc_arch: "compute_80".to_string(),
            compile_specialization: "wmma".to_string(),
            block_threads: 256,
            warps_per_cta: 8,
            registers_per_thread: 64,
            static_shared_bytes: 4096,
            dynamic_shared_bytes: 0,
            active_ctas_per_sm: 2,
            resident_cta_cap: 256,
        }
    }

    #[test]
    fn upsert_kernel_caps_keeps_distinct_dynamic_shared_memory_variants() {
        let mut profile = test_profile();
        let base = kernel_caps();
        let dynamic_variant = GpuKernelCaps {
            dynamic_shared_bytes: 16 * 1024,
            ..base.clone()
        };

        profile.upsert_kernel_caps(base.clone());
        profile.upsert_kernel_caps(dynamic_variant.clone());

        assert_eq!(profile.kernel_caps, vec![base, dynamic_variant]);
    }

    #[test]
    fn upsert_kernel_caps_replaces_a_fully_identical_key_in_place() {
        let mut profile = test_profile();
        let base = kernel_caps();
        let second = GpuKernelCaps {
            block_threads: 128,
            warps_per_cta: 4,
            registers_per_thread: 48,
            ..base.clone()
        };
        let replacement = GpuKernelCaps {
            registers_per_thread: 60,
            ..base.clone()
        };

        profile.upsert_kernel_caps(base);
        profile.upsert_kernel_caps(second.clone());
        profile.upsert_kernel_caps(replacement.clone());

        assert_eq!(profile.kernel_caps.len(), 2);
        assert_eq!(profile.kernel_caps[0], replacement);
        assert_eq!(profile.kernel_caps[1], second);
    }

    #[test]
    fn upsert_kernel_caps_keeps_every_other_identity_component_distinct() {
        let base = kernel_caps();
        let variants = [
            (
                "entrypoint",
                GpuKernelCaps {
                    entrypoint: "gpu_persistent_subtree_topk_alt".to_string(),
                    ..base.clone()
                },
            ),
            (
                "kernel_source_hash",
                GpuKernelCaps {
                    kernel_source_hash: "other-source-hash".to_string(),
                    ..base.clone()
                },
            ),
            (
                "nvrtc_arch",
                GpuKernelCaps {
                    nvrtc_arch: "compute_90".to_string(),
                    ..base.clone()
                },
            ),
            (
                "compile_specialization",
                GpuKernelCaps {
                    compile_specialization: "raw".to_string(),
                    ..base.clone()
                },
            ),
            (
                "block_threads",
                GpuKernelCaps {
                    block_threads: 128,
                    ..base.clone()
                },
            ),
        ];

        for (identity_component, variant) in variants {
            let mut profile = test_profile();
            profile.upsert_kernel_caps(base.clone());
            profile.upsert_kernel_caps(variant.clone());

            assert_eq!(
                profile.kernel_caps,
                vec![base.clone(), variant],
                "changing {identity_component} must append a distinct capability variant"
            );
        }
    }

    #[test]
    fn hardware_decision_default_is_coherent_legacy_observation() {
        let decision = GpuHardwareDecision::default();

        assert_eq!(decision.legacy_plan, GpuPersistentLaunchPlan::default());
        assert_eq!(decision.applied_plan, decision.legacy_plan);
        assert_eq!(decision.recommended_plan, None);
        assert_eq!(decision.applied_source, "legacy");
        assert_eq!(decision.reason, "legacy_observation_only");
    }

    #[test]
    fn hardware_capability_helpers_cover_architecture_and_capacity_boundaries() {
        let mut device = GpuDeviceCaps {
            compute_capability_major: 7,
            ..GpuDeviceCaps::default()
        };
        assert!(!device.supports_persistent_wmma());
        device.compute_capability_major = 8;
        assert!(device.supports_persistent_wmma());

        assert_eq!(resident_cta_cap(0, 4), 0);
        assert_eq!(resident_cta_cap(4, 0), 0);
        assert_eq!(resident_cta_cap(usize::MAX, 2), usize::MAX);
    }

    #[test]
    fn populated_profile_round_trips_through_stable_json() {
        let profile = GpuHardwareAdaptationProfile::new(
            PiPNNGpuSubtreePolicy::Report,
            PiPNNGpuSubtreePolicy::Off,
            GpuDeviceCaps {
                device_id: 1,
                device_uuid: "GPU-test-uuid".to_string(),
                device_name: "test-gpu".to_string(),
                compute_capability_major: 8,
                compute_capability_minor: 9,
                sm_count: 128,
                total_hbm_bytes: 80 * 1024 * 1024,
                ..GpuDeviceCaps::default()
            },
        );
        let mut profile = profile;
        profile.memory.snapshots.push(GpuMemorySnapshot {
            label: "before_build".to_string(),
            free_hbm_bytes: 60 * 1024 * 1024,
            total_hbm_bytes: 80 * 1024 * 1024,
            captured_at_unix_ms: 123,
        });
        profile.upsert_kernel_caps(kernel_caps());
        profile.decision.recommended_plan = Some(GpuPersistentLaunchPlan {
            total_worker_ctas: 240,
            resident_cta_cap: 256,
            work_demand_ctas: 300,
            plan_source: "calibration".to_string(),
            calibration_key: Some("test-calibration".to_string()),
            decision_reason: "calibrated_occupancy".to_string(),
            ..GpuPersistentLaunchPlan::default()
        });

        let value = serde_json::to_value(&profile).unwrap();

        assert_eq!(value["schema_version"], "shear-gpu-hardware-adaptation-v1");
        assert_eq!(value["launch_policy"], "report");
        assert_eq!(value["hbm_policy"], "off");
        assert_eq!(value["decision"]["applied_source"], "legacy");
        assert_eq!(
            serde_json::from_value::<GpuHardwareAdaptationProfile>(value).unwrap(),
            profile
        );
    }
}
