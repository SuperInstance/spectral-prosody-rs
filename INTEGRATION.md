# INTEGRATION.md — spectral-prosody-rs

> **Repository:** `SuperInstance/spectral-prosody-rs`  
> **Language:** Rust  
> **License:** MIT  
> **Purpose:** Spectral analysis of prosodic features in agent communication — pitch, energy, rhythm, and MFCC-inspired features.

---

## What This Crate Provides

`spectral-prosody-rs` gives agents an "ear" for communication. It extracts four fundamental prosodic channels from time-series signals:

- **Pitch contours** — fundamental frequency trajectories with semitone conversion, voicing detection, and jitter measurement
- **Energy envelopes** — frame-level energy with RMS, dB scaling, peak detection, dynamic range, and slope analysis
- **Rhythmic patterns** — autocorrelation-based tempo detection, inter-onset intervals, rhythmic regularity, and spectral flux
- **Spectral features** — power spectrum, DCT, mel filter banks, and full MFCC pipelines with delta/delta-delta features

All operations are pure Rust with zero external dependencies.

---

## Quick Start

```toml
[dependencies]
spectral-prosody-rs = { git = "https://github.com/SuperInstance/spectral-prosody-rs" }
```

```rust
use spectral_prosody_rs::pitch::{PitchContour, extract_pitch_contour};
use spectral_prosody_rs::energy::{EnergyEnvelope, extract_energy_envelope};
use spectral_prosody_rs::rhythm::{detect_rhythm, detect_tempo};
use spectral_prosody_rs::spectral_features::{compute_mfcc, ProsodicFeatures};

// Extract pitch from a speech signal
let contour = extract_pitch_contour(&signal, 16000.0, 3200, 1600, 50.0, 500.0);
println!("Mean pitch: {:.1} Hz, voicing: {:.2}", contour.mean(), contour.voicing_ratio());

// Energy envelope
let envelope = extract_energy_envelope(&signal, 16000.0, 800, 400);
println!("RMS: {:.4}, dynamic range: {:.1} dB", envelope.rms(), envelope.dynamic_range_db());

// Detect tempo
if let Some(tempo) = detect_tempo(&envelope) {
    println!("Detected tempo: {:.1} BPM", tempo);
}

// Full MFCC feature extraction
let features = ProsodicFeatures::extract(&signal, 16000.0, 400, 200, 13, 26);
println!("Frames: {}, feature dim: {}", features.num_frames(), features.feature_dim());
```

---

## Cross-Repository Integration

### 1. Conservation-Law Energy Budgeting for Prosodic Streams

Feed prosodic energy envelopes into `conservation-law-rs` to model communicative energy budgets. Each agent's speech signal has a total energy budget that must be conserved across processing stages.

```rust
use spectral_prosody_rs::energy::{extract_energy_envelope, EnergyEnvelope};
use conservation_law_rs::budget::EnergyBudget;

// Extract envelope from agent's outgoing speech
let envelope = extract_energy_envelope(&signal, 16000.0, 800, 400);
let total_energy: f64 = envelope.values.iter().sum();

// Allocate energy budget across prosodic dimensions
let mut budget = EnergyBudget::new(total_energy);
budget.allocate("pitch_tracking", total_energy * 0.15);
budget.allocate("rhythm_analysis", total_energy * 0.25);
budget.allocate("mfcc_extraction", total_energy * 0.40);
budget.allocate("residual", total_energy * 0.20);

assert!(budget.check_conservation(), "Prosodic energy must be conserved");
```

### 2. Fleet Registry via si-cli

Register the spectral-prosody capability with `si-cli` so other agents can discover it.

```bash
# From the si-cli repo
cargo run -- scan /path/to/spectral-prosody-rs --sync

# Generate a CAPABILITY.toml for this crate
cargo run -- generate capability --name spectral-prosody-rs --output CAPABILITY.toml
```

### 3. Optimal Transport Between Agent Prosodic Distributions

Use `optimal-transport-agents-rs` to compare the prosodic feature distributions of different agents. Each agent's speech has a characteristic prosodic "fingerprint" that can be compared via Wasserstein distance.

```rust
use spectral_prosody_rs::spectral_features::ProsodicFeatures;
use optimal_transport_agents_rs::distribution::AgentDistribution;
use optimal_transport_agents_rs::sinkhorn::wasserstein_2;

// Extract MFCC features for two agents
let features_a = ProsodicFeatures::extract(&signal_a, 16000.0, 400, 200, 13, 26);
let features_b = ProsodicFeatures::extract(&signal_b, 16000.0, 400, 200, 13, 26);

// Build distributions from mean MFCC vectors per frame
let dist_a = AgentDistribution::from_support_points(
    features_a.mfcc.iter().map(|f| f.clone()).collect(),
    vec![1.0 / features_a.num_frames() as f64; features_a.num_frames()],
);
let dist_b = AgentDistribution::from_support_points(
    features_b.mfcc.iter().map(|f| f.clone()).collect(),
    vec![1.0 / features_b.num_frames() as f64; features_b.num_frames()],
);

let distance = wasserstein_2(&dist_a, &dist_b, 1.0).unwrap();
println!("Prosodic distance between agents: {:.4}", distance);
```

### 4. Topological Analysis of Prosodic Feature Clouds

Use `witness-topology-rs` to discover the topological shape of an agent's prosodic feature space. Do different emotional states form distinct connected components? Are there loops in the prosodic manifold?

```rust
use spectral_prosody_rs::spectral_features::compute_mfcc;
use witness_topology_rs::landmark::{maxmin_landmarks, distance_matrix};
use witness_topology_rs::witness::build_witness_complex;
use witness_topology_rs::persistence::{compute_persistence, betti_numbers};

// Build point cloud from MFCC frames
let mut point_cloud: Vec<Vec<f64>> = Vec::new();
for frame in features.mfcc.iter() {
    point_cloud.push(frame.clone());
}

// Select landmarks and build witness complex
let landmarks = maxmin_landmarks(&point_cloud, 20);
let complex = build_witness_complex(&point_cloud, &landmarks, 5);
let dist = distance_matrix(&point_cloud);
let pairs = compute_persistence(&complex, &dist);
let betti = betti_numbers(&pairs, f64::INFINITY);

println!("Prosodic topology: β₀={}, β₁={}, β₂={}",
    betti.get(0).unwrap_or(&0),
    betti.get(1).unwrap_or(&0),
    betti.get(2).unwrap_or(&0));
```

### 5. Fleet Budget Auditing via si-fleet-api

When spectral-prosody processing is deployed as a fleet service, use `si-fleet-api` to verify that the computational budget (γ) plus the entropy budget (η) equals the total resource allocation.

```bash
# Query fleet budgets for the prosody-processing agents
curl -s "https://fleet-api.example.com/api/fleet/budgets?agent_type=prosody" | jq '.'

# Verify conservation for each prosody agent
for agent_id in $(curl -s "https://fleet-api.example.com/api/fleet/budgets" | jq -r '.[].agent_id'); do
  curl -s "https://fleet-api.example.com/api/fleet/audit?agent=$agent_id" | jq '.conservation_passed'
done
```

### 6. Multi-Scale Prosodic Analysis with Renormalization Group

Use `renormalization-group-rs` to analyze prosodic features across time scales — from micro-prosody (individual phonemes) to macro-prosody (paragraph-level intonation patterns).

```rust
use spectral_prosody_rs::energy::EnergyEnvelope;
use renormalization_group_rs::coarse_grain::coarse_grain_energy;

// Coarse-grain energy envelope at multiple scales
let scales = vec![1, 2, 4, 8, 16];
for scale in &scales {
    let coarse = coarse_grain_energy(&envelope.values, *scale);
    println!("Scale {}: {} frames, mean energy {:.4}", scale, coarse.len(),
        coarse.iter().sum::<f64>() / coarse.len() as f64);
}
```

---

## Design Patterns

### Pure Functional Processing

All feature extractors are pure functions: `signal → features`. No mutable global state, no side effects. This makes them trivially parallelizable and testable.

### Frame-Based Streaming

The API is designed for frame-based processing: extract features from overlapping windows, producing a time-series of feature vectors. This supports both batch and streaming use cases.

### Zero-Allocation Reuse

Where possible, operations reuse buffers (e.g., `smooth_energy` returns a new `EnergyEnvelope` but could be adapted to write into a pre-allocated buffer for real-time use).

---

## Integration Checklist

- [ ] Add `spectral-prosody-rs` to `Cargo.toml` of consuming crate
- [ ] Ensure signal sample rates match between audio input and feature extractors
- [ ] Verify frame sizes are powers of 2 for efficient DFT (or use `rustfft` for production)
- [ ] Connect `EnergyEnvelope` outputs to `conservation-law-rs` budget tracking
- [ ] Register capabilities in `CAPABILITY.toml` for fleet discovery
- [ ] Add `si-cli` scan to CI to keep dependency graph current
- [ ] Monitor fleet budgets via `si-fleet-api` when deployed at scale

---

## Related Repositories

| Repository | Integration Point |
|---|---|
| `conservation-law-rs` | Energy budget conservation across prosodic channels |
| `optimal-transport-agents-rs` | Wasserstein comparison of agent prosodic distributions |
| `witness-topology-rs` | Topological shape of prosodic feature clouds |
| `renormalization-group-rs` | Multi-scale prosodic analysis |
| `si-cli` | Capability scanning and fleet registry |
| `si-fleet-api` | Fleet budget auditing and conservation checks |
| `ecosystem-dashboard` | Live monitoring of prosody-processing agents |
