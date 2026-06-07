//! Energy envelope tracking for prosodic analysis.

/// Energy envelope of a signal.
#[derive(Debug, Clone)]
pub struct EnergyEnvelope {
    pub values: Vec<f64>,
    pub sample_rate: f64,
    pub frame_shift: f64,
}

impl EnergyEnvelope {
    pub fn new(values: Vec<f64>, sample_rate: f64, frame_shift: f64) -> Self {
        Self {
            values,
            sample_rate,
            frame_shift,
        }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Mean energy.
    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.iter().sum::<f64>() / self.values.len() as f64
    }

    /// Peak energy.
    pub fn peak(&self) -> f64 {
        self.values
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max)
    }

    /// Energy in dB.
    pub fn to_db(&self, reference: f64) -> Vec<f64> {
        self.values
            .iter()
            .map(|&e| {
                if e > 0.0 && reference > 0.0 {
                    10.0 * (e / reference).log10()
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect()
    }

    /// Root mean square energy.
    pub fn rms(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        let mean_sq = self.values.iter().map(|v| v * v).sum::<f64>() / self.values.len() as f64;
        mean_sq.sqrt()
    }

    /// Dynamic range (peak - min in dB).
    pub fn dynamic_range_db(&self) -> f64 {
        let min_val = self.values.iter().cloned().filter(|&v| v > 0.0).fold(f64::INFINITY, f64::min);
        let max_val = self.peak();
        if min_val <= 0.0 || max_val <= 0.0 {
            return 0.0;
        }
        10.0 * (max_val / min_val).log10()
    }
}

/// Compute frame energy (sum of squares).
pub fn frame_energy(frame: &[f64]) -> f64 {
    frame.iter().map(|&x| x * x).sum()
}

/// Compute RMS of a frame.
pub fn frame_rms(frame: &[f64]) -> f64 {
    if frame.is_empty() {
        return 0.0;
    }
    let mean_sq = frame.iter().map(|x| x * x).sum::<f64>() / frame.len() as f64;
    mean_sq.sqrt()
}

/// Extract energy envelope from signal.
pub fn extract_energy_envelope(
    signal: &[f64],
    sample_rate: f64,
    frame_length: usize,
    frame_shift: usize,
) -> EnergyEnvelope {
    let mut energies = Vec::new();
    let mut start = 0;
    while start + frame_length <= signal.len() {
        let frame = &signal[start..start + frame_length];
        let energy = frame_rms(frame);
        energies.push(energy);
        start += frame_shift;
    }
    EnergyEnvelope::new(energies, sample_rate, frame_shift as f64 / sample_rate)
}

/// Smooth energy envelope with exponential moving average.
pub fn smooth_energy(envelope: &EnergyEnvelope, alpha: f64) -> EnergyEnvelope {
    if envelope.values.is_empty() {
        return EnergyEnvelope::new(vec![], envelope.sample_rate, envelope.frame_shift);
    }
    let mut smoothed = Vec::with_capacity(envelope.values.len());
    smoothed.push(envelope.values[0]);
    for i in 1..envelope.values.len() {
        let prev = smoothed[i - 1];
        let curr = envelope.values[i];
        smoothed.push(alpha * curr + (1.0 - alpha) * prev);
    }
    EnergyEnvelope::new(smoothed, envelope.sample_rate, envelope.frame_shift)
}

/// Detect energy peaks (local maxima above threshold).
pub fn detect_peaks(envelope: &EnergyEnvelope, threshold: f64) -> Vec<usize> {
    let mut peaks = Vec::new();
    for i in 1..envelope.values.len().saturating_sub(1) {
        if envelope.values[i] > envelope.values[i - 1]
            && envelope.values[i] > envelope.values[i + 1]
            && envelope.values[i] > threshold
        {
            peaks.push(i);
        }
    }
    peaks
}

/// Compute energy slope (trend).
pub fn energy_slope(envelope: &EnergyEnvelope) -> f64 {
    let n = envelope.values.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let sum_x: f64 = (0..envelope.values.len()).map(|i| i as f64).sum();
    let sum_y: f64 = envelope.values.iter().sum();
    let sum_xy: f64 = envelope
        .values
        .iter()
        .enumerate()
        .map(|(i, &v)| i as f64 * v)
        .sum();
    let sum_xx: f64 = (0..envelope.values.len()).map(|i| (i as f64) * (i as f64)).sum();

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-15 {
        return 0.0;
    }

    (n * sum_xy - sum_x * sum_y) / denom
}

/// Normalize energy to [0, 1].
pub fn normalize_energy(envelope: &EnergyEnvelope) -> EnergyEnvelope {
    let max_val = envelope.peak();
    if max_val <= 0.0 {
        return envelope.clone();
    }
    let normalized: Vec<f64> = envelope.values.iter().map(|&v| v / max_val).collect();
    EnergyEnvelope::new(normalized, envelope.sample_rate, envelope.frame_shift)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_energy() {
        let frame = vec![1.0, 2.0, 3.0];
        assert!((frame_energy(&frame) - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_frame_rms() {
        let frame = vec![3.0, 4.0];
        // sqrt((9 + 16) / 2) = sqrt(12.5) ≈ 3.536
        let rms = frame_rms(&frame);
        assert!((rms - (12.5_f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_extract_energy_envelope() {
        let signal: Vec<f64> = (0..3200).map(|i| (i as f64 / 100.0).sin()).collect();
        let env = extract_energy_envelope(&signal, 16000.0, 800, 400);
        assert!(env.len() > 0);
        assert!(env.mean() > 0.0);
    }

    #[test]
    fn test_energy_mean() {
        let env = EnergyEnvelope::new(vec![1.0, 2.0, 3.0], 16000.0, 0.01);
        assert!((env.mean() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_energy_peak() {
        let env = EnergyEnvelope::new(vec![1.0, 5.0, 3.0], 16000.0, 0.01);
        assert!((env.peak() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_energy_rms() {
        let env = EnergyEnvelope::new(vec![3.0, 4.0], 16000.0, 0.01);
        let rms = env.rms();
        assert!((rms - (12.5_f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_to_db() {
        let env = EnergyEnvelope::new(vec![1.0, 10.0, 100.0], 16000.0, 0.01);
        let db = env.to_db(1.0);
        assert!((db[0] - 0.0).abs() < 1e-10);
        assert!((db[1] - 10.0).abs() < 1e-10);
        assert!((db[2] - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_smooth_energy() {
        let env = EnergyEnvelope::new(vec![0.0, 1.0, 0.0, 1.0], 16000.0, 0.01);
        let smooth = smooth_energy(&env, 0.5);
        assert_eq!(smooth.len(), 4);
        // Smoothed values should be less extreme
        assert!(smooth.values[1] < 1.0);
    }

    #[test]
    fn test_detect_peaks() {
        let env = EnergyEnvelope::new(vec![1.0, 5.0, 1.0, 3.0, 1.0], 16000.0, 0.01);
        let peaks = detect_peaks(&env, 2.0);
        assert_eq!(peaks, vec![1, 3]);
    }

    #[test]
    fn test_energy_slope() {
        let env = EnergyEnvelope::new(vec![1.0, 2.0, 3.0, 4.0], 16000.0, 0.01);
        let slope = energy_slope(&env);
        assert!((slope - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_energy() {
        let env = EnergyEnvelope::new(vec![2.0, 4.0, 6.0], 16000.0, 0.01);
        let norm = normalize_energy(&env);
        assert!((norm.values[0] - 1.0 / 3.0).abs() < 1e-10);
        assert!((norm.values[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dynamic_range() {
        let env = EnergyEnvelope::new(vec![1.0, 10.0, 100.0], 16000.0, 0.01);
        let dr = env.dynamic_range_db();
        assert!((dr - 20.0).abs() < 1e-10); // 10*log10(100/1) = 20 dB
    }

    #[test]
    fn test_empty_envelope() {
        let env = EnergyEnvelope::new(vec![], 16000.0, 0.01);
        assert!(env.is_empty());
        assert_eq!(env.mean(), 0.0);
        assert_eq!(env.rms(), 0.0);
    }
}
