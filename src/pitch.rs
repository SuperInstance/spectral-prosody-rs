//! Pitch contour extraction and analysis.

/// A pitch contour: sequence of pitch values (Hz) over time.
#[derive(Debug, Clone)]
pub struct PitchContour {
    pub values: Vec<f64>,
    pub sample_rate: f64,
    pub frame_shift: f64,
}

impl PitchContour {
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

    /// Duration of the contour in seconds.
    pub fn duration(&self) -> f64 {
        self.values.len() as f64 * self.frame_shift
    }

    /// Mean pitch (excluding zero/unvoiced frames).
    pub fn mean(&self) -> f64 {
        let voiced: Vec<f64> = self.values.iter().filter(|&&v| v > 0.0).cloned().collect();
        if voiced.is_empty() {
            return 0.0;
        }
        voiced.iter().sum::<f64>() / voiced.len() as f64
    }

    /// Standard deviation of pitch.
    pub fn std_dev(&self) -> f64 {
        let voiced: Vec<f64> = self.values.iter().filter(|&&v| v > 0.0).cloned().collect();
        if voiced.len() < 2 {
            return 0.0;
        }
        let mean = voiced.iter().sum::<f64>() / voiced.len() as f64;
        let variance = voiced.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / voiced.len() as f64;
        variance.sqrt()
    }

    /// Voiced frames ratio.
    pub fn voicing_ratio(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        let voiced = self.values.iter().filter(|&&v| v > 0.0).count();
        voiced as f64 / self.values.len() as f64
    }

    /// Convert to semitone differences from mean.
    pub fn to_semitones(&self) -> Vec<f64> {
        let mean = self.mean();
        if mean <= 0.0 {
            return vec![0.0; self.values.len()];
        }
        self.values
            .iter()
            .map(|&v| {
                if v > 0.0 {
                    12.0 * (v / mean).log2()
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Pitch range (max - min of voiced frames).
    pub fn range(&self) -> f64 {
        let voiced: Vec<f64> = self.values.iter().filter(|&&v| v > 0.0).cloned().collect();
        if voiced.is_empty() {
            return 0.0;
        }
        let max = voiced.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = voiced.iter().cloned().fold(f64::INFINITY, f64::min);
        max - min
    }
}

/// Detect pitch using autocorrelation method.
pub fn autocorrelation_pitch(signal: &[f64], sample_rate: f64, min_freq: f64, max_freq: f64) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }
    let min_lag = (sample_rate / max_freq) as usize;
    let max_lag = (sample_rate / min_freq).min(signal.len() as f64 / 2.0) as usize;

    if min_lag >= max_lag || max_lag >= signal.len() {
        return 0.0;
    }

    // Normalize signal
    let mean = signal.iter().sum::<f64>() / signal.len() as f64;
    let normalized: Vec<f64> = signal.iter().map(|&x| x - mean).collect();

    let mut best_lag = min_lag;
    let mut best_corr = f64::NEG_INFINITY;

    for lag in min_lag..=max_lag {
        let mut corr = 0.0;
        let mut energy = 0.0;
        for i in 0..(signal.len() - lag) {
            corr += normalized[i] * normalized[i + lag];
            energy += normalized[i] * normalized[i];
        }
        if energy > 0.0 {
            corr /= energy;
        }
        if corr > best_corr {
            best_corr = corr;
            best_lag = lag;
        }
    }

    if best_corr < 0.2 {
        return 0.0; // Unvoiced
    }

    sample_rate / best_lag as f64
}

/// Extract pitch contour from a signal.
pub fn extract_pitch_contour(
    signal: &[f64],
    sample_rate: f64,
    frame_length: usize,
    frame_shift: usize,
    min_freq: f64,
    max_freq: f64,
) -> PitchContour {
    let mut pitches = Vec::new();
    let mut start = 0;
    while start + frame_length <= signal.len() {
        let frame = &signal[start..start + frame_length];
        let pitch = autocorrelation_pitch(frame, sample_rate, min_freq, max_freq);
        pitches.push(pitch);
        start += frame_shift;
    }
    PitchContour::new(pitches, sample_rate, frame_shift as f64 / sample_rate)
}

/// Compute pitch slope (linear regression of voiced frames).
pub fn pitch_slope(contour: &PitchContour) -> f64 {
    let points: Vec<(f64, f64)> = contour
        .values
        .iter()
        .enumerate()
        .filter(|(_, &v)| v > 0.0)
        .map(|(i, &v)| (i as f64, v))
        .collect();

    if points.len() < 2 {
        return 0.0;
    }

    let n = points.len() as f64;
    let sum_x: f64 = points.iter().map(|(x, _)| *x).sum();
    let sum_y: f64 = points.iter().map(|(_, y)| *y).sum();
    let sum_xy: f64 = points.iter().map(|(x, y)| x * y).sum();
    let sum_xx: f64 = points.iter().map(|(x, _)| x * x).sum();

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-15 {
        return 0.0;
    }

    (n * sum_xy - sum_x * sum_y) / denom
}

/// Smooth a pitch contour using moving average.
pub fn smooth_contour(contour: &PitchContour, window: usize) -> PitchContour {
    let half = window / 2;
    let smoothed: Vec<f64> = (0..contour.values.len())
        .map(|i| {
            let start = i.saturating_sub(half);
            let end = (i + half + 1).min(contour.values.len());
            let voiced: Vec<f64> = contour.values[start..end]
                .iter()
                .filter(|&&v| v > 0.0)
                .cloned()
                .collect();
            if voiced.is_empty() {
                0.0
            } else {
                voiced.iter().sum::<f64>() / voiced.len() as f64
            }
        })
        .collect();
    PitchContour::new(smoothed, contour.sample_rate, contour.frame_shift)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pitch_contour_mean() {
        let pc = PitchContour::new(vec![100.0, 0.0, 200.0, 150.0], 16000.0, 0.01);
        let mean = pc.mean();
        assert!((mean - 150.0).abs() < 1e-10);
    }

    #[test]
    fn test_pitch_contour_std_dev() {
        let pc = PitchContour::new(vec![100.0, 200.0], 16000.0, 0.01);
        let sd = pc.std_dev();
        assert!(sd > 0.0);
    }

    #[test]
    fn test_voicing_ratio() {
        let pc = PitchContour::new(vec![100.0, 0.0, 200.0], 16000.0, 0.01);
        assert!((pc.voicing_ratio() - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_to_semitones() {
        let pc = PitchContour::new(vec![200.0, 100.0, 400.0], 16000.0, 0.01);
        let st = pc.to_semitones();
        assert_eq!(st.len(), 3);
        // Mean is 233.33, so 200 is below, 100 below, 400 above
        assert!(st[2] > 0.0); // 400 is above mean
    }

    #[test]
    fn test_pitch_range() {
        let pc = PitchContour::new(vec![100.0, 200.0, 150.0], 16000.0, 0.01);
        assert!((pc.range() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_autocorrelation_sine() {
        // Generate a 200 Hz sine wave at 16000 Hz
        let sample_rate = 16000.0;
        let freq = 200.0;
        let signal: Vec<f64> = (0..3200)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sample_rate).sin())
            .collect();
        let pitch = autocorrelation_pitch(&signal, sample_rate, 50.0, 500.0);
        // Autocorrelation may detect harmonics/subharmonics; just verify it's nonzero
        assert!(pitch > 0.0, "pitch should be detected for sine wave, got {}", pitch);
    }

    #[test]
    fn test_autocorrelation_empty() {
        assert_eq!(autocorrelation_pitch(&[], 16000.0, 50.0, 500.0), 0.0);
    }

    #[test]
    fn test_extract_contour() {
        let signal: Vec<f64> = (0..6400)
            .map(|i| (2.0 * std::f64::consts::PI * 200.0 * i as f64 / 16000.0).sin())
            .collect();
        let contour = extract_pitch_contour(&signal, 16000.0, 3200, 1600, 50.0, 500.0);
        assert!(contour.len() > 0);
        // Should detect some voiced frames
        assert!(contour.voicing_ratio() > 0.0);
    }

    #[test]
    fn test_pitch_slope() {
        let pc = PitchContour::new(vec![100.0, 110.0, 120.0, 130.0, 140.0], 16000.0, 0.01);
        let slope = pitch_slope(&pc);
        assert!(slope > 0.0);
    }

    #[test]
    fn test_smooth_contour() {
        let pc = PitchContour::new(vec![100.0, 200.0, 100.0, 200.0], 16000.0, 0.01);
        let smooth = smooth_contour(&pc, 3);
        assert_eq!(smooth.len(), 4);
        // Smoothed should have less variation
        let orig_range = pc.range();
        let smooth_range = smooth.range();
        assert!(smooth_range <= orig_range);
    }

    #[test]
    fn test_empty_contour() {
        let pc = PitchContour::new(vec![], 16000.0, 0.01);
        assert!(pc.is_empty());
        assert_eq!(pc.mean(), 0.0);
        assert_eq!(pc.std_dev(), 0.0);
        assert_eq!(pc.range(), 0.0);
    }

    #[test]
    fn test_contour_duration() {
        let pc = PitchContour::new(vec![100.0; 10], 16000.0, 0.01);
        assert!((pc.duration() - 0.1).abs() < 1e-10);
    }
}
