// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Maika Namuo

use crate::dsp::AudioBlock;
use crate::util::audio::{
    DB_FLOOR, DEFAULT_SAMPLE_RATE, WindowKind, mixdown_into_deque, power_to_db,
    window_coefficients,
};
use realfft::{RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex32;
use std::collections::VecDeque;
use std::sync::Arc;

pub const NUM_PITCH_CLASSES: usize = 12;

pub const NOTE_NAMES: [&str; NUM_PITCH_CLASSES] =
    ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

const DEFAULT_HOP_SIZE: usize = 1024;
const DEFAULT_MIN_FREQ_HZ: f32 = 27.5; // A0 — full piano range
const DEFAULT_MAX_FREQ_HZ: f32 = 4186.0; // C8

// Multi-resolution FFT — one large window for the bass register, one small
// window for the treble.  This is the standard approach used in chroma
// filterbank implementations (cf. librosa's chroma_stft with variable
// n_fft, or any CQT-inspired design) to give genuine per-semitone frequency
// resolution throughout the full pitch range without unreasonable latency.
//
// Low band  (DEFAULT_MIN_FREQ_HZ – CROSSOVER_HZ):
//   16384-pt FFT at 44100 Hz → 2.69 Hz/bin
//   Semitone at A0  (27.5 Hz) = 1.64 Hz → 0.61 bins/semitone  (marginal, but
//     Gaussian weighting still gives useful results for most bass content)
//   Semitone at C2  (65.4 Hz) = 3.90 Hz → 1.45 bins/semitone  (reliable)
//   Semitone at C3 (130.8 Hz) = 7.79 Hz → 2.90 bins/semitone  (good)
//
// High band (CROSSOVER_HZ – DEFAULT_MAX_FREQ_HZ):
//   2048-pt FFT at 44100 Hz → 21.5 Hz/bin
//   Semitone at 1000 Hz = 59.5 Hz → 2.77 bins/semitone  (reliable)
//   Semitone at C8 (4186 Hz) = 249 Hz → 11.6 bins/semitone  (excellent)
//
// Update rate: both bands are computed on every hop (DEFAULT_HOP_SIZE samples,
// ~23 ms at 44100 Hz).  The large low-band window introduces ~372 ms of startup
// delay but contributes zero extra display latency once primed.
const LO_FFT_SIZE: usize = 16384;
const HI_FFT_SIZE: usize = 2048;

// Frequency at which the low band hands off to the high band.
// ~1000 Hz sits just below C6 (1047 Hz) so no pitch class straddles
// the crossover.
const CROSSOVER_HZ: f32 = 1000.0;

// Gaussian half-width (semitones) for bin-to-class weighting
const SIGMA_SEMITONES: f32 = 0.5;

pub const MIN_SMOOTHING: f32 = 0.01;
pub const MAX_SMOOTHING: f32 = 0.5;
pub const MIN_GATE_DB: f32 = -100.0;
pub const MAX_GATE_DB: f32 = -20.0;
pub const MIN_PEAK_DECAY: f32 = 0.900;
pub const MAX_PEAK_DECAY: f32 = 0.999;
pub const MIN_REFERENCE_HZ: f32 = 420.0;
pub const MAX_REFERENCE_HZ: f32 = 460.0;

// Keep the old name as an alias so the settings UI import keeps compiling.
pub const MIN_FLOOR_DB: f32 = MIN_GATE_DB;
pub const MAX_FLOOR_DB: f32 = MAX_GATE_DB;

#[derive(Debug, Clone, Copy)]
pub struct ChromaConfig {
    pub sample_rate: f32,
    /// 0 = slow, 1 = instant
    pub smoothing: f32,
    /// noise gate: bins below this level are excluded from accumulation
    pub floor_db: f32,
    /// < 1.0, applied every hop
    pub peak_decay: f32,
    /// Reference tuning for A4 in Hz (default 440.0)
    pub reference_hz: f32,
    // These are intentionally not exposed in settings or the UI; fixed sensible defaults.
    pub hop_size: usize,
    pub min_freq_hz: f32,
    pub max_freq_hz: f32,
}

impl Default for ChromaConfig {
    fn default() -> Self {
        Self {
            sample_rate: DEFAULT_SAMPLE_RATE,
            smoothing: 0.07,
            floor_db: -70.0,
            peak_decay: 0.997,
            reference_hz: 440.0,
            hop_size: DEFAULT_HOP_SIZE,
            min_freq_hz: DEFAULT_MIN_FREQ_HZ,
            max_freq_hz: DEFAULT_MAX_FREQ_HZ,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ChromaSnapshot {
    pub bins: [f32; NUM_PITCH_CLASSES],
    pub peak_bins: [f32; NUM_PITCH_CLASSES],
}

// Precomputed per-FFT-bin: power is split between the two nearest pitch classes
// using Gaussian weights (normalized per-class so each class has equal total
// sensitivity regardless of how many FFT bins fall near it).
//
// Inner tuple: (pitch_class_index, normalized_gaussian_weight)
type BinContrib = Option<[(u8, f32); 2]>;

/// Continuous pitch class (C=0, fractional) for a given frequency,
/// using a configurable A4 reference.
fn pitch_class_continuous(hz: f32, c0_hz: f64) -> Option<f32> {
    if hz <= 0.0 || !hz.is_finite() {
        return None;
    }
    Some(((12.0 * (hz as f64 / c0_hz).log2()).rem_euclid(12.0)) as f32)
}

/// C0 frequency derived from A4 reference (A4 is MIDI 69 = 9 semitones above C4
/// = 57 semitones above C0).
fn c0_from_reference(reference_hz: f32) -> f64 {
    reference_hz as f64 * 2.0_f64.powf(-69.0 / 12.0)
}

/// Precompute per-bin Gaussian pitch-class contributions for a single FFT band.
///
/// Two-pass: pass 1 computes raw Gaussian weights, pass 2 normalizes each
/// pitch class's total weight to 1.0 so that all 12 classes have equal
/// sensitivity regardless of the number of FFT bins that fall near each one.
fn precompute_bin_contribs(
    fft_size: usize,
    sample_rate: f32,
    min_freq: f32,
    max_freq: f32,
    reference_hz: f32,
) -> Box<[BinContrib]> {
    let spectrum_len = fft_size / 2 + 1;
    let bin_hz = sample_rate / fft_size as f32;
    let c0_hz = c0_from_reference(reference_hz);

    // Pass 1: raw Gaussian weights.
    let mut contribs: Vec<BinContrib> = (0..spectrum_len)
        .map(|b| {
            let hz = b as f32 * bin_hz;
            if hz < min_freq || hz > max_freq {
                return None;
            }
            let pc_cont = pitch_class_continuous(hz, c0_hz)?;

            let pc0 = (pc_cont.round() as usize).rem_euclid(NUM_PITCH_CLASSES);
            let dist0 = {
                let d = pc_cont - pc0 as f32;
                if d > 6.0 { d - 12.0 } else if d < -6.0 { d + 12.0 } else { d }
            };
            let pc1 = if dist0 >= 0.0 {
                (pc0 + 1) % NUM_PITCH_CLASSES
            } else {
                (pc0 + NUM_PITCH_CLASSES - 1) % NUM_PITCH_CLASSES
            };
            let dist1 = if dist0 >= 0.0 { dist0 - 1.0 } else { dist0 + 1.0 };

            let w0 = (-0.5 * (dist0 / SIGMA_SEMITONES).powi(2)).exp();
            let w1 = (-0.5 * (dist1 / SIGMA_SEMITONES).powi(2)).exp();

            Some([(pc0 as u8, w0), (pc1 as u8, w1)])
        })
        .collect();

    // Pass 2: normalize per pitch-class so each class has equal total weight.
    // Without this, treble classes (many bins/semitone) accumulate
    // disproportionately more energy than bass classes (few bins/semitone).
    let mut class_total = [0.0_f32; NUM_PITCH_CLASSES];
    for pairs in contribs.iter().flatten() {
        for &(pc, w) in pairs {
            class_total[pc as usize] += w;
        }
    }
    for pairs in contribs.iter_mut().flatten() {
        for (pc, w) in pairs.iter_mut() {
            let total = class_total[*pc as usize];
            if total > 0.0 {
                *w /= total;
            }
        }
    }

    contribs.into_boxed_slice()
}

// ─── Single FFT band ──────────────────────────────────────────────────────────

struct BandFft {
    fft: Arc<dyn RealToComplex<f32>>,
    fft_input: Vec<f32>,
    fft_output: Vec<Complex32>,
    scratch: Vec<Complex32>,
    window: Arc<[f32]>,
    bin_contribs: Box<[BinContrib]>,
    fft_size: usize,
    /// 1 / fft_size² — the Parseval-normalized power scale.
    /// Using 1/N² gives consistent absolute power across different FFT sizes:
    /// a unit-amplitude sine yields power ≈ 0.25 regardless of N.
    scale: f32,
}

impl BandFft {
    fn new(
        fft_size: usize,
        sample_rate: f32,
        min_freq: f32,
        max_freq: f32,
        reference_hz: f32,
    ) -> Self {
        let fft_size = fft_size.next_power_of_two().max(64);
        let fft = RealFftPlanner::new().plan_fft_forward(fft_size);
        let window = window_coefficients(WindowKind::Hann, fft_size);
        let bin_contribs =
            precompute_bin_contribs(fft_size, sample_rate, min_freq, max_freq, reference_hz);
        let fft_input = fft.make_input_vec();
        let fft_output = fft.make_output_vec();
        let scratch = fft.make_scratch_vec();
        Self {
            fft,
            fft_input,
            fft_output,
            scratch,
            window,
            bin_contribs,
            fft_size,
            scale: 1.0 / (fft_size as f32 * fft_size as f32),
        }
    }

    /// Fill the FFT input from the *tail* of `mono_buf` (most recent samples),
    /// apply the window, and run the FFT.  Returns false if the buffer is too
    /// short.
    fn process_from_tail(&mut self, mono_buf: &VecDeque<f32>) -> bool {
        if mono_buf.len() < self.fft_size {
            return false;
        }
        let start = mono_buf.len() - self.fft_size;
        let window = self.window.clone();
        for (dst, (src, &w)) in self
            .fft_input
            .iter_mut()
            .zip(mono_buf.iter().skip(start).zip(window.iter()))
        {
            *dst = src * w;
        }
        self.fft
            .process_with_scratch(&mut self.fft_input, &mut self.fft_output, &mut self.scratch)
            .is_ok()
    }

    /// Add this band's spectral energy into `class_energy`, skipping bins
    /// below the noise gate.
    fn accumulate(&self, class_energy: &mut [f32; NUM_PITCH_CLASSES], gate_db: f32) {
        for (contrib, output) in self.bin_contribs.iter().zip(self.fft_output.iter()) {
            let Some(pairs) = contrib else { continue };
            let power = output.norm_sqr() * self.scale;
            if power_to_db(power, DB_FLOOR) < gate_db {
                continue;
            }
            for &(pc, w) in pairs {
                class_energy[pc as usize] += power * w;
            }
        }
    }
}

// ─── Processor ───────────────────────────────────────────────────────────────

pub struct ChromaProcessor {
    config: ChromaConfig,
    /// Large-window FFT for the bass register (min_freq..CROSSOVER_HZ).
    band_lo: BandFft,
    /// Small-window FFT for the treble register (CROSSOVER_HZ..max_freq).
    band_hi: BandFft,
    /// Audio accumulation buffer — must hold at least LO_FFT_SIZE samples.
    mono_buf: VecDeque<f32>,
    smoothed_bins: [f32; NUM_PITCH_CLASSES],
    peak_bins: [f32; NUM_PITCH_CLASSES],
    snapshot: ChromaSnapshot,
}

impl ChromaProcessor {
    pub fn new(config: ChromaConfig) -> Self {
        let (band_lo, band_hi) = Self::make_bands(&config);
        Self {
            config,
            band_lo,
            band_hi,
            mono_buf: VecDeque::new(),
            smoothed_bins: [0.0; NUM_PITCH_CLASSES],
            peak_bins: [0.0; NUM_PITCH_CLASSES],
            snapshot: ChromaSnapshot::default(),
        }
    }

    pub fn config(&self) -> ChromaConfig {
        self.config
    }

    pub fn update_config(&mut self, config: ChromaConfig) {
        let rebuild = (config.sample_rate - self.config.sample_rate).abs() > f32::EPSILON
            || (config.min_freq_hz - self.config.min_freq_hz).abs() > 0.01
            || (config.max_freq_hz - self.config.max_freq_hz).abs() > 0.01
            || (config.reference_hz - self.config.reference_hz).abs() > 0.01;
        self.config = config;
        if rebuild {
            self.rebuild();
        }
    }

    fn make_bands(config: &ChromaConfig) -> (BandFft, BandFft) {
        let lo = BandFft::new(
            LO_FFT_SIZE,
            config.sample_rate,
            config.min_freq_hz,
            CROSSOVER_HZ,
            config.reference_hz,
        );
        let hi = BandFft::new(
            HI_FFT_SIZE,
            config.sample_rate,
            CROSSOVER_HZ,
            config.max_freq_hz,
            config.reference_hz,
        );
        (lo, hi)
    }

    fn rebuild(&mut self) {
        let (lo, hi) = Self::make_bands(&self.config);
        self.band_lo = lo;
        self.band_hi = hi;
        self.mono_buf.clear();
        self.smoothed_bins = [0.0; NUM_PITCH_CLASSES];
        self.peak_bins = [0.0; NUM_PITCH_CLASSES];
    }

    pub fn process_block(&mut self, block: &AudioBlock<'_>) -> Option<ChromaSnapshot> {
        if block.frame_count() == 0 {
            return None;
        }

        let sample_rate = block.sample_rate.max(1.0);
        if (sample_rate - self.config.sample_rate).abs() > f32::EPSILON {
            self.config.sample_rate = sample_rate;
            self.rebuild();
        }

        mixdown_into_deque(&mut self.mono_buf, block.samples, block.channels.max(1));

        let hop_size = self.config.hop_size.clamp(1, LO_FFT_SIZE);
        let mut any = false;

        // Process one hop at a time.  Both bands fire together; the large lo-band
        // window requires LO_FFT_SIZE samples in the buffer before we start.
        while self.mono_buf.len() >= LO_FFT_SIZE {
            let mut class_energy = [0.0_f32; NUM_PITCH_CLASSES];
            let gate_db = self.config.floor_db;

            if self.band_lo.process_from_tail(&self.mono_buf) {
                self.band_lo.accumulate(&mut class_energy, gate_db);
            }
            if self.band_hi.process_from_tail(&self.mono_buf) {
                self.band_hi.accumulate(&mut class_energy, gate_db);
            }

            // Max-normalize → sqrt for perceptual amplitude scaling → EMA smooth.
            let peak = class_energy.iter().copied().fold(0.0_f32, f32::max);
            let normalized: [f32; NUM_PITCH_CLASSES] = if peak > 0.0 {
                std::array::from_fn(|i| (class_energy[i] / peak).sqrt())
            } else {
                [0.0; NUM_PITCH_CLASSES]
            };

            let alpha = self.config.smoothing.clamp(MIN_SMOOTHING, MAX_SMOOTHING);
            let decay = self.config.peak_decay.clamp(MIN_PEAK_DECAY, MAX_PEAK_DECAY);

            for ((smoothed, peak_hold), norm) in self
                .smoothed_bins
                .iter_mut()
                .zip(self.peak_bins.iter_mut())
                .zip(normalized.iter())
            {
                *smoothed += alpha * (norm - *smoothed);
                *peak_hold = (*peak_hold * decay).max(*smoothed);
            }

            self.mono_buf.drain(..hop_size);
            any = true;
        }

        if any {
            self.snapshot.bins = self.smoothed_bins;
            self.snapshot.peak_bins = self.peak_bins;
        }
        Some(self.snapshot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: f32 = 48_000.0;

    fn block(samples: &[f32], channels: usize) -> AudioBlock<'_> {
        AudioBlock::now(samples, channels, RATE)
    }

    #[test]
    fn pitch_class_mapping() {
        let c0 = c0_from_reference(440.0);
        let pc = |hz: f32| pitch_class_continuous(hz, c0).map(|p| p.round() as usize % 12);
        assert_eq!(pc(261.63), Some(0)); // C4 -> C
        assert_eq!(pc(440.0), Some(9)); // A4 -> A
        assert_eq!(pc(659.26), Some(4)); // E5 -> E
    }

    #[test]
    fn reference_hz_shifts_pitch_classes() {
        let c0_432 = c0_from_reference(432.0);
        let pc =
            |hz: f32| pitch_class_continuous(hz, c0_432).map(|p| p.round() as usize % 12);
        assert_eq!(pc(432.0), Some(9));
        let cont = pitch_class_continuous(440.0, c0_432).unwrap();
        assert!(cont > 9.0, "440 Hz should be sharp of A when reference is 432");
    }

    #[test]
    fn process_sine_wave_returns_snapshot() {
        let mut proc = ChromaProcessor::new(ChromaConfig::default());
        // Need LO_FFT_SIZE + a few hops worth of samples to prime the buffer.
        let n = LO_FFT_SIZE + DEFAULT_HOP_SIZE * 4;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / RATE).sin())
            .collect();
        let snap = proc.process_block(&block(&samples, 1));
        assert!(snap.is_some());
    }

    #[test]
    fn a440_activates_a_pitch_class() {
        let mut proc = ChromaProcessor::new(ChromaConfig {
            smoothing: 1.0,
            ..Default::default()
        });
        // Enough samples to fill the lo-band window and produce at least one frame.
        let n = LO_FFT_SIZE + DEFAULT_HOP_SIZE;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / RATE).sin())
            .collect();
        let snap = proc.process_block(&block(&samples, 1)).unwrap();
        let a_class = snap.bins[9];
        let max_other = snap
            .bins
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != 9)
            .map(|(_, &v)| v)
            .fold(0.0_f32, f32::max);
        assert!(
            a_class > max_other,
            "A class ({a_class:.3}) should dominate other classes (max {max_other:.3})"
        );
    }

    #[test]
    fn noise_gate_suppresses_silence() {
        let mut proc = ChromaProcessor::new(ChromaConfig {
            smoothing: 1.0,
            floor_db: -60.0,
            ..Default::default()
        });
        let n = LO_FFT_SIZE + DEFAULT_HOP_SIZE;
        let samples: Vec<f32> = (0..n)
            .map(|i| 0.001 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / RATE).sin())
            .collect();
        let snap = proc.process_block(&block(&samples, 1)).unwrap();
        let max_bin = snap.bins.iter().copied().fold(0.0_f32, f32::max);
        assert!(max_bin < 0.01, "near-silent signal should not activate bins: {max_bin:.4}");
    }

    #[test]
    fn bass_e2_activates_e_pitch_class() {
        // E2 = 82.4 Hz — standard bass guitar low string. Verifies that the
        // low-frequency band resolves bass-register content.
        let mut proc = ChromaProcessor::new(ChromaConfig {
            smoothing: 1.0,
            ..Default::default()
        });
        let n = LO_FFT_SIZE + DEFAULT_HOP_SIZE;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 82.41 * i as f32 / RATE).sin())
            .collect();
        let snap = proc.process_block(&block(&samples, 1)).unwrap();
        // E = pitch class 4
        let e_class = snap.bins[4];
        let max_other = snap
            .bins
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != 4)
            .map(|(_, &v)| v)
            .fold(0.0_f32, f32::max);
        assert!(
            e_class > max_other,
            "E class ({e_class:.3}) should dominate for E2 (82.4 Hz); max other = {max_other:.3}"
        );
    }
}
