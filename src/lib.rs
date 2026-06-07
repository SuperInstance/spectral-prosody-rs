//! # spectral-prosody-rs
//!
//! Spectral analysis of prosodic features in agent communication.
//!
//! Provides pitch contour analysis, energy envelope tracking,
//! MFCC-inspired feature extraction, and rhythmic pattern analysis.

#![allow(clippy::needless_range_loop)]

pub mod energy;
pub mod pitch;
pub mod rhythm;
pub mod spectral_features;
