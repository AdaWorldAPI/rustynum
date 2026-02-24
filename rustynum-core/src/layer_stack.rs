//! Layer stack + collapse gate for multi-writer concurrent state.
//!
//! Ground truth is `&self` forever during processing cycles.
//! Each writer has their own delta layer with `&mut`.
//! The CollapseGate decides what to do with accumulated deltas:
//! - **Flow**: commit all deltas to ground truth (freeze / "ice-cake")
//! - **Hold**: keep deltas floating — accumulate across cycles
//! - **Block**: discard all deltas — irreconcilable contradiction

use crate::delta::DeltaLayer;
use crate::fingerprint::Fingerprint;

/// What to do with accumulated deltas.
///
/// The Collapse Gate evaluates conflict between deltas (via AND + popcount)
/// and decides whether to commit, hold, or discard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollapseGate {
    /// Commit deltas to ground truth (freeze / "ice-cake").
    /// Low conflict — writers modified independent regions.
    Flow,
    /// Keep deltas floating without committing.
    /// Ambiguous — accumulate more evidence across additional cycles.
    Hold,
    /// Discard all deltas. Ground truth unchanged.
    /// High conflict — writers contradicted each other.
    Block,
}

/// A stack of delta layers over immutable ground truth.
///
/// Ground truth is `&self` forever during processing. Each writer
/// gets their own delta layer with `&mut`. The collapse gate decides
/// what to do with accumulated changes.
///
/// ```text
/// effective[i] = ground ^ delta[i]
/// conflict(i,j) = delta[i] AND delta[j] → popcount > threshold
/// commit: ground ^= delta[0] ^ delta[1] ^ ...
/// ```
pub struct LayerStack<const N: usize> {
    ground: Fingerprint<N>,
    deltas: Vec<DeltaLayer<N>>,
}

impl<const N: usize> LayerStack<N> {
    /// Create a new stack with the given ground truth and N writer slots.
    pub fn new(initial: Fingerprint<N>, num_writers: usize) -> Self {
        let deltas = (0..num_writers)
            .map(|i| DeltaLayer::new(i as u32))
            .collect();
        Self {
            ground: initial,
            deltas,
        }
    }

    /// Borrow the ground truth (always immutable).
    #[inline]
    pub fn ground(&self) -> &Fingerprint<N> {
        &self.ground
    }

    /// Number of writer slots.
    #[inline]
    pub fn num_writers(&self) -> usize {
        self.deltas.len()
    }

    /// Get mutable access to a writer's delta layer.
    #[inline]
    pub fn writer_mut(&mut self, idx: usize) -> &mut DeltaLayer<N> {
        &mut self.deltas[idx]
    }

    /// Get immutable access to a writer's delta layer.
    #[inline]
    pub fn writer(&self, idx: usize) -> &DeltaLayer<N> {
        &self.deltas[idx]
    }

    /// Read a writer's current view (ground ^ their delta).
    #[inline]
    pub fn writer_view(&self, idx: usize) -> Fingerprint<N> {
        self.deltas[idx].read(&self.ground)
    }

    /// Add a new writer slot. Returns its index.
    pub fn add_writer(&mut self) -> usize {
        let idx = self.deltas.len();
        self.deltas.push(DeltaLayer::new(idx as u32));
        idx
    }

    /// Check for conflicts between any pair of deltas.
    pub fn has_conflicts(&self) -> bool {
        for i in 0..self.deltas.len() {
            for j in (i + 1)..self.deltas.len() {
                if self.deltas[i].conflicts_with(&self.deltas[j]) {
                    return true;
                }
            }
        }
        false
    }

    /// Maximum conflict (overlap popcount) between any pair of deltas.
    pub fn max_conflict_bits(&self) -> u32 {
        let mut max = 0u32;
        for i in 0..self.deltas.len() {
            for j in (i + 1)..self.deltas.len() {
                let bits = self.deltas[i].conflict_bits(&self.deltas[j]);
                if bits > max {
                    max = bits;
                }
            }
        }
        max
    }

    /// Evaluate the collapse gate based on conflict analysis.
    ///
    /// - `conflict_threshold`: max overlapping bits before blocking.
    pub fn evaluate(&self, conflict_threshold: u32) -> CollapseGate {
        let max_conflict = self.max_conflict_bits();

        if max_conflict > conflict_threshold {
            return CollapseGate::Block;
        }

        let total_change: u32 = self.deltas.iter().map(|d| d.changed_bits()).sum();
        if total_change == 0 {
            CollapseGate::Hold
        } else {
            CollapseGate::Flow
        }
    }

    /// FLOW: commit all deltas to ground truth (ice-cake).
    ///
    /// XOR is associative + commutative, so order doesn't matter.
    /// After commit, all deltas are cleared.
    pub fn commit(&mut self) {
        for delta in &self.deltas {
            self.ground ^= delta.delta();
        }
        self.clear();
    }

    /// BLOCK: discard all deltas. Ground truth unchanged.
    pub fn clear(&mut self) {
        for delta in &mut self.deltas {
            // Reset delta to zero by writing ground = ground
            let zero_patch = delta.delta().clone();
            delta.xor_patch(&zero_patch);
        }
    }

    /// Read the effective value through all deltas composed.
    ///
    /// `effective = ground ^ delta[0] ^ delta[1] ^ ...`
    pub fn read_all(&self) -> Fingerprint<N> {
        let mut result = self.ground.clone();
        for delta in &self.deltas {
            result ^= delta.delta();
        }
        result
    }

    /// Total number of changed bits across all deltas (union of diffs).
    pub fn total_changed_bits(&self) -> u32 {
        let effective = self.read_all();
        self.ground.hamming_distance(&effective)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_stack_reads_ground() {
        let ground = Fingerprint::<4> {
            words: [0xDEAD, 0xBEEF, 0xCAFE, 0xBABE],
        };
        let stack = LayerStack::new(ground.clone(), 3);
        assert_eq!(stack.num_writers(), 3);
        assert_eq!(*stack.ground(), ground);
        assert_eq!(stack.read_all(), ground);
    }

    #[test]
    fn test_writer_view_matches_write() {
        let ground = Fingerprint::<4>::zero();
        let desired = Fingerprint::<4> {
            words: [0xFF, 0, 0, 0],
        };
        let mut stack = LayerStack::new(ground, 2);
        let g = stack.ground().clone();
        stack.writer_mut(0).write(&g, &desired);
        assert_eq!(stack.writer_view(0), desired);
    }

    #[test]
    fn test_independent_writes_no_conflict() {
        let ground = Fingerprint::<4>::zero();
        let mut stack = LayerStack::new(ground.clone(), 2);

        let mut a = Fingerprint::<4>::zero();
        a.words[0] = 0xFF;
        stack.writer_mut(0).write(&ground, &a);

        let mut b = Fingerprint::<4>::zero();
        b.words[1] = 0xFF;
        stack.writer_mut(1).write(&ground, &b);

        assert!(!stack.has_conflicts());
        assert_eq!(stack.evaluate(0), CollapseGate::Flow);
    }

    #[test]
    fn test_overlapping_writes_conflict() {
        let ground = Fingerprint::<4>::zero();
        let mut stack = LayerStack::new(ground.clone(), 2);

        let mut desired = Fingerprint::<4>::zero();
        desired.words[0] = 0xFF;
        stack.writer_mut(0).write(&ground, &desired);
        stack.writer_mut(1).write(&ground, &desired);

        assert!(stack.has_conflicts());
        assert_eq!(stack.evaluate(0), CollapseGate::Block);
    }

    #[test]
    fn test_commit_merges_to_ground() {
        let ground = Fingerprint::<4>::zero();
        let mut stack = LayerStack::new(ground.clone(), 2);

        let mut a = Fingerprint::<4>::zero();
        a.words[0] = 0xFF;
        stack.writer_mut(0).write(&ground, &a);

        let mut b = Fingerprint::<4>::zero();
        b.words[1] = 0xFF00;
        stack.writer_mut(1).write(&ground, &b);

        stack.commit();

        assert_eq!(stack.ground().words[0], 0xFF);
        assert_eq!(stack.ground().words[1], 0xFF00);
        // After commit, all deltas cleared
        assert!(stack.writer(0).is_clean());
        assert!(stack.writer(1).is_clean());
    }

    #[test]
    fn test_clear_discards_deltas() {
        let ground = Fingerprint::<4>::zero();
        let mut stack = LayerStack::new(ground.clone(), 2);

        let mut desired = Fingerprint::<4>::zero();
        desired.words[0] = 0xFF;
        stack.writer_mut(0).write(&ground, &desired);

        stack.clear();

        assert!(stack.writer(0).is_clean());
        assert_eq!(stack.read_all(), ground);
    }

    #[test]
    fn test_no_changes_is_hold() {
        let ground = Fingerprint::<4>::zero();
        let stack = LayerStack::new(ground, 2);
        assert_eq!(stack.evaluate(10), CollapseGate::Hold);
    }

    #[test]
    fn test_ground_never_mutated_by_writes() {
        let ground = Fingerprint::<4> {
            words: [1, 2, 3, 4],
        };
        let ground_copy = ground.clone();
        let mut stack = LayerStack::new(ground, 3);

        for i in 0..3 {
            let mut val = Fingerprint::<4>::zero();
            val.words[i] = 0xFFFF;
            let g = stack.ground().clone();
            stack
                .writer_mut(i)
                .write(&g, &val);
        }

        assert_eq!(*stack.ground(), ground_copy);
    }

    #[test]
    fn test_total_changed_bits() {
        let ground = Fingerprint::<2>::zero();
        let mut stack = LayerStack::new(ground.clone(), 2);

        // Writer 0: set 8 bits in word 0
        stack
            .writer_mut(0)
            .xor_patch(&Fingerprint { words: [0xFF, 0] });

        // Writer 1: set 4 bits in word 1
        stack
            .writer_mut(1)
            .xor_patch(&Fingerprint { words: [0, 0x0F] });

        assert_eq!(stack.total_changed_bits(), 12);
    }

    #[test]
    fn test_add_writer() {
        let ground = Fingerprint::<2>::zero();
        let mut stack = LayerStack::new(ground, 1);
        assert_eq!(stack.num_writers(), 1);

        let idx = stack.add_writer();
        assert_eq!(idx, 1);
        assert_eq!(stack.num_writers(), 2);
        assert!(stack.writer(idx).is_clean());
    }
}
