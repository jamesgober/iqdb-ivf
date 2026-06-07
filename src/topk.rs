//! Bounded-heap top-`k` selection.
//!
//! Selects the smallest `k` entries from a slice of distances and returns
//! the chosen indices in best-first order. The selection runs in
//! `O(n log k)` using a max-heap of size `k`: the heap's root is the
//! worst-best entry seen so far, so a new entry is admitted iff it would
//! improve the heap.
//!
//! Ties are broken by *lower sequence number wins*. The caller passes a
//! `seqs: &[u64]` slice parallel to `distances`; the sequence number is a
//! monotonic insertion stamp the index assigns at `insert` time. Given two
//! candidates at the same distance, the one inserted first (smaller `seq`)
//! is considered better.
//!
//! `f32` ordering goes through [`f32::total_cmp`] — `Ord` on raw `f32` is
//! not provided by the standard library, and `partial_cmp` returns `None`
//! on NaN. `total_cmp` defines a total order that handles every payload a
//! distance computation might produce without panicking.
//!
//! ## Why a local copy
//!
//! This file is a verbatim copy of `iqdb-flat/src/topk.rs`, which is
//! `pub(crate)` and therefore not re-exportable. A future PR can lift
//! the helper into `iqdb-index` (or a dedicated `iqdb-topk` crate) and
//! delete both copies; tracked in `dev/ROADMAP.md`. Until then, IVF
//! avoids depending on the brute-force-search crate just to borrow a
//! sort helper.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// One scored candidate: a distance, the candidate's insertion-sequence
/// number, and its position in the caller's `distances` slice.
///
/// Ordering is `(dist, seq)`; `idx` is carried for output addressing only
/// and does not affect comparison.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Scored {
    pub(crate) dist: f32,
    pub(crate) seq: u64,
    pub(crate) idx: usize,
}

impl Scored {
    fn cmp_key(&self, other: &Self) -> Ordering {
        // Primary: smaller distance is "smaller" (better).
        // Secondary: smaller insertion sequence is "smaller" (better) — the
        // stable tiebreaker.
        self.dist
            .total_cmp(&other.dist)
            .then(self.seq.cmp(&other.seq))
    }
}

impl PartialEq for Scored {
    fn eq(&self, other: &Self) -> bool {
        self.cmp_key(other) == Ordering::Equal
    }
}
impl Eq for Scored {}
impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Scored {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cmp_key(other)
    }
}

/// Select the smallest `k` entries from `distances` and return their
/// indices, sorted best-first.
///
/// `seqs` MUST have the same length as `distances`; `seqs[i]` is the
/// insertion-sequence number of the candidate at position `i`. Ties on
/// `distances[i]` are broken by `seqs[i]` (smaller wins).
///
/// Returns an empty `Vec` when `k == 0` or `distances` is empty. If
/// `k > distances.len()`, returns every index in best-first order.
pub(crate) fn select_topk_indices(distances: &[f32], seqs: &[u64], k: usize) -> Vec<usize> {
    debug_assert_eq!(distances.len(), seqs.len(), "distances and seqs must align");
    if k == 0 || distances.is_empty() {
        return Vec::new();
    }
    let cap = k.min(distances.len());
    let mut heap: BinaryHeap<Scored> = BinaryHeap::with_capacity(cap);
    for (idx, (&dist, &seq)) in distances.iter().zip(seqs.iter()).enumerate() {
        let entry = Scored { dist, seq, idx };
        if heap.len() < cap {
            heap.push(entry);
        } else if heap.peek().is_some_and(|worst| entry < *worst) {
            let _evicted = heap.pop();
            heap.push(entry);
        }
    }
    heap.into_sorted_vec().into_iter().map(|s| s.idx).collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn seqs_in_order(n: usize) -> Vec<u64> {
        (0..n as u64).collect()
    }

    #[test]
    fn select_topk_zero_k_returns_empty() {
        let out = select_topk_indices(&[1.0, 2.0, 3.0], &seqs_in_order(3), 0);
        assert!(out.is_empty());
    }

    #[test]
    fn select_topk_empty_distances_returns_empty() {
        let out = select_topk_indices(&[], &[], 5);
        assert!(out.is_empty());
    }

    #[test]
    fn select_topk_k_greater_than_n_returns_all_sorted() {
        let out = select_topk_indices(&[3.0, 1.0, 2.0], &seqs_in_order(3), 10);
        assert_eq!(out, vec![1, 2, 0]);
    }

    #[test]
    fn select_topk_returns_best_first() {
        let out = select_topk_indices(&[5.0, 1.0, 4.0, 2.0, 3.0], &seqs_in_order(5), 3);
        assert_eq!(out, vec![1, 3, 4]);
    }

    #[test]
    fn select_topk_breaks_ties_by_lower_seq() {
        let out = select_topk_indices(&[1.0, 1.0, 1.0, 0.5], &[0, 1, 2, 3], 3);
        assert_eq!(out, vec![3, 0, 1]);
    }

    #[test]
    fn select_topk_tiebreaker_is_seq_not_idx() {
        let out = select_topk_indices(&[1.0, 1.0, 1.0, 1.0], &[1, 3, 0, 2], 4);
        assert_eq!(out, vec![2, 0, 3, 1]);
    }

    #[test]
    fn select_topk_handles_nan_via_total_cmp() {
        let out = select_topk_indices(&[f32::NAN, 1.0, 2.0], &seqs_in_order(3), 3);
        assert_eq!(out, vec![1, 2, 0]);
    }
}
