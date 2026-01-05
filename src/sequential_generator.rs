use std::sync::atomic::{AtomicU64, Ordering};

pub struct SequentialGenerator {
    current: AtomicU64,
}

impl SequentialGenerator {
    pub const fn new(start_at: u64) -> Self {
        Self {
            current: AtomicU64::new(start_at),
        }
    }

    pub fn next(&self) -> u64 {
        self.current.fetch_add(1, Ordering::SeqCst)
    }

    pub fn prev(&self) -> u64 {
        self.current.fetch_sub(1, Ordering::SeqCst)
    }
}
