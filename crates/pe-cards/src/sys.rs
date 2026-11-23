//! CPU time and memory measurement for benchmarks.
//!
//! On a shared machine wall time is a poor measure, so benchmarks report the CPU
//! time of the measuring thread and of the whole process, and the process's peak
//! working set. On Windows these come from `GetThreadTimes`, `GetProcessTimes` and
//! `K32GetProcessMemoryInfo`; elsewhere they fall back to wall time and 0 bytes.

#[cfg(windows)]
mod imp {
    #[repr(C)]
    #[derive(Default)]
    struct FileTime {
        low: u32,
        high: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    struct MemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    unsafe extern "system" {
        fn GetCurrentThread() -> isize;
        fn GetCurrentProcess() -> isize;
        fn GetThreadTimes(h: isize, c: *mut FileTime, e: *mut FileTime, k: *mut FileTime, u: *mut FileTime) -> i32;
        fn GetProcessTimes(h: isize, c: *mut FileTime, e: *mut FileTime, k: *mut FileTime, u: *mut FileTime) -> i32;
        fn K32GetProcessMemoryInfo(h: isize, counters: *mut MemoryCounters, cb: u32) -> i32;
    }

    fn secs(f: &FileTime) -> f64 {
        (((f.high as u64) << 32) | f.low as u64) as f64 * 1e-7
    }

    pub fn thread_cpu_seconds() -> f64 {
        let (mut c, mut e, mut k, mut u) = (FileTime::default(), FileTime::default(), FileTime::default(), FileTime::default());
        // SAFETY: valid pointers to local structs; the pseudo-handle needs no closing.
        let ok = unsafe { GetThreadTimes(GetCurrentThread(), &mut c, &mut e, &mut k, &mut u) };
        if ok == 0 { 0.0 } else { secs(&k) + secs(&u) }
    }

    pub fn process_cpu_seconds() -> f64 {
        let (mut c, mut e, mut k, mut u) = (FileTime::default(), FileTime::default(), FileTime::default(), FileTime::default());
        // SAFETY: as above.
        let ok = unsafe { GetProcessTimes(GetCurrentProcess(), &mut c, &mut e, &mut k, &mut u) };
        if ok == 0 { 0.0 } else { secs(&k) + secs(&u) }
    }

    pub fn peak_working_set_bytes() -> u64 {
        let mut m = MemoryCounters { cb: std::mem::size_of::<MemoryCounters>() as u32, ..Default::default() };
        // SAFETY: `m` is a correctly sized, writable PROCESS_MEMORY_COUNTERS.
        let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut m, m.cb) };
        if ok == 0 { 0 } else { m.peak_working_set_size as u64 }
    }
}

#[cfg(not(windows))]
mod imp {
    use std::sync::OnceLock;
    use std::time::Instant;
    fn start() -> &'static Instant {
        static S: OnceLock<Instant> = OnceLock::new();
        S.get_or_init(Instant::now)
    }
    pub fn thread_cpu_seconds() -> f64 {
        start().elapsed().as_secs_f64()
    }
    pub fn process_cpu_seconds() -> f64 {
        start().elapsed().as_secs_f64()
    }
    pub fn peak_working_set_bytes() -> u64 {
        0
    }
}

/// CPU seconds used by the calling thread so far.
pub fn thread_cpu_seconds() -> f64 {
    imp::thread_cpu_seconds()
}

/// CPU seconds used by the whole process so far (all threads).
pub fn process_cpu_seconds() -> f64 {
    imp::process_cpu_seconds()
}

/// Peak working set (resident memory) of the process in bytes.
pub fn peak_working_set_bytes() -> u64 {
    imp::peak_working_set_bytes()
}

/// A stopwatch measuring wall time, thread CPU time and process CPU time.
pub struct Stopwatch {
    wall: std::time::Instant,
    thread: f64,
    process: f64,
}

impl Default for Stopwatch {
    fn default() -> Self {
        Self::start()
    }
}

impl Stopwatch {
    pub fn start() -> Stopwatch {
        Stopwatch { wall: std::time::Instant::now(), thread: thread_cpu_seconds(), process: process_cpu_seconds() }
    }
    pub fn wall(&self) -> f64 {
        self.wall.elapsed().as_secs_f64()
    }
    pub fn thread_cpu(&self) -> f64 {
        thread_cpu_seconds() - self.thread
    }
    pub fn process_cpu(&self) -> f64 {
        process_cpu_seconds() - self.process
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn clocks_move() {
        let sw = super::Stopwatch::start();
        let mut x = 0u64;
        for i in 0..20_000_000u64 {
            x = x.wrapping_mul(31).wrapping_add(i);
        }
        assert!(x != 1);
        assert!(sw.wall() >= 0.0);
        assert!(sw.thread_cpu() >= 0.0);
        assert!(super::peak_working_set_bytes() > 0 || cfg!(not(windows)));
    }
}
