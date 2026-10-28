//! Python's threading semantics the server is written in: Event, a Lock that any thread may release and that is
//! acquired with a timeout, and generators (a producer that runs only when its consumer asks for the next item, and
//! whose `close()` raises at the yield it waits at, so its cleanup runs before close() returns).

use std::sync::mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender};
use std::sync::{Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// threading.Event
#[derive(Default)]
pub struct Event {
    flag: Mutex<bool>,
    cv: Condvar,
}

impl Event {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn set(&self) {
        *self.flag.lock().unwrap() = true;
        self.cv.notify_all();
    }
    pub fn clear(&self) {
        *self.flag.lock().unwrap() = false;
    }
    pub fn is_set(&self) -> bool {
        *self.flag.lock().unwrap()
    }
    /// True once set, False after `timeout` (None: forever).
    pub fn wait(&self, timeout: Option<Duration>) -> bool {
        let g = self.flag.lock().unwrap();
        match timeout {
            None => *self.cv.wait_while(g, |f| !*f).unwrap(),
            Some(t) => *self.cv.wait_timeout_while(g, t, |f| !*f).unwrap().0,
        }
    }
}

/// threading.Lock: acquire with a timeout, release from any thread.
#[derive(Default)]
pub struct Sem {
    held: Mutex<bool>,
    cv: Condvar,
}

impl Sem {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn try_acquire(&self) -> bool {
        let mut h = self.held.lock().unwrap();
        if *h {
            return false;
        }
        *h = true;
        true
    }
    pub fn acquire(&self) {
        let mut h = self.cv.wait_while(self.held.lock().unwrap(), |h| *h).unwrap();
        *h = true;
    }
    pub fn acquire_timeout(&self, t: Duration) -> bool {
        let (mut h, _) = self.cv.wait_timeout_while(self.held.lock().unwrap(), t, |h| *h).unwrap();
        if *h {
            return false;
        }
        *h = true;
        true
    }
    pub fn release(&self) {
        *self.held.lock().unwrap() = false;
        self.cv.notify_one();
    }
    pub fn locked(&self) -> bool {
        *self.held.lock().unwrap()
    }
}

/// Held for a scope: released on drop.
pub struct SemGuard<'a>(pub &'a Sem);

impl Drop for SemGuard<'_> {
    fn drop(&mut self) {
        self.0.release();
    }
}

/// The consumer closed the generator (Python's GeneratorExit at the yield).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Closed;

/// How a generator body ended early: closed by its consumer, or an error of its own.
#[derive(Debug, Clone)]
pub enum Stop<E> {
    Closed,
    Err(E),
}

impl<E> From<Closed> for Stop<E> {
    fn from(_: Closed) -> Self {
        Stop::Closed
    }
}

enum Cmd {
    Next,
    Close,
}

enum Msg<T, E> {
    Item(T),
    Return(Result<(), Stop<E>>),
}

pub struct Yielder<T, E> {
    tx: SyncSender<Msg<T, E>>,
    rx: Receiver<Cmd>,
    closed: std::cell::Cell<bool>,
}

impl<T, E> Yielder<T, E> {
    /// `yield v`: hands `v` to the consumer and waits for it to ask again; Err(Closed) when it closed instead.
    pub fn yield_(&self, v: T) -> Result<(), Closed> {
        if self.closed.get() || self.tx.send(Msg::Item(v)).is_err() {
            self.closed.set(true);
            return Err(Closed);
        }
        match self.rx.recv() {
            Ok(Cmd::Next) => Ok(()),
            _ => {
                self.closed.set(true);
                Err(Closed)
            }
        }
    }
}

/// A Python generator: the body runs on its own thread in lockstep with the consumer.
pub struct Gen<T: Send + 'static, E: Send + 'static> {
    cmd: Option<SyncSender<Cmd>>,
    rx: Receiver<Msg<T, E>>,
    handle: Option<JoinHandle<()>>,
    finished: bool,
}

impl<T: Send + 'static, E: Send + 'static> Gen<T, E> {
    pub fn new<F>(body: F) -> Self
    where
        F: FnOnce(&Yielder<T, E>) -> Result<(), Stop<E>> + Send + 'static,
    {
        let (ctx, crx) = sync_channel::<Cmd>(1);
        let (itx, irx) = sync_channel::<Msg<T, E>>(1);
        let handle = std::thread::Builder::new()
            .name("strata-gen".into())
            .spawn(move || {
                // the body starts at the first next(); a generator closed before that never runs (as in Python)
                if !matches!(crx.recv(), Ok(Cmd::Next)) {
                    return;
                }
                let y = Yielder { tx: itx.clone(), rx: crx, closed: std::cell::Cell::new(false) };
                let r = body(&y);
                let _ = itx.send(Msg::Return(r));
            })
            .expect("spawn generator");
        Gen { cmd: Some(ctx), rx: irx, handle: Some(handle), finished: false }
    }

    /// next(gen): Ok(Some(item)), Ok(None) when the body returned, Err(e) when it raised.
    pub fn next(&mut self) -> Result<Option<T>, E> {
        if self.finished {
            return Ok(None);
        }
        let Some(cmd) = &self.cmd else { return Ok(None) };
        if cmd.send(Cmd::Next).is_err() {
            self.finish();
            return Ok(None);
        }
        match self.rx.recv() {
            Ok(Msg::Item(v)) => Ok(Some(v)),
            Ok(Msg::Return(r)) => {
                self.finish();
                match r {
                    Ok(()) | Err(Stop::Closed) => Ok(None),
                    Err(Stop::Err(e)) => Err(e),
                }
            }
            Err(_) => {
                self.finish();
                Ok(None)
            }
        }
    }

    /// next() with a deadline on the producer's answer (None: it is still working).
    pub fn next_timeout(&mut self, t: Duration) -> Option<Result<Option<T>, E>> {
        if self.finished {
            return Some(Ok(None));
        }
        let cmd = self.cmd.as_ref()?;
        if cmd.send(Cmd::Next).is_err() {
            self.finish();
            return Some(Ok(None));
        }
        self.wait_answer(t)
    }

    fn wait_answer(&mut self, t: Duration) -> Option<Result<Option<T>, E>> {
        match self.rx.recv_timeout(t) {
            Ok(Msg::Item(v)) => Some(Ok(Some(v))),
            Ok(Msg::Return(r)) => {
                self.finish();
                Some(match r {
                    Ok(()) | Err(Stop::Closed) => Ok(None),
                    Err(Stop::Err(e)) => Err(e),
                })
            }
            Err(RecvTimeoutError::Timeout) => None,
            Err(RecvTimeoutError::Disconnected) => {
                self.finish();
                Some(Ok(None))
            }
        }
    }

    /// Wait more for the answer of a next_timeout() that timed out.
    pub fn wait_more(&mut self, t: Duration) -> Option<Result<Option<T>, E>> {
        if self.finished {
            return Some(Ok(None));
        }
        self.wait_answer(t)
    }

    fn finish(&mut self) {
        self.finished = true;
        self.cmd = None;
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }

    /// gen.close(): the body's pending yield fails with Closed, its cleanup runs; an error the cleanup raises is
    /// returned (Python: an exception in a generator's finally propagates out of close()).
    pub fn close(&mut self) -> Result<(), E> {
        if self.finished {
            return Ok(());
        }
        let Some(cmd) = self.cmd.take() else { return Ok(()) };
        let _ = cmd.send(Cmd::Close);
        drop(cmd);
        let mut out = Ok(());
        loop {
            match self.rx.recv() {
                Ok(Msg::Item(_)) => continue, // yielded again after the close: it is closed again
                Ok(Msg::Return(Err(Stop::Err(e)))) => {
                    out = Err(e);
                    break;
                }
                Ok(Msg::Return(_)) | Err(_) => break,
            }
        }
        self.finished = true;
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        out
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }
}

impl<T: Send + 'static, E: Send + 'static> Drop for Gen<T, E> {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// Seconds since an Instant, as Python's time.monotonic() differences.
pub fn since(t: Instant) -> f64 {
    t.elapsed().as_secs_f64()
}

/// time.time()
pub fn now() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn a_generator_runs_in_lockstep_and_close_runs_its_cleanup() {
        let log = Arc::new(Mutex::new(vec![]));
        let l2 = log.clone();
        let mut g: Gen<i32, String> = Gen::new(move |y| {
            let r = (|| -> Result<(), Stop<String>> {
                for i in 0..10 {
                    l2.lock().unwrap().push(format!("make {i}"));
                    y.yield_(i)?;
                }
                Ok(())
            })();
            l2.lock().unwrap().push("cleanup".into());
            if matches!(r, Err(Stop::Closed)) {
                return Err(Stop::Err("raised in finally".into()));
            }
            r
        });
        assert_eq!(g.next().unwrap(), Some(0));
        assert_eq!(g.next().unwrap(), Some(1));
        assert_eq!(*log.lock().unwrap(), vec!["make 0", "make 1"]); // nothing ran ahead of the consumer
        assert_eq!(g.close(), Err("raised in finally".to_string()));
        assert_eq!(log.lock().unwrap().last().unwrap(), "cleanup");
        assert_eq!(g.next().unwrap(), None);
    }

    #[test]
    fn an_unstarted_generator_closes_without_running() {
        let ran = Arc::new(Mutex::new(false));
        let r2 = ran.clone();
        let mut g: Gen<i32, String> = Gen::new(move |_| {
            *r2.lock().unwrap() = true;
            Ok(())
        });
        assert_eq!(g.close(), Ok(()));
        assert!(!*ran.lock().unwrap());
    }

    #[test]
    fn sem_times_out_and_releases_across_threads() {
        let s = Arc::new(Sem::new());
        assert!(s.try_acquire());
        assert!(!s.acquire_timeout(Duration::from_millis(20)));
        let s2 = s.clone();
        std::thread::spawn(move || s2.release()).join().unwrap();
        assert!(s.acquire_timeout(Duration::from_millis(20)));
    }
}
