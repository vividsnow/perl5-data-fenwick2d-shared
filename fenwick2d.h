/*
 * fenwick2d.h -- Shared-memory 2-D Fenwick tree (binary indexed tree) for Linux
 *
 * A fixed rows x cols grid of signed int64 cells supporting O(log rows*log cols)
 * point update and rectangle-sum query: add a delta at cell (x,y), ask the sum
 * over the origin rectangle [1..x]x[1..y] or any axis-aligned rectangle.  The
 * grid lives in a shared mapping so several processes update and query one
 * structure; a write-preferring futex rwlock with reader-slot dead-process
 * recovery guards mutation.
 *
 * Layout: Header -> reader_slots[1024] -> grid[(rows+1)*(cols+1) int64]
 *         (row-major, 1-indexed; row 0 and col 0 unused)
 */

#ifndef F2D_H
#define F2D_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <sys/syscall.h>
#include <linux/futex.h>
#include <pthread.h>

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#error "fenwick2d.h: requires little-endian architecture"
#endif


/* ================================================================
 * Constants
 * ================================================================ */

#define F2D_MAGIC        0x42443246U  /* Fenwick2D */
#define F2D_VERSION      1
#define F2D_ERR_BUFLEN   256
#ifndef F2D_READER_SLOTS
#define F2D_READER_SLOTS 1024         /* max concurrent reader processes for dead-process recovery */
#endif
#define F2D_MIN_DIM      1
#define F2D_MAX_DIM      0x1000000ULL   /* 2^24 per side; grid is (rows+1)*(cols+1) int64 */

#define F2D_ERR(fmt, ...) do { if (errbuf) snprintf(errbuf, F2D_ERR_BUFLEN, fmt, ##__VA_ARGS__); } while (0)

/* ================================================================
 * Structs
 * ================================================================ */

/* Per-process slot for dead-process recovery.  Each shared rwlock counter
 * (the main rwlock-reader count, rwlock_waiters, rwlock_writers_waiting)
 * is mirrored here so a wrlock timeout can attribute and reverse a dead
 * process's contribution instead of waiting for the slow per-op timeout
 * drain. */
typedef struct {
    uint32_t pid;            /* 0 = unclaimed */
    uint32_t subcount;       /* in-flight rdlock acquisitions for this process */
    uint32_t waiters_parked; /* contribution to hdr->rwlock_waiters         */
    uint32_t writers_parked; /* contribution to hdr->rwlock_writers_waiting */
} F2dReaderSlot;

struct F2dHeader {
    uint32_t magic, version;          /* 0,4 */
    uint32_t _pad0;                   /* 8 */
    uint32_t _pad1;                   /* 12 */
    uint64_t rows;                    /* 16  grid rows (1..rows), 1-indexed */
    uint64_t cols;                    /* 24  grid cols (1..cols), 1-indexed */
    uint64_t total_size;              /* 32 */
    uint64_t reader_slots_off;        /* 40 */
    uint64_t tree_off;                /* 48  (rows+1)*(cols+1) int64 grid, row-major */
    uint32_t rwlock;                  /* 56 */
    uint32_t rwlock_waiters;          /* 60 */
    uint32_t rwlock_writers_waiting;  /* 64 */
    uint32_t slotless_readers;  /* live readers holding the lock with no reader-slot (was padding) */
    uint64_t stat_ops;                /* 72 */
    uint8_t  _pad[176];               /* 80..255 */
};
typedef struct F2dHeader F2dHeader;

_Static_assert(sizeof(F2dHeader) == 256, "F2dHeader must be 256 bytes");

/* ---- Process-local handle ---- */

typedef struct F2dHandle {
    F2dHeader     *hdr;
    F2dReaderSlot *reader_slots;  /* F2D_READER_SLOTS entries */
    void         *base;          /* mmap base */
    uint64_t      tree_off;      /* validated tree-array offset, cached: never re-read from the peer-writable header */
    uint64_t      rows;          /* cached grid rows */
    uint64_t      cols;          /* cached grid cols */
    size_t        mmap_size;
    char         *path;          /* backing file path (strdup'd) */
    int           backing_fd;    /* memfd or reopened-fd to close on destroy, -1 for file/anon */
    uint32_t      my_slot_idx;   /* UINT32_MAX if all slots taken (no recovery for this handle) */
    uint32_t      cached_pid;    /* getpid() cached at last slot claim */
    uint32_t      cached_fork_gen; /* f2d_fork_gen value at last slot claim */
    uint32_t slotless_held; /* rwlock read-locks held with no reader-slot */
} F2dHandle;

/* ================================================================
 * Futex-based write-preferring read-write lock
 * with reader-slot dead-process recovery
 * ================================================================ */

#define F2D_RWLOCK_SPIN_LIMIT 32
#define F2D_LOCK_TIMEOUT_SEC  2  /* FUTEX_WAIT timeout for stale lock detection */

static inline void f2d_rwlock_spin_pause(void) {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ volatile("pause" ::: "memory");
#elif defined(__aarch64__)
    __asm__ volatile("yield" ::: "memory");
#else
    __asm__ volatile("" ::: "memory");
#endif
}

/* Extract writer PID from rwlock value (lower 31 bits when write-locked). */
#define F2D_RWLOCK_WRITER_BIT 0x80000000U
#define F2D_RWLOCK_PID_MASK   0x7FFFFFFFU
#define F2D_RWLOCK_WR(pid)    (F2D_RWLOCK_WRITER_BIT | ((uint32_t)(pid) & F2D_RWLOCK_PID_MASK))

/* Check if a PID is alive. Returns 1 if alive or unknown, 0 if definitely dead. */
/* Liveness via kill(pid,0). NOTE: cannot detect PID reuse -- if a dead
 * lock-holder's PID is recycled to an unrelated live process before recovery
 * runs, this reports "alive" and that slot's orphaned contribution is not
 * reclaimed until the recycled process exits. Robust detection would require
 * a per-slot process-start-time epoch (a header-layout/version change).
 * Documented under "Crash Safety" in the POD. */
static inline int f2d_pid_alive(uint32_t pid) {
    if (pid == 0) return 1; /* no owner recorded, assume alive */
    return !(kill((pid_t)pid, 0) == -1 && errno == ESRCH);
}

/* Force-recover a stale write lock left by a dead process.
 * CAS to OUR pid to hold the lock while fixing shared state, then release.
 * Using our pid (not a bare WRITER_BIT sentinel) means a subsequent
 * recovering process can detect and re-recover if we crash mid-recovery. */
static inline void f2d_recover_stale_lock(F2dHandle *h, uint32_t observed_rwlock) {
    F2dHeader *hdr = h->hdr;
    uint32_t mypid = F2D_RWLOCK_WR((uint32_t)getpid());
    if (!__atomic_compare_exchange_n(&hdr->rwlock, &observed_rwlock,
            mypid, 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
        return;
    /* We now hold the write lock as mypid.  No additional shared state needs
     * repair here (this module has no seqlock); just release the lock. */
    __atomic_store_n(&hdr->rwlock, 0, __ATOMIC_RELEASE);
    if (__atomic_load_n(&hdr->rwlock_waiters, __ATOMIC_RELAXED) > 0)
        syscall(SYS_futex, &hdr->rwlock, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
}

static const struct timespec f2d_lock_timeout = { F2D_LOCK_TIMEOUT_SEC, 0 };

/* Process-global fork-generation counter.  Incremented in the pthread_atfork
 * child callback so every open handle detects a fork transition on the next
 * lock call without paying a getpid() syscall on the hot path. */
static uint32_t f2d_fork_gen = 1;
static pthread_once_t f2d_atfork_once = PTHREAD_ONCE_INIT;
static void f2d_on_fork_child(void) {
    __atomic_add_fetch(&f2d_fork_gen, 1, __ATOMIC_RELAXED);
}
static void f2d_atfork_init(void) {
    pthread_atfork(NULL, NULL, f2d_on_fork_child);
}

/* Ensure this process owns a reader slot.  Called from the lock helpers so
 * that fork()'d children pick up their own slot lazily instead of sharing
 * the parent's.  Hot-path is a single relaxed load + compare; only on a
 * fork-generation mismatch do we touch getpid() and scan slots. */
static inline void f2d_claim_reader_slot(F2dHandle *h) {
    uint32_t cur_gen = __atomic_load_n(&f2d_fork_gen, __ATOMIC_RELAXED);
    if (__builtin_expect(cur_gen == h->cached_fork_gen && h->my_slot_idx != UINT32_MAX, 1))
        return;
    /* Cold path -- register the atfork hook once per process, then claim. */
    pthread_once(&f2d_atfork_once, f2d_atfork_init);
    /* Re-read after pthread_once: f2d_on_fork_child may have bumped it. */
    cur_gen = __atomic_load_n(&f2d_fork_gen, __ATOMIC_RELAXED);
    uint32_t now_pid = (uint32_t)getpid();
    h->cached_pid = now_pid;
    if (cur_gen != h->cached_fork_gen) h->slotless_held = 0;  /* fork: child holds none of the parent's slotless read locks */
    h->cached_fork_gen = cur_gen;
    h->my_slot_idx = UINT32_MAX;
    uint32_t start = now_pid % F2D_READER_SLOTS;
    for (uint32_t i = 0; i < F2D_READER_SLOTS; i++) {
        uint32_t s = (start + i) % F2D_READER_SLOTS;
        uint32_t expected = 0;
        if (__atomic_compare_exchange_n(&h->reader_slots[s].pid,
                &expected, now_pid, 0,
                __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
            /* Zero all mirror fields, not just subcount: a SIGKILL'd
             * predecessor may have left waiters_parked/writers_parked
             * non-zero, and f2d_recover_dead_readers won't drain them
             * once we own the slot (the CAS expects the dead PID). */
            __atomic_store_n(&h->reader_slots[s].subcount, 0, __ATOMIC_RELAXED);
            __atomic_store_n(&h->reader_slots[s].waiters_parked, 0, __ATOMIC_RELAXED);
            __atomic_store_n(&h->reader_slots[s].writers_parked, 0, __ATOMIC_RELAXED);
            h->my_slot_idx = s;
            return;
        }
    }
    /* Table full -- leave my_slot_idx = UINT32_MAX so we silently skip
     * tracking for this handle (lock still works; just no recovery). */
}

/* Atomically subtract `sub` from a counter, capped at 0 (never underflows). */
static inline void f2d_atomic_sub_cap(uint32_t *p, uint32_t sub) {
    if (!sub) return;
    uint32_t cur = __atomic_load_n(p, __ATOMIC_RELAXED);
    for (;;) {
        uint32_t want = (cur > sub) ? cur - sub : 0;
        if (__atomic_compare_exchange_n(p, &cur, want,
                1, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
            return;
    }
}

/* Try to claim a dead slot (CAS pid -> 0) and drain its parked-waiter
 * contributions back to the global counters.  A no-op if the slot was stolen
 * by another recoverer or had no waiter contribution to drain.
 *
 * Note: subcount/waiters_parked/writers_parked are NOT zeroed here.
 * Between our CAS and a follow-up store, a new process could claim the
 * slot and start populating these fields -- our stores would clobber its
 * state.  f2d_claim_reader_slot zeros all three on every claim, so
 * leaving stale values is harmless. */
static inline void f2d_drain_dead_slot(F2dHandle *h, uint32_t i, uint32_t pid) {
    F2dHeader *hdr = h->hdr;
    uint32_t expected = pid;
    /* ACQ_REL on success: RELEASE publishes pid=0 to other observers;
     * ACQUIRE syncs us with prior writes from the dead process to
     * waiters_parked/writers_parked.  On weakly-ordered archs (aarch64)
     * a plain RELAXED load before the CAS could miss those writes;
     * loading them after the CAS keeps them inside the acquire window. */
    if (!__atomic_compare_exchange_n(&h->reader_slots[i].pid, &expected, 0,
            0, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
        return;
    uint32_t wp    = __atomic_load_n(&h->reader_slots[i].waiters_parked, __ATOMIC_RELAXED);
    uint32_t writp = __atomic_load_n(&h->reader_slots[i].writers_parked, __ATOMIC_RELAXED);
    if (wp)    f2d_atomic_sub_cap(&hdr->rwlock_waiters, wp);
    if (writp) f2d_atomic_sub_cap(&hdr->rwlock_writers_waiting, writp);
}

/* Scan reader slots for dead-process recovery.
 *
 * For each dead PID with non-zero contributions to the shared rwlock,
 * rwlock_waiters, or rwlock_writers_waiting counters, drain its share back
 * out so live processes don't have to wait for the slow per-op timeout
 * decrement to drain it for them.
 *
 * For the main rwlock counter we use the "no live reader holds -> force-
 * reset to 0" trick (precise) because per-process attribution of the
 * subcount is racy across the inc-counter-then-inc-subcount window. */
static inline void f2d_recover_dead_readers(F2dHandle *h) {
    if (!h->reader_slots) return;
    F2dHeader *hdr = h->hdr;
    int any_live_reader = 0;
    int found_dead_reader = 0;

    /* Pass 1: classify slots.  Slots with dead pid and sc == 0 (no rwlock
     * contribution to lose) are wiped immediately to free the slot for
     * future claimants and drain any orphan parked-waiter counters.  Slots
     * with dead pid and sc > 0 are left intact in this pass: if force-
     * reset cannot fire (because a live reader is concurrently present),
     * wiping the dead slot would lose the only record of its orphan
     * rwlock contribution and strand writers permanently once the live
     * reader releases. */
    for (uint32_t i = 0; i < F2D_READER_SLOTS; i++) {
        uint32_t pid = __atomic_load_n(&h->reader_slots[i].pid, __ATOMIC_ACQUIRE);
        if (pid == 0) continue;
        uint32_t sc = __atomic_load_n(&h->reader_slots[i].subcount, __ATOMIC_RELAXED);
        if (f2d_pid_alive(pid)) {
            if (sc > 0) any_live_reader = 1;
            continue;
        }
        if (sc > 0) { found_dead_reader = 1; continue; }
        f2d_drain_dead_slot(h, i, pid);
    }

    /* Pass 2: only if force-reset will fire.  Issue the rwlock force-
     * reset CAS FIRST, while the window since pass 1's last scan is
     * still narrow (a handful of instructions, as in the original
     * single-pass code).  A new reader that started rdlock between
     * pass 1's scan and the CAS will either:
     *   (a) have already CAS'd rwlock from cur to cur+1 -- our CAS then
     *       fails (cur mismatched), recovery yields and a future
     *       cycle retries; or
     *   (b) be still in the subcount-bump phase -- our CAS sees the
     *       stale cur and resets to 0; the new reader's subsequent CAS
     *       rwlock(0 -> 1) succeeds cleanly.
     * Only after the CAS resolves do we wipe the deferred dead slots,
     * keeping that work outside the race-sensitive window. */
    /* A live reader with no slot (table was full) is invisible to the scan
     * above but still holds a +1 in the lock word; never force-reset under it. */
    if (__atomic_load_n(&hdr->slotless_readers, __ATOMIC_RELAXED) > 0)
        any_live_reader = 1;
    if (found_dead_reader && !any_live_reader) {
        /* ACQUIRE: a late reader's subcount++ (before its rwlock CAS) is then visible below. */
        uint32_t cur = __atomic_load_n(&hdr->rwlock, __ATOMIC_ACQUIRE);
        int drain_ok = 1;   /* keep dead slots if the reset doesn't fire */
        if (cur > 0 && cur < F2D_RWLOCK_WRITER_BIT) {
            /* Re-scan for a live reader (fail-safe: only suppresses a reset). */
            int live_now = __atomic_load_n(&hdr->slotless_readers, __ATOMIC_RELAXED) > 0;
            for (uint32_t i = 0; !live_now && i < F2D_READER_SLOTS; i++) {
                uint32_t p = __atomic_load_n(&h->reader_slots[i].pid, __ATOMIC_ACQUIRE);
                if (p && f2d_pid_alive(p) &&
                    __atomic_load_n(&h->reader_slots[i].subcount, __ATOMIC_RELAXED) > 0)
                    live_now = 1;
            }
            if (live_now) {
                drain_ok = 0;
            } else if (__atomic_compare_exchange_n(&hdr->rwlock, &cur, 0,
                    0, __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {
                if (__atomic_load_n(&hdr->rwlock_waiters, __ATOMIC_RELAXED) > 0)
                    syscall(SYS_futex, &hdr->rwlock, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
            } else {
                drain_ok = 0;   /* rwlock changed under us -- shares may still be live */
            }
        }
        if (drain_ok) {
            for (uint32_t i = 0; i < F2D_READER_SLOTS; i++) {
                uint32_t p = __atomic_load_n(&h->reader_slots[i].pid, __ATOMIC_ACQUIRE);
                if (p == 0 || f2d_pid_alive(p)) continue;
                f2d_drain_dead_slot(h, i, p);
            }
        }
    }
}

/* Inspect the lock word after a futex-wait timeout.  If a dead writer
 * holds it, force-recover the lock.  Otherwise drain dead readers' shares
 * of the rwlock/waiter counters.  Called from rdlock and wrlock ETIMEDOUT
 * branches -- identical recovery logic in both. */
static inline void f2d_recover_after_timeout(F2dHandle *h) {
    F2dHeader *hdr = h->hdr;
    uint32_t val = __atomic_load_n(&hdr->rwlock, __ATOMIC_RELAXED);
    if (val >= F2D_RWLOCK_WRITER_BIT) {
        uint32_t pid = val & F2D_RWLOCK_PID_MASK;
        if (!f2d_pid_alive(pid))
            f2d_recover_stale_lock(h, val);
    } else {
        f2d_recover_dead_readers(h);
    }
}

/* Park/unpark helpers: bump the global waiter counters together with this
 * process's mirrored slot counters so a wrlock-timeout recovery scan can
 * attribute and reverse a dead PID's contribution.  Kept paired to make
 * accidental drift between global and per-slot counts impossible. */
static inline void f2d_park_reader(F2dHandle *h) {
    if (h->my_slot_idx != UINT32_MAX)
        __atomic_add_fetch(&h->reader_slots[h->my_slot_idx].waiters_parked, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&h->hdr->rwlock_waiters, 1, __ATOMIC_RELAXED);
}
static inline void f2d_unpark_reader(F2dHandle *h) {
    __atomic_sub_fetch(&h->hdr->rwlock_waiters, 1, __ATOMIC_RELAXED);
    if (h->my_slot_idx != UINT32_MAX)
        __atomic_sub_fetch(&h->reader_slots[h->my_slot_idx].waiters_parked, 1, __ATOMIC_RELAXED);
}
static inline void f2d_park_writer(F2dHandle *h) {
    if (h->my_slot_idx != UINT32_MAX) {
        __atomic_add_fetch(&h->reader_slots[h->my_slot_idx].waiters_parked, 1, __ATOMIC_RELAXED);
        __atomic_add_fetch(&h->reader_slots[h->my_slot_idx].writers_parked, 1, __ATOMIC_RELAXED);
    }
    __atomic_add_fetch(&h->hdr->rwlock_waiters, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&h->hdr->rwlock_writers_waiting, 1, __ATOMIC_RELAXED);
}
static inline void f2d_unpark_writer(F2dHandle *h) {
    __atomic_sub_fetch(&h->hdr->rwlock_waiters, 1, __ATOMIC_RELAXED);
    __atomic_sub_fetch(&h->hdr->rwlock_writers_waiting, 1, __ATOMIC_RELAXED);
    if (h->my_slot_idx != UINT32_MAX) {
        __atomic_sub_fetch(&h->reader_slots[h->my_slot_idx].waiters_parked, 1, __ATOMIC_RELAXED);
        __atomic_sub_fetch(&h->reader_slots[h->my_slot_idx].writers_parked, 1, __ATOMIC_RELAXED);
    }
}

/* Reader accounting: a reader mirrors its +1 in the lock word so dead-reader
 * recovery can see it. A slotted reader uses its slot subcount; a reader that
 * could not claim a slot (table full) uses the global hdr->slotless_readers,
 * so recovery's force-reset never fires out from under it. leave() peels
 * slotless first so a later slot claim cannot misattribute the decrement. */
static inline void f2d_reader_enter(F2dHandle *h) {
    if (h->my_slot_idx != UINT32_MAX) {
        __atomic_add_fetch(&h->reader_slots[h->my_slot_idx].subcount, 1, __ATOMIC_RELAXED);
    } else {
        __atomic_add_fetch(&h->hdr->slotless_readers, 1, __ATOMIC_RELAXED);
        h->slotless_held++;
    }
}
static inline void f2d_reader_leave(F2dHandle *h) {
    if (h->slotless_held > 0) {
        h->slotless_held--;
        __atomic_sub_fetch(&h->hdr->slotless_readers, 1, __ATOMIC_RELAXED);
    } else if (h->my_slot_idx != UINT32_MAX) {
        __atomic_sub_fetch(&h->reader_slots[h->my_slot_idx].subcount, 1, __ATOMIC_RELAXED);
    }
}

static inline void f2d_rwlock_rdlock(F2dHandle *h) {
    f2d_claim_reader_slot(h);
    F2dHeader *hdr = h->hdr;
    uint32_t *lock = &hdr->rwlock;
    uint32_t *writers_waiting = &hdr->rwlock_writers_waiting;
    /* Claim subcount BEFORE bumping the shared rwlock counter.  This way
     * a concurrent writer-side recovery scan that sees our PID alive with
     * subcount > 0 will (correctly) defer force-reset, even while we are
     * still spinning trying to win the rwlock CAS.  Without this, a reader
     * killed between rwlock CAS-success and subcount++ would let recovery
     * force-reset rwlock to 0 underneath us, causing a UINT32_MAX wrap on
     * our eventual rdunlock dec. */
    f2d_reader_enter(h);
    for (int spin = 0; ; spin++) {
        uint32_t cur = __atomic_load_n(lock, __ATOMIC_RELAXED);
        /* Write-preferring: when lock is free (cur==0) and writers are
         * waiting, yield to let the writer acquire. When readers are
         * already active (cur>=1), new readers may join freely. */
        if (cur > 0 && cur < F2D_RWLOCK_WRITER_BIT) {
            if (__atomic_compare_exchange_n(lock, &cur, cur + 1,
                    1, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
                return;
        } else if (cur == 0 && !__atomic_load_n(writers_waiting, __ATOMIC_RELAXED)) {
            if (__atomic_compare_exchange_n(lock, &cur, 1,
                    1, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
                return;
        }
        if (__builtin_expect(spin < F2D_RWLOCK_SPIN_LIMIT, 1)) {
            f2d_rwlock_spin_pause();
            continue;
        }
        f2d_park_reader(h);
        cur = __atomic_load_n(lock, __ATOMIC_RELAXED);
        /* Sleep when write-locked OR when yielding to waiting writers */
        if (cur >= F2D_RWLOCK_WRITER_BIT || cur == 0) {
            long rc = syscall(SYS_futex, lock, FUTEX_WAIT, cur,
                              &f2d_lock_timeout, NULL, 0);
            if (rc == -1 && errno == ETIMEDOUT) {
                f2d_unpark_reader(h);
                f2d_recover_after_timeout(h);
                spin = 0;
                continue;
            }
        }
        f2d_unpark_reader(h);
        spin = 0;
    }
}

static inline void f2d_rwlock_rdunlock(F2dHandle *h) {
    F2dHeader *hdr = h->hdr;
    /* Release the shared counter BEFORE dropping our subcount so that
     * "any live PID with subcount > 0" is a reliable in-flight indicator
     * for the writer-side recovery scan.  Inverting these would create a
     * window where we still own a unit of rwlock but our slot subcount is
     * 0, letting recovery force-reset rwlock underneath us. */
    uint32_t after = __atomic_sub_fetch(&hdr->rwlock, 1, __ATOMIC_RELEASE);
    f2d_reader_leave(h);
    if (after == 0 && __atomic_load_n(&hdr->rwlock_waiters, __ATOMIC_RELAXED) > 0)
        syscall(SYS_futex, &hdr->rwlock, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
}

static inline void f2d_rwlock_wrlock(F2dHandle *h) {
    f2d_claim_reader_slot(h);  /* refresh cached_pid across fork */
    F2dHeader *hdr = h->hdr;
    uint32_t *lock = &hdr->rwlock;
    /* Encode PID in the rwlock word itself (0x80000000 | pid) to eliminate
     * any crash window between acquiring the lock and storing the owner. */
    uint32_t mypid = F2D_RWLOCK_WR(h->cached_pid);
    for (int spin = 0; ; spin++) {
        uint32_t expected = 0;
        if (__atomic_compare_exchange_n(lock, &expected, mypid,
                1, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
            return;
        if (__builtin_expect(spin < F2D_RWLOCK_SPIN_LIMIT, 1)) {
            f2d_rwlock_spin_pause();
            continue;
        }
        f2d_park_writer(h);
        uint32_t cur = __atomic_load_n(lock, __ATOMIC_RELAXED);
        if (cur != 0) {
            long rc = syscall(SYS_futex, lock, FUTEX_WAIT, cur,
                              &f2d_lock_timeout, NULL, 0);
            if (rc == -1 && errno == ETIMEDOUT) {
                f2d_unpark_writer(h);
                f2d_recover_after_timeout(h);
                spin = 0;
                continue;
            }
        }
        f2d_unpark_writer(h);
        spin = 0;
    }
}

static inline void f2d_rwlock_wrunlock(F2dHandle *h) {
    F2dHeader *hdr = h->hdr;
    __atomic_store_n(&hdr->rwlock, 0, __ATOMIC_RELEASE);
    if (__atomic_load_n(&hdr->rwlock_waiters, __ATOMIC_RELAXED) > 0)
        syscall(SYS_futex, &hdr->rwlock, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
}

/* ================================================================
 * Layout math + create / open / destroy
 *
 * Layout: Header -> reader_slots[1024] -> grid[(rows+1)*(cols+1) int64]
 * ================================================================ */

/* Single source of truth for the mmap region layout offsets. */
typedef struct { uint64_t reader_slots, tree; } F2dLayout;

static inline F2dLayout f2d_layout(void) {
    F2dLayout L;
    L.reader_slots = sizeof(F2dHeader);
    L.tree         = L.reader_slots + (uint64_t)F2D_READER_SLOTS * sizeof(F2dReaderSlot);
    L.tree         = (L.tree + 7) & ~(uint64_t)7;   /* 8-byte align the int64 tree array */
    return L;
}

/* the grid is 1-indexed row-major: (rows+1)*(cols+1) int64 slots, row/col 0 unused */
static inline uint64_t f2d_total_size(uint64_t rows, uint64_t cols) {
    F2dLayout L = f2d_layout();
    return L.tree + (rows + 1) * (cols + 1) * sizeof(int64_t);
}

static inline void f2d_init_header(void *base, uint64_t rows, uint64_t cols, uint64_t total) {
    F2dLayout L = f2d_layout();
    F2dHeader *hdr = (F2dHeader *)base;
    /* Zero the header + reader-slot region; the grid relies on the fresh mapping
       being OS zero-filled (every prefix sum starts at 0). */
    memset(base, 0, (size_t)L.tree);
    hdr->magic            = F2D_MAGIC;
    hdr->version          = F2D_VERSION;
    hdr->rows             = rows;
    hdr->cols             = cols;
    hdr->total_size       = total;
    hdr->reader_slots_off = L.reader_slots;
    hdr->tree_off         = L.tree;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

static inline int64_t *f2d_tree(F2dHandle *h) {
    return (int64_t *)((char *)h->base + h->tree_off);
}

/* Layer B trusted bound: number of int64 tree slots guaranteed within the real
 * mapping.  Derived from the process-local mmap_size (fixed at attach, not
 * peer-writable) and the SAME tree_off f2d_tree() uses, so a peer that corrupts
 * hdr->rows / hdr->cols / tree_off after attach-time validation can never drive
 * an access outside the mapping.  Equals (rows+1)*(cols+1) for a valid grid. */
static inline uint64_t f2d_tree_slots_max(F2dHandle *h) {
    uint64_t off = h->tree_off;
    if (off >= h->mmap_size) return 0;
    return (h->mmap_size - off) / sizeof(int64_t);
}

static inline F2dHandle *f2d_setup(void *base, size_t map_size,
                                 const char *path, int backing_fd) {
    F2dHeader *hdr = (F2dHeader *)base;
    F2dHandle *h = (F2dHandle *)calloc(1, sizeof(F2dHandle));
    if (!h) {
        munmap(base, map_size);
        if (backing_fd >= 0) close(backing_fd);
        return NULL;
    }
    h->hdr          = hdr;
    h->base         = base;
    h->reader_slots = (F2dReaderSlot *)((uint8_t *)base + hdr->reader_slots_off);
    h->tree_off     = hdr->tree_off;   /* single validated read; bound and pointer stay consistent */
    h->rows         = hdr->rows;
    h->cols         = hdr->cols;
    h->mmap_size    = map_size;
    h->path         = path ? strdup(path) : NULL;
    h->backing_fd   = backing_fd;
    h->my_slot_idx  = UINT32_MAX;
    return h;
}

/* Validate a mapped header (shared by f2d_create reopen and f2d_open_fd). */
static inline int f2d_validate_header(const F2dHeader *hdr, uint64_t file_size) {
    if (hdr->magic != F2D_MAGIC) return 0;
    if (hdr->version != F2D_VERSION) return 0;
    if (hdr->rows < F2D_MIN_DIM || hdr->rows > F2D_MAX_DIM) return 0;
    if (hdr->cols < F2D_MIN_DIM || hdr->cols > F2D_MAX_DIM) return 0;
    if (hdr->total_size != file_size) return 0;
    if (hdr->total_size != f2d_total_size(hdr->rows, hdr->cols)) return 0;
    F2dLayout L = f2d_layout();
    if (hdr->reader_slots_off != L.reader_slots) return 0;
    if (hdr->tree_off != L.tree) return 0;
    return 1;
}

/* validate the requested grid dimensions (rows, cols) */
static int f2d_validate_args(uint64_t rows, uint64_t cols, char *errbuf) {
    if (errbuf) errbuf[0] = '\0';
    if (rows < F2D_MIN_DIM || rows > F2D_MAX_DIM) { F2D_ERR("rows must be between 1 and 2^24"); return 0; }
    if (cols < F2D_MIN_DIM || cols > F2D_MAX_DIM) { F2D_ERR("cols must be between 1 and 2^24"); return 0; }
    if ((cols + 1) > (UINT64_MAX / 8) / (rows + 1)) { F2D_ERR("rows * cols too large for the grid"); return 0; }
    return 1;
}

/* Securely obtain a fd for a path-backed segment: create it exclusively
 * (O_CREAT|O_EXCL|O_NOFOLLOW at `mode`, default 0600 = owner-only), or, if it
 * already exists, attach to it (O_RDWR|O_NOFOLLOW, no O_CREAT). O_EXCL blocks a
 * pre-seeded or hard-linked file and O_NOFOLLOW a symlink swap, so a local
 * attacker can no longer redirect or poison the backing store through the path.
 * Cross-user sharing is opt-in via a wider `mode` (e.g. 0660); the caller still
 * validates the file's contents via f2d_validate_header. */
static int f2d_secure_open(const char *path, mode_t mode, char *errbuf) {
    for (int attempt = 0; attempt < 100; attempt++) {
        int fd = open(path, O_RDWR|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC, mode);
        if (fd >= 0) { (void)fchmod(fd, mode); return fd; }   /* exact mode: umask narrowed the O_EXCL create */
        if (errno != EEXIST) { F2D_ERR("create %s: %s", path, strerror(errno)); return -1; }
        fd = open(path, O_RDWR|O_NOFOLLOW|O_CLOEXEC);
        if (fd >= 0) return fd;
        if (errno == ENOENT) continue;   /* creator unlinked between our two opens; retry */
        F2D_ERR("open %s: %s", path, strerror(errno));  /* ELOOP => symlink rejected */
        return -1;
    }
    F2D_ERR("open %s: create/attach kept racing", path);
    return -1;
}

static F2dHandle *f2d_create(const char *path, uint64_t rows, uint64_t cols, mode_t mode, char *errbuf) {
    if (!f2d_validate_args(rows, cols, errbuf)) return NULL;

    uint64_t total = f2d_total_size(rows, cols);
    int anonymous = (path == NULL);
    int fd = -1;
    size_t map_size;
    void *base;

    if (anonymous) {
        map_size = (size_t)total;
        base = mmap(NULL, map_size, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
        if (base == MAP_FAILED) { F2D_ERR("mmap: %s", strerror(errno)); return NULL; }
    } else {
        fd = f2d_secure_open(path, mode, errbuf);
        if (fd < 0) return NULL;
        if (flock(fd, LOCK_EX) < 0) { F2D_ERR("flock: %s", strerror(errno)); close(fd); return NULL; }
        struct stat st;
        if (fstat(fd, &st) < 0) { F2D_ERR("fstat: %s", strerror(errno)); flock(fd, LOCK_UN); close(fd); return NULL; }
        int is_new = (st.st_size == 0);
        if (!is_new && (uint64_t)st.st_size < sizeof(F2dHeader)) {
            F2D_ERR("%s: file too small (%lld)", path, (long long)st.st_size);
            flock(fd, LOCK_UN); close(fd); return NULL;
        }
        if (is_new && ftruncate(fd, (off_t)total) < 0) {
            F2D_ERR("ftruncate: %s", strerror(errno)); flock(fd, LOCK_UN); close(fd); return NULL;
        }
        map_size = is_new ? (size_t)total : (size_t)st.st_size;
        base = mmap(NULL, map_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) { F2D_ERR("mmap: %s", strerror(errno)); flock(fd, LOCK_UN); close(fd); return NULL; }
        if (!is_new) {
            if (!f2d_validate_header((F2dHeader *)base, (uint64_t)st.st_size)) {
                F2D_ERR("invalid Fenwick2D tree file"); munmap(base, map_size); flock(fd, LOCK_UN); close(fd); return NULL;
            }
            flock(fd, LOCK_UN); close(fd);
            return f2d_setup(base, map_size, path, -1);
        }
    }
    f2d_init_header(base, rows, cols, total);
    if (fd >= 0) { flock(fd, LOCK_UN); close(fd); }
    return f2d_setup(base, map_size, path, -1);
}

static F2dHandle *f2d_create_memfd(const char *name, uint64_t rows, uint64_t cols, char *errbuf) {
    if (!f2d_validate_args(rows, cols, errbuf)) return NULL;

    uint64_t total = f2d_total_size(rows, cols);
    int fd = memfd_create(name ? name : "fenwick2d", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (fd < 0) { F2D_ERR("memfd_create: %s", strerror(errno)); return NULL; }
    if (ftruncate(fd, (off_t)total) < 0) {
        F2D_ERR("ftruncate: %s", strerror(errno)); close(fd); return NULL;
    }
    (void)fcntl(fd, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW);
    void *base = mmap(NULL, (size_t)total, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { F2D_ERR("mmap: %s", strerror(errno)); close(fd); return NULL; }
    f2d_init_header(base, rows, cols, total);
    return f2d_setup(base, (size_t)total, NULL, fd);
}

static F2dHandle *f2d_open_fd(int fd, char *errbuf) {
    if (errbuf) errbuf[0] = '\0';
    struct stat st;
    if (fstat(fd, &st) < 0) { F2D_ERR("fstat: %s", strerror(errno)); return NULL; }
    if ((uint64_t)st.st_size < sizeof(F2dHeader)) { F2D_ERR("too small"); return NULL; }
    size_t ms = (size_t)st.st_size;
    void *base = mmap(NULL, ms, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { F2D_ERR("mmap: %s", strerror(errno)); return NULL; }
    if (!f2d_validate_header((F2dHeader *)base, (uint64_t)st.st_size)) {
        F2D_ERR("invalid Fenwick2D tree table"); munmap(base, ms); return NULL;
    }
    int myfd = fcntl(fd, F_DUPFD_CLOEXEC, 0);
    if (myfd < 0) { F2D_ERR("fcntl: %s", strerror(errno)); munmap(base, ms); return NULL; }
    return f2d_setup(base, ms, NULL, myfd);
}

static void f2d_destroy(F2dHandle *h) {
    if (!h) return;
    /* Release our reader slot on clean teardown (else short-lived-reader churn
     * exhausts the slot table); skip if a lock is still held (subcount>0). */
    if (h->reader_slots && h->my_slot_idx != UINT32_MAX && h->cached_pid &&
        h->cached_fork_gen == __atomic_load_n(&f2d_fork_gen, __ATOMIC_RELAXED) &&
        __atomic_load_n(&h->reader_slots[h->my_slot_idx].subcount, __ATOMIC_ACQUIRE) == 0) {
        uint32_t expected = h->cached_pid;
        __atomic_compare_exchange_n(&h->reader_slots[h->my_slot_idx].pid,
                &expected, 0, 0, __ATOMIC_RELEASE, __ATOMIC_RELAXED);
    }
    if (h->backing_fd >= 0) close(h->backing_fd);
    if (h->base) munmap(h->base, h->mmap_size);
    free(h->path);
    free(h);
}

static inline int f2d_msync(F2dHandle *h) {
    if (!h || !h->base) return 0;
    return msync(h->base, h->mmap_size, MS_SYNC);
}

/* ================================================================
 * 2-D Fenwick tree (binary indexed tree) operations -- callers hold the lock.
 * 1-indexed row-major grid; along each axis a BIT node x covers (x-lowbit(x), x],
 * lowbit(x) = x & -x.  Values are signed int64 (deltas may be negative).
 * ================================================================ */

static inline uint64_t f2d_lowbit(uint64_t x) { return x & (~x + 1ULL); }

/* flat row-major index of grid cell (i,j) in the (rows+1)x(cols+1) array */
static inline uint64_t f2d_idx(uint64_t i, uint64_t j, uint64_t cols) {
    return i * (cols + 1) + j;
}
/* highest 1-based (row, col) safely inside the mapping.  cols is used as-is (a
 * corrupt-large cols is caught by the per-access flat bound below); rows is
 * clamped so (rows+1) full rows of stride (cols+1) fit -- Layer B. */
static inline void f2d_dims(F2dHandle *h, uint64_t *rmax, uint64_t *cmax) {
    uint64_t stride = h->cols + 1;
    uint64_t smax   = f2d_tree_slots_max(h);
    uint64_t fit1   = stride ? smax / stride : 0;    /* whole (rows+1) that fit */
    uint64_t maxr   = fit1 ? fit1 - 1 : 0;
    *rmax = h->rows < maxr ? h->rows : maxr;
    *cmax = h->cols;
}

/* add delta at grid cell (x,y) (1-based); caller holds the write lock */
static void f2d_update_locked(F2dHandle *h, uint64_t x, uint64_t y, int64_t delta) {
    int64_t *tree = f2d_tree(h);
    uint64_t rmax, cmax; f2d_dims(h, &rmax, &cmax);
    uint64_t smax = f2d_tree_slots_max(h);
    for (uint64_t i = x; i >= 1 && i <= rmax; i += f2d_lowbit(i))
        for (uint64_t j = y; j >= 1 && j <= cmax; j += f2d_lowbit(j)) {
            uint64_t f = f2d_idx(i, j, cmax);
            if (f < smax) tree[f] += delta;          /* Layer B: bulletproof flat bound */
        }
}

/* prefix sum over the rectangle [1..x] x [1..y] (1-based; x or y == 0 -> 0) */
static int64_t f2d_prefix_locked(F2dHandle *h, uint64_t x, uint64_t y) {
    const int64_t *tree = f2d_tree(h);
    uint64_t rmax, cmax; f2d_dims(h, &rmax, &cmax);
    uint64_t smax = f2d_tree_slots_max(h);
    if (x > rmax) x = rmax;
    if (y > cmax) y = cmax;
    int64_t s = 0;
    for (uint64_t i = x; i > 0; i -= f2d_lowbit(i))
        for (uint64_t j = y; j > 0; j -= f2d_lowbit(j)) {
            uint64_t f = f2d_idx(i, j, cmax);
            if (f < smax) s += tree[f];
        }
    return s;
}

/* sum over the inclusive rectangle [x1..x2] x [y1..y2] (1-based) via inclusion-exclusion */
static int64_t f2d_rect_locked(F2dHandle *h, uint64_t x1, uint64_t y1, uint64_t x2, uint64_t y2) {
    return f2d_prefix_locked(h, x2,     y2)
         - f2d_prefix_locked(h, x1 - 1, y2)
         - f2d_prefix_locked(h, x2,     y1 - 1)
         + f2d_prefix_locked(h, x1 - 1, y1 - 1);
}

/* value at cell (x,y) == the 1x1 rectangle sum */
static int64_t f2d_point_locked(F2dHandle *h, uint64_t x, uint64_t y) {
    return f2d_rect_locked(h, x, y, x, y);
}

/* reset the whole grid to 0 (caller holds the write lock) */
static inline void f2d_clear_locked(F2dHandle *h) {
    uint64_t slots = (h->rows + 1) * (h->cols + 1);
    uint64_t slots_max = f2d_tree_slots_max(h);      /* Layer B: clamp memset to the mapping */
    if (slots > slots_max) slots = slots_max;
    memset(f2d_tree(h), 0, (size_t)(slots * sizeof(int64_t)));
}

#endif /* F2D_H */
