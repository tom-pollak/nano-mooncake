# Nano Mooncake — Technical Design (IRIS + Triton, GPU-only KV Cache)

**Scope:** Minimal, performant, and safe distributed KV cache for LLM KV tensors, GPU-resident, single-hop GPU↔GPU transfers via IRIS from inside Triton kernels. No CPU/SSD tiers, no replication/HA, no device-side allocation. Simple control plane, fast data plane.

---

## 1. Overview

Nano Mooncake exposes object-level `put/get/remove` and a plan-based API that lets prefill Triton kernels **push** KV pages directly into **decode-owned** VRAM using IRIS. Objects are immutable after publish. A tiny master performs allocation and routing; data never flows through the master.

**Key properties**

* **GPU-only:** Data stored in owner GPU VRAM.
* **One-hop, zero-copy:** IRIS `store/load` between GPUs inside Triton kernels.
* **Atomic publish:** `READY` flip with release semantics; readers use acquire.
* **Safe eviction:** header state + reader refcount; FIFO policy with watermarks.
* **Rank-correct:** Explicit mapping to IRIS ranks.

---

## 2. Architecture

```
+--------------------------+   control plane (rank 0 master)
| Master                   |  tables: object_index, usage_by_rank
| - alloc & routing        |  RPCs: open_for_write, get_location, remove
+-------------+------------+
              |
              v
+-------------+------------+   data plane (IRIS one-hop GPU↔GPU)
| Producer (prefill GPU)   | -- iris.store -->  +-------------------------+
| - Triton kernels         |                    | Owner (decode GPU)      |
| - uses WritePlan         | <-- iris.load  --  | - header + payload      |
+--------------------------+                    | - READY atomic          |
                                               +-------------------------+
```

* **Routing:** `preferred_segment` → owner rank; otherwise `owner = xxh64(key) % world_size`.
* **Master:** rank 0 process using torch.distributed point-to-point RPC.

---

## 3. Public API (Python)

```python
class NanoKVCache:
    def __init__(self, heap_bytes: int, master_rank: int = 0,
                 page_bytes: int = 256*1024): ...

    # Object API
    def put(self, key: str, tensor: torch.Tensor, *,
            preferred_segment: int | None = None) -> None: ...

    def get(self, key: str, *, dst: torch.Tensor | None = None) -> torch.Tensor: ...
    def get_into(self, key: str, dst: torch.Tensor) -> None: ...
    def remove(self, key: str) -> None: ...
    def touch(self, key: str) -> None: ...  # best-effort last_access bump

    # Plan-based (recommended for prefill→decode)
    def open_for_write(self, key: str, bytes_total: int, *,
                       preferred_segment: int | None = None) -> WritePlan: ...

    def commit(self, key: str) -> None: ...  # idempotent; device usually flips READY
    def get_location(self, key: str) -> ObjMeta | NotReady: ...  # non-blocking

    # Introspection / maintenance
    def stats(self) -> dict: ...
    def evict_until_below(self, usage_ratio: float) -> None: ...
```

**Notes**

* `put()` is sugar over `open_for_write` + host-side copy kernel; hot path is plan-based in-kernel writes.
* `get_location()` returns `NotReady` rather than blocking; callers may poll/overlap.

---

## 4. Data Structures

### 4.1 Header Layout (Owner GPU VRAM)

**128 bytes total, 64B-aligned**

```
+--------------------+--------------------+--------------------+--------------------+
| key_hash (u64)     | key_fp_hi (u64)    | key_len (u32)      | header_version(u32)|
+--------------------+--------------------+--------------------+--------------------+
| payload_base (u64) | bytes_total (u64)  | page_size (u32)    | n_pages (u32)      |
+--------------------+--------------------+--------------------+--------------------+
| last_page_bytes(u32)| epoch (u32)       | state (u32)        | reader_refcnt(u32) |
+--------------------+--------------------+--------------------+--------------------+
| state_ts (u64)     | last_access (u64)  | flags (u32)        | _pad (u32)         |
+--------------------+--------------------+--------------------+--------------------+
```

* **state:** 0=RESERVED, 1=WRITING, 2=READY, 3=DEAD
* **identity:** master keeps full key; header carries xxh64 + optional xxh128 high for collision-guard.
* **tail:** `last_page_bytes` ensures readers clamp the final page copy.

### 4.2 Host Metadata

```python
@dataclass
class ObjMeta:
    key_hash: int
    owner_rank: int        # IRIS rank id
    header_ptr: int        # device pointer (owner)
    payload_ptr: int       # device pointer (owner)
    bytes_total: int
    page_bytes: int
    n_pages: int
    last_page_bytes: int
    epoch: int
```

### 4.3 WritePlan (to Producer)

```python
@dataclass
class WritePlan:
    key_hash: int
    dst_rank: int          # IRIS rank id (owner)
    header_ptr: int        # device pointer on owner
    payload_ptr: int       # device pointer on owner
    page_bytes: int
    n_pages: int
    last_page_bytes: int
    bytes_total: int
    epoch: int
```

---

## 5. Memory Ordering & Concurrency

* **Publish:** Producer flips `state=READY` with `iris.atomic_xchg(..., sem="release", scope="sys")` **after** all page stores.
* **Readers:** Spin until `state==READY` via `iris.atomic_cas(..., sem="acquire", scope="sys")`.
* **Refcount:** Reader increments `reader_refcnt` (`+1` acquire) before first read and decrements (`-1` release) after last read. Eviction requires `reader_refcnt==0`.
* **Immutability:** Values are immutable post-publish; updates are `remove` + `put`.

---

## 6. Control Plane (Master)

**Tables**

* `object_index: dict[key_hash, ObjMeta]` (master keeps full key for identity)
* `usage_by_rank: {rank: bytes_used}`; `heap_bytes_by_rank`
* Optional `free_list_by_rank: list[(ptr, len)]`

**Routing**

* `owner = preferred_segment if provided else xxh64(key) % world_size`

**RPCs**

* `OPEN_FOR_WRITE(key, bytes_total, preferred_segment) -> WritePlan`

  1. Compute `n_pages`, `last_page_bytes`.
  2. If owner usage > high watermark, run evictor until below low; on failure return `Busy`.
  3. Reserve header + payload, initialize header to `WRITING` (release), fill fields.
  4. Return `WritePlan` (device pointers + sizes + epoch).
* `GET_LOCATION(key) -> ObjMeta | NotReady`
* `REMOVE(key)` → CAS `READY→DEAD` if `reader_refcnt==0`, reclaim, drop index.
* **Watchdog:** reclaim `RESERVED/WRITING` stuck past timeout using `state_ts`.

**Rank mapping**

* On init, assert `iris.get_num_ranks() == dist.get_world_size()`; build a rank map if needed; store **IRIS ranks** in metadata/plans.

---

## 7. Data Plane (Triton + IRIS)

### 7.1 Producer page push

```python
@triton.jit
def kv_store_page_from_src(src_ptr, n_elems, payload_ptr, page_idx, page_bytes,
                           src_rank: tl.constexpr, dst_rank: tl.constexpr,
                           heap_bases_ptr: tl.tensor, use_cg: tl.constexpr):
    offs = tl.arange(0, n_elems)
    mask = offs < n_elems
    vals = tl.load(src_ptr + offs, mask=mask, cache_modifier=(".cg" if use_cg == 1 else None))
    iris.store(payload_ptr + page_idx * page_bytes + offs, vals,
               src_rank, dst_rank, heap_bases_ptr, mask=mask)
```

**Publish**

```python
@triton.jit
def kv_commit_ready(header_state_ptr, src_rank: tl.constexpr, dst_rank: tl.constexpr, heap_bases_ptr):
    iris.atomic_xchg(header_state_ptr, 2, src_rank, dst_rank, heap_bases_ptr,
                     sem="release", scope="sys")  # 2 = READY
```

### 7.2 Reader fetch (remote)

```python
@triton.jit
def kv_read_remote(dst_ptr, payload_ptr, n_bytes,
                   my_rank: tl.constexpr, src_rank: tl.constexpr, heap_bases_ptr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    m = i < n_bytes
    vals = iris.load(payload_ptr + i, my_rank, src_rank, heap_bases_ptr, mask=m)
    tl.store(dst_ptr + i, vals, mask=m)
```

**Reader enter/exit**

```python
# Wait READY (acquire)
while iris.atomic_cas(header_state_ptr, 2, 2, my_rank, owner_rank, heap_bases_ptr,
                      sem="acquire", scope="sys") != 2:
    pass
# Refcount
iris.atomic_add(reader_refcnt_ptr, 1, my_rank, owner_rank, heap_bases_ptr, sem="acquire", scope="sys")
# ... read pages; clamp final page to last_page_bytes ...
iris.atomic_add(reader_refcnt_ptr, -1, my_rank, owner_rank, heap_bases_ptr, sem="release", scope="sys")
```

---

## 8. Allocation & Eviction

**Allocation**

* Per-owner **bump pointer** for payload; optional simple free list for reclaimed ranges.
* Alignment: header 64B; payload pages 128–512 KB (default **256 KB**).

**Eviction policy**

* High/low watermarks (e.g., 95% / 85%). Evict oldest READY objects (FIFO) until below low.
* Safety: skip if `reader_refcnt>0` or `state!=READY`. Evictor attempts `CAS READY→DEAD`.

**Minimal FIFO evictor (host pseudocode)**

```python
def evict_until_below(owner_rank: int, low_ratio: float):
    used = usage_by_rank[owner_rank]
    total = heap_bytes_by_rank[owner_rank]
    target = int(low_ratio * total)
    spins = 0
    while used > target and spins < 4 * len(fifo_queue[owner_rank]):
        key = fifo_queue[owner_rank].popleft()
        meta = object_index.get(key)
        if not meta:
            continue
        hdr = meta.header_ptr
        rr = iris_read_u32(hdr + OFF_REFCNT)
        st = iris_read_u32(hdr + OFF_STATE)
        if st != READY or rr != 0:
            fifo_queue[owner_rank].append(key); spins += 1; continue
        ok = iris_cas_u32(hdr + OFF_STATE, expected=READY, new=DEAD, sem='acquire')
        if not ok:
            fifo_queue[owner_rank].append(key); spins += 1; continue
        free_range(owner_rank, meta.payload_ptr, meta.n_pages * meta.page_bytes)
        free_header(owner_rank, meta.header_ptr)
        used -= meta.n_pages * meta.page_bytes
        usage_by_rank[owner_rank] = used
        del object_index[key]
```

---

## 9. Telemetry & Testing

**Stats** (integers only)

* Per-rank: `bytes_used`, `bytes_total`, `ready_count`, `evict_count`, `oom_count`.
* Per-op moving averages: `t_plan_us`, `bytes_tx`, `bytes_rx`.
* Errors: `stuck_put_reclaimed`, `evict_in_use_retry`.

**Tests**

1. Correctness over 64 KB–8 MB KV sizes; optional debug CRC per page.
2. Race safety: concurrent `get/evict`; verify refcount prevents tear.
3. Recovery: simulate crash mid-put; watchdog reclaims WRITING.
4. Throughput: sweep page size (128/256/512 KB), tile/batch sizes; report GB/s and p50/p99 put latency.

---

## 10. Practical Defaults

* Page size: **256 KB** (benchmark 128–512 KB).
* Use `cache_modifier=".cg"` on producer source loads.
* Always pass **IRIS ranks** to kernels (verify rank map at init).
* Clamp reads/writes on tail page using `last_page_bytes`.
* READY publish: `release + scope=sys`; reader enter: `acquire + scope=sys`.
