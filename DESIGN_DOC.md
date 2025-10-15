# Nano Mooncake — A Minimal GPU-Only KV Cache over IRIS (Python/Triton)

> **Status**: Design document
> **Version**: v1 (host-allocated, device-pushed) with forward-compatibility to v2 (device-allocated)
> **Audience**: Systems & inference engineers integrating PD-disaggregated LLM inference (prefill → decode)

---

## 0. Purpose & Scope

**Nano Mooncake** is a small, fast, and pragmatic distributed KV cache tailored to LLM KV tensors. It provides **one-hop, zero-copy RDMA** GPU↔GPU transfers using **IRIS** inside **Triton** kernels. The design intentionally prioritizes *simplicity* and *clean integration* with existing Triton attention/prefill kernels.

- **v1** (this doc): **Host-allocated** address plans; **device-pushed** writes (prefill kernels call `iris.store` directly into decode cache; kernel flips READY).
- **v2** (future): **Device-allocated** addresses (producer kernel performs remote allocation via IRIS atomics); identical public API; same on-device header/state model.

We keep the CPU/SSD tiers, HA, and complex policies out of scope for v1. The end result is a minimal component you can drop into PD-disaggregated inference to share KV at line rate with very little code.

---

## 1. Goals and Non-Goals

### Goals
- **GPU-only**, VRAM-resident object cache (KV pages).
- **Zero-copy** one-hop transfers via IRIS (`iris.store`, `iris.load`, atomics).
- **Simple API**: `put`, `get`, `remove`, plus **plan-based** `open_for_write`, `commit`, `get_location`.
- **Preferred placement** (`preferred_segment`) to push prefill results *directly* into decode VRAM.
- **Immutable objects**; atomic **READY** publish; minimal yet safe consistency.
- **MVP → v1 → v2** evolution without breaking public API.

### Non-Goals (v1)
- Persistence (SSD), CPU/DRAM tiering, replication & HA.
- Multi-NIC striping (can come later with IRIS/ROCm evolution).
- Complex eviction (start with watermark + FIFO/LRU-lite).

---

## 2. Terminology

- **Rank**: A process/GPU participating in the IRIS symmetric heap (via torch.distributed world size).
- **Owner**: The rank that *stores* a given object (typically a decode rank).
- **Producer**: The rank that *writes* the object (typically a prefill rank).
- **Plan**: Host-allocated descriptor (header & payload offsets on owner) consumed by device kernels.
- **Header**: Fixed metadata struct in owner VRAM holding object state & layout.
- **Payload**: The contiguous page-first KV data buffer in owner VRAM.

---

## 3. High-Level Architecture

```
+----------------------+ control plane (small RPCs; rank0 as master)
| Master (rank 0) | <---------------------------------------------------+
| - object_index | |
| - alloc state | |
| - routing | |
+----------+-----------+ |
| open_for_write / get_location / remove |
v |
+----------+-----------+ data plane (IRIS one-hop RDMA) |
| Producer (prefill) | -- iris.store --> +----------------------------+ |
| - Triton kernels | | Owner (decode rank, VRAM) | |
| - uses WritePlan | <-- iris.load -- | - Header arena + Payload | |
+----------------------+ | - READY atomic | |
+----------------------------+
```


**Control plane** (rank 0 master) stays tiny—just metadata and allocation.
**Data plane** moves bytes directly GPU↔GPU via IRIS from inside Triton kernels (one-hop).

---

## 4. Public API (Python)

Public surface is intentionally minimal and stable across v1→v2:

```python
class NanoKVCache:
    def __init__(self, heap_bytes: int, master_rank: int = 0, page_bytes: int = 256*1024): ...

    # Simple object API (host-managed copy using IRIS kernels under the hood)
    def put(self, key: str, tensor: torch.Tensor, *, preferred_segment: int | None = None) -> None: ...
    def get(self, key: str, *, dst: torch.Tensor | None = None) -> torch.Tensor: ...
    def remove(self, key: str) -> None: ...

    # Plan-based API (preferred for prefill→decode in-kernel writes)
    def open_for_write(self, key: str, bytes_total: int, *,
                       preferred_segment: int | None = None) -> WritePlan: ...
    def commit(self, key: str) -> None: ...               # often a no-op (kernel commits READY)
    def get_location(self, key: str) -> ObjMeta: ...      # returns owner + offsets for device loads

    # Introspection / maintenance
    def stats(self) -> dict: ...
    def evict_until_below(self, usage_ratio: float) -> None: ...
```

- `preferred_segment` pin-points the **owner** rank (e.g., decode GPU).
- `put()` is sugar over `open_for_write` + a host copy kernel; **prefill should use `open_for_write` and write inside its own kernels** for best overlap.
- The above **does not change** for v2; only internals switch to device allocation when enabled.

---

## 5. Data Structures (Deep Dive)

### 5.1 On-Device Object Header (in owner VRAM)

A compact, 64B-aligned record per object. Written by host (v1) or device (v2); final **READY** flip by producer kernel.

```text
struct ObjHeader {
  u64 key_hash;        // xxhash64(key)
  u32 n_pages;         // number of pages
  u32 page_size;       // bytes per page
  u64 payload_base;    // VRAM offset of page 0 (owner)
  u64 bytes_total;     // complete logical bytes
  u32 state;           // 0=RESERVED,1=WRITING,2=READY,3=DEAD
  u32 epoch;           // version (ABA guard on reuse)
  u64 last_access;     // optional (updated by readers)
  u32 flags;           // optional (pin, priority, policy)
  u32 _pad;
}
```

**Ordering & visibility**:
- Writer fills fields → sets `state=WRITING` (release).
- After all `iris.store` page writes, writer `atomic_xchg(state, READY, release)`.
- Device readers spin-check `state==READY` with acquire before accessing payload.

### 5.2 Host-Side Metadata (authoritative index)

```python
@dataclass
class ObjMeta:
    key_hash: int
    owner_rank: int
    header_off: int
    payload_off: int
    bytes_total: int
    page_bytes: int
    n_pages: int
    epoch: int
```

- Lives in `object_index: dict[int, ObjMeta]` at the master (rank 0).
- Published when object becomes READY (from host commit in v1; from announce ring in v2).

### 5.3 WritePlan (host alloc → device consume)

```python
@dataclass
class WritePlan:
    key_hash: int
    dst_rank: int
    header_off: int
    payload_base: int
    page_bytes: int
    n_pages: int
    bytes_total: int
    epoch: int
```

- `open_for_write()` returns this to the producer. Prefill kernels take these fields as args and directly `iris.store` pages, then flip READY.
- **Stable** across v1→v2.

### 5.4 Owner-Side Heap Control (for v2 device-alloc)

A small control block and arenas in owner VRAM to enable **device-side** remote allocation later, without API changes:

```text
struct HeapCtl {
  // Payload arena
  u64 heap_base;
  u64 heap_limit;
  u64 bump_ptr;      // atomic bytes bump (payload)

  // Header arena
  u64 hdr_base;
  u32 hdr_count;
  u32 hdr_head;      // atomic index allocator

  // Optional freelists (lock-free) per size class
  u64 free_head[8];

  // Device→host announce ring
  u64 ann_base;
  u32 ann_capacity;
  u32 ann_head;      // device producer
  u32 ann_tail;      // host consumer (host writes)

  // Watermarks & version
  u32 epoch;
  u32 water_hi;      // %, e.g., 95
  u32 water_lo;      // %, e.g., 85
  u32 _pad;
};

struct Announce {
  u64 key_hash;
  u32 owner_rank;
  u32 epoch;
  u64 header_off;
  u64 payload_off;
  u32 n_pages;
  u32 page_size;
  u64 bytes_total;
};
```

- **v1**: `HeapCtl` may exist but is not required; host does allocation.
- **v2**: Producer kernel uses remote atomics on `bump_ptr`/`hdr_head`, then publishes `Announce` to host poller which fills/updates `object_index`.

---

## 6. Control Plane (Rank 0 Master)

Minimal service using `torch.distributed` point-to-point messages:

- **Tables**:
  - `object_index: dict[key_hash, ObjMeta]`
  - `bump_ptr_by_rank: dict[rank, int]`
  - `heap_bytes_by_rank: dict[rank, int]`
  - `free_list_by_rank: dict[rank, list[(off,len)]]` (optional in v1)
- **Routing**:
  - `preferred_segment` → owner = requested rank
  - else `owner = hash(key) % world_size` (or “stick to requester”)

**API endpoints**:
- `OPEN_FOR_WRITE(key, bytes_total, preferred_segment) -> WritePlan`
- `COMMIT(key)` (no-op if kernel already set READY; may finalize index)
- `GET_LOCATION(key) -> ObjMeta` (block or error if not READY)
- `REMOVE(key)` (CAS header READY→DEAD, free space, drop index)

**Eviction**:
- Watermarks per owner rank (high/low). If `open_for_write` would exceed high, evict until below low:
  - Pick victims FIFO or by `last_access`.
  - Ensure safety: CAS READY→DEAD before reclaim.
  - Reclaim payload and header (v1: may skip and rely on bump until reboot; v2: push to freelist).

---

## 7. Data Plane (Triton + IRIS)

### 7.1 Producer (prefill) kernel snippets

- **Direct page write** (tile in registers/shared; no local global read):

```python
@triton.jit
def kv_store_page_values(values, dst_base, page_idx, page_bytes,
                         src_rank: tl.constexpr, dst_rank: tl.constexpr, heap_bases_ptr):
    iris.store(dst_base + page_idx * page_bytes,
               values, src_rank, dst_rank, heap_bases_ptr)
```

- **From local global (streaming) to remote** (use `.cg` on source load to avoid L1 pollution):

```python
@triton.jit
def kv_store_page_from_src(src_ptr, n_elems, dst_base, page_idx, page_bytes,
                           src_rank: tl.constexpr, dst_rank: tl.constexpr,
                           heap_bases_ptr, use_cg: tl.constexpr):
    offs = tl.arange(0, n_elems)
    if use_cg == 1:
        vals = tl.load(src_ptr + offs, mask=offs<n_elems, cache_modifier=".cg")
    else:
        vals = tl.load(src_ptr + offs, mask=offs<n_elems)
    iris.store(dst_base + page_idx * page_bytes,
               vals, src_rank, dst_rank, heap_bases_ptr, mask=offs<n_elems)
```

- **Commit READY** (device-side):

```python
@triton.jit
def kv_commit_ready(header_ptr, src_rank: tl.constexpr, dst_rank: tl.constexpr, heap_bases_ptr):
    # 2 = READY; release ensures earlier stores are globally visible
    iris.atomic_xchg(header_ptr + STATE_OFF, 2, src_rank, dst_rank, heap_bases_ptr,
                     sem="release", scope="sys")
```

**Prefill kernel** calls the above directly once passed the `WritePlan` fields.

### 7.2 Consumer (decode) read kernel

```python
@triton.jit
def kv_read_remote(dst_local_ptr, src_payload_base, n_bytes,
                   my_rank: tl.constexpr, src_rank: tl.constexpr, heap_bases_ptr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    m = i < n_bytes
    vals = iris.load(src_payload_base + i, my_rank, src_rank, heap_bases_ptr, mask=m)
    tl.store(dst_local_ptr + i, vals, mask=m)
```

- Decode can also directly operate on resident cache if layout matches; otherwise pulls into a workspace.

---

## 8. Operational Flows

### 8.1 MVP (v0): host-managed put/get (works today)
1. `open_for_write(key, bytes, preferred_segment)` → `WritePlan`
2. Host launches a copy kernel (`kv_store_page_from_src`) to write `tensor` into owner payload.
3. Tiny kernel `kv_commit_ready` or producer kernel flip READY.
4. Master publishes `ObjMeta` in `object_index`.
5. `get(key)` on any rank uses `GET_LOCATION` and launches `kv_read_remote` if remote.

**Pros**: trivial to implement; already hits one-hop, zero-copy data path.
**Cons**: plan RPC adds ~0.1–0.3 ms overhead (size/QPS dependent).

### 8.2 v1: plan-based in-kernel writes (recommended path)
- Identical to MVP except **prefill** calls write/commit **inside** its Triton kernels using the `WritePlan`.
- Cleaner overlap (compute tile → remote store next tile).

### 8.3 v2 (future): device-allocated hot path (no host alloc)
1. Producer kernel does remote **bump** on owner `HeapCtl.bump_ptr` to reserve payload.
2. Producer kernel allocates a header slot (remote `hdr_head`), writes header, sets `WRITING`.
3. Producer kernel stores pages, flips `READY`.
4. Producer kernel writes **Announce** entry; host poller updates `object_index`.

**API unchanged**. `open_for_write` can become a fast-path no-op (or return immediately if device-alloc is enabled), but we keep it for symmetry and to carry the `key_hash/epoch`.

---

## 9. Consistency & Concurrency

- **Immutable values** after `put` completes; updates are `remove` + `put`.
- **Atomic publish**: writer sets `READY` with `release`; device readers should acquire before reading.
- **Master serializes** conflicting ops per key (e.g., no duplicate key unless removed first).
- **Timeout reclaim**: headers stuck in `RESERVED/WRITING` past T are reclaimed (CAS to DEAD; free).

---

## 10. Allocation & Eviction

### v1 Allocation
- **Bump pointer per owner rank** (bytes). Optional free list for reclaimed holes.
- Alignment: header 64B; payload page 4–16 KB aligned (default page 256 KB).

### Eviction
- Trigger when `used/total > high` (e.g., 95%); evict until `< low` (e.g., 85%).
- Policy: FIFO or timestamp (`last_access` updated best-effort by readers).
- Safety: CAS READY→DEAD before reuse; skip `RESERVED/WRITING`.

### v2 Allocation
- Device bump (remote atomic add) + header index (remote atomic add).
- Optional: per-size-class freelists (lock-free stacks) for reuse; compaction left for host tooling.
- **Announce ring** to host keeps `object_index` coherent without blocking device hot path.

---

## 11. Benchmarking & Telemetry

### What to measure
- **Control-plane latency**: `t_plan` for `open_for_write` (RPC + alloc) in µs.
- **Copy throughput**: `t_copy_per_MB` via IRIS (`iris.store`/`iris.load`) at various sizes.
- **End-to-end** `put` latency p50/p99 for common KV sizes (e.g., 0.5–8 MB).
- **Concurrency scaling**: multiple producers writing to multiple owners; look at tail lat.
- **Overlap**: compute vs IO (Nsight Systems markers around store loops).

### Rule of thumb
- If typical objects are **≤ 3–5 MB** or writers are many, **v2 device-alloc** can reduce p50/p99 by removing `t_plan` and smoothing overlap.
- For **≫ 10–20 MB** objects, `t_plan` is dominated by data transfer; v1 is often sufficient.

### Counters to expose
- Per-rank: heap_used/heap_total, evictions, READY count, ann_ring depth (v2).
- Copy stats: bytes_tx/rx, avg page time, failures/timeouts.
- Errors: CAS failures, OOM, reclaim events.

---

## 12. Migration Plan (No Breaking Changes)

1. **MVP (v0)**: `open_for_write` + host copy kernel; device `kv_commit_ready`; `get`.
2. **v1**: Encourage plan-based **in-kernel** writes in prefill; host copy path remains for testing.
3. **v2**: Enable **device-alloc** behind a feature flag:
   - Initialize `HeapCtl`, header arena, announce ring at owner ranks.
   - Add host poller thread to ingest `Announce` and fill `object_index`.
   - `open_for_write` may become a lightweight “key registration” (no allocation) or passthrough.

**API remains**: `put`, `get`, `remove`, `open_for_write`, `commit`, `get_location`.

---

## 13. Practical Defaults & Tips

- **Page size**: start at **256 KB** (benchmark 128–512 KB).
- **Source loads**: use `cache_modifier=".cg"` to keep prefill L1 clean when reading from local global.
- **Layouts**: prefer **page-first** (or page-first-direct) to write/load contiguous blocks.
- **Routing**: `preferred_segment` for PD-disaggregation; else `hash(key) % world_size`.
- **Safety**: ensure READY uses `sem="release", scope="sys"`; device readers use acquire sequencing.
- **Recovery**: on restart, host scans headers to rebuild index; reclaim non-READY after timeout.

---

## 14. Example (Plan-Based Prefill → Decode)

```python
# Host-side: create cache and allocate a write plan
cache = NanoKVCache(heap_bytes=2<<30, master_rank=0, page_bytes=256*1024)
plan = cache.open_for_write(key="sess:42:kv:L0", bytes_total=kv_bytes, preferred_segment=decode_rank)

# Producer kernel args
plan_hdr_ptr   = cache.device_ptr(plan.dst_rank, plan.header_off)
plan_payload   = cache.device_ptr(plan.dst_rank, plan.payload_base)
heap_bases_ptr = cache.heap_bases_tensor  # iris.get_heap_bases()

# Prefill Triton kernel pseudo-call
prefill_kernel[grid](
    ... # compute inputs
    plan_hdr_ptr, plan_payload, plan.page_bytes, plan.n_pages,
    src_rank=cache.rank, dst_rank=plan.dst_rank,
    heap_bases_ptr=heap_bases_ptr,
)

# Kernel internally:
#   - compute tiles -> iris.store(plan_payload + p*page_bytes, tile, ...)
#   - kv_commit_ready(plan_hdr_ptr, ...)

# Decode side
meta = cache.get_location("sess:42:kv:L0")    # ObjMeta
local = cache.get("sess:42:kv:L0")            # pulls via iris.load if remote
```

---

## 15. Future Extensions

- **Replication**: multi-owner `WritePlan` fan-out writes for hotspots (producer writes 2–3 copies).
- **Per-object policy**: header flags for pin/priority; owner evictor honors flags.
- **Backpressure**: device-side check of watermarks, fall back to host-alloc or retry.
- **Multi-NIC**: IRIS evolution may allow path pinning/striping; batch pages accordingly.
- **Security**: if multi-tenant, add capability tokens and bounds checks (out of v1 scope).

---

## 16. Summary

- v1 delivers the **fast path today**: host alloc, device writes, atomic READY.
- The public API is **clean and stable**, designed so v2 can **flip to device allocation** without breaking users.
- The key to performance is **keeping the hot path in device**: produce KV → `iris.store` to decode → READY.
- Device-alloc is a **latency/concurrency optimization**; add it where it measurably reduces tail latency for your workloads.
