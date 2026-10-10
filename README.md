# Collection bridge and native migration

The existing MeTTa memory interface uses Chroma by default. An optional
`memory_backend.json` beside `lib_chromadb.py` (or the path selected by
`METTACLAW_MEMORY_BACKEND_CONFIG`) enables a native CeTTa collection adapter.
Embedding inference stays with the application's configured provider.

## Local setup

Install and test the native service using the
[CeTTa guide](https://github.com/godelclaw/CeTTa/blob/feature/our-cetta-unified/docs/memory/README.md).
Then, in this checkout:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-migration.txt
mkdir -m 700 private-memory
cp examples/memory-profile.json private-memory/profile.json
# Edit the copy to describe the existing collection's actual embedding provider.
.venv/bin/python configure_memory.py --name my-memories --profile private-memory/profile.json
```

`configure_memory.py` discovers the current user's installed release and socket,
verifies release hashes, and writes a private configuration in `chroma` mode.
It refuses to overwrite an existing configuration. Use `--release`, `--socket`,
`--state-dir` and `--config` for other layouts. The example profile is deliberately
synthetic; it is not a downloadable model or an inference configuration.

An existing application can use its own Python environment; the requirements
file pins the environment used for this migration suite. Configure the existing
application's `METTACLAW_CHROMA_DIR` and `METTACLAW_EMBED_*` settings as usual.
For a configuration outside this checkout, also set
`METTACLAW_MEMORY_BACKEND_CONFIG`. The bridge's metadata checks must agree with
those runtime settings. Private configs, exports and data directories are
ignored by Git; generated test fixtures contain no personal records.

The configuration supplies `mode`, `state_dir`, `client_dir`, `socket`, `name`,
`profile`, `worker` and `module`. Use the paths from a verified CeTTa memory
service release. `profile` includes `embedding_output_dim`, embedding model and
version/protocol metadata, and `hnsw:space: "cosine"`. A legacy collection lacking
model provenance needs that limitation explicitly recorded. Different agents
use distinct collection names and state directories.
If legacy metadata uses different model/backend labels, preserve `profile` and
bind the reviewed current provider in a separate `runtime_profile` field. The
bridge checks its `embedding_*` settings on import. This records the migration
baseline; it does not prove the historical provenance of each embedding or
infer model equivalence from a filename. A later provider change requires
explicit review of compatibility and, when needed, re-embedding.

Modes:

* `chroma`: existing behavior, with the migration lock around bridge operations.
* `shadow`: Chroma answers; a bounded queue (16 jobs, one background worker)
  compares native retrieval. Changed generations and full queues are skipped.
* `native`: the CeTTa collection answers and commits writes. Chroma is mirrored
  for rollback; a mirror failure is observable and cannot turn a committed
  native write into a reported failure.

`shadow.json` records counts, generation, overlap, latency and error class;
it contains no query text, documents or IDs. `fallback.json` reports whether
the fallback mirror is current. These files describe the latest writing
process; they are not a multi-process historical metrics store. Shadow reads
run outside the foreground lock. Shadow write mirroring is synchronous and
can add latency if the native owner is slow; its errors preserve Chroma's
successful result. Missing synchronization state triggers background rebuilding
from the current Chroma embeddings. No embedding model is called to rebuild.

The native policy is in CeTTa's `lib/petta/lib_memory_collection.metta`.
Python coordinates transport, migration, locks and comparisons. It does not
implement a second filter or ranking language. The adapter supports the API
subset used here: cosine retrieval, ID/time/equality lookups, metadata patches,
episode links and deletion. Unsupported operations fail explicitly.

## Migration commands

Run `memory_migrate.py` in the application's Chroma/NumPy environment. Every
command takes `--config CONFIG --chroma DIRECTORY`; `--collection` defaults to
`memories`. The actions are:

```
export   --snapshot PRIVATE_FILE
replay   --snapshot PRIVATE_FILE --receipt RECEIPT --queries 128
shadow   --receipt RECEIPT
native   --receipt RECEIPT
rollback
status
```

Export reads the complete collection using Chroma's public API while holding
the same lock used by all bridge writers. It never copies a live SQLite file.
Raw external Chroma writers must be quiescent; the lock cannot coordinate
uncontrolled writers. Private CMEM1 exports include texts and vectors and are
created with private permissions. Keep them outside public repositories.

Replay requires `chroma` mode, so its native candidate is not serving answers.
It checks complete source equality, exact embedding bytes, an independent NumPy
cosine reference, measured Chroma overlap, updates/deletion, worker recovery and
checkpoint reopening. The seeded queries use recorded document vectors and
perturbations; they are not a relevance evaluation or a transcript of live
questions. A failed replay writes a failed receipt.

Shadow and native activation synchronize under the writer lock. Native
activation checks every ID, document, metadata value and embedding byte again
at the switch boundary. A passing receipt is pinned by hash and collection
identity. Operators must additionally qualify the deployed implementation,
restart behavior and their workload before activation. The receipt is not an
unforgeable or universal proof of safety.
Activation retains its own private receipt under the configured state directory;
the running application does not depend on an export or scratch-report path.

Rollback copies current native records into Chroma, removes obsolete records
and metadata fields, and verifies the result before changing modes. IDs,
documents and metadata must match exactly. Chroma's float32 cosine round trip
may normalize/rescale embeddings: the verifier reports rounded vectors and
requires relative L2 error at most `2e-6` against the float32 source. The
original native collection remains retained. A failed check leaves native
primary. There is no automatic fallback to a potentially stale mirror.

The native service has its own journal/checkpoint and retains embedding bytes.
Current deletion does not erase old exports, journals or application logs;
retention is a separate policy. This migration changes the memory collection,
not unrelated application databases.

## Tests

`tests/test_memory_backend.py --cetta-root REPO --server SERVER --worker WORKER`
uses a generated 257-by-16 fixture and temporary stores. It checks replay,
shadow authority, mirrored CRUD, restart rebuilding, bounded queue behavior,
foreground progress during a blocked comparison, native mode and verified
rollback. It rejects a deliberately corrupted rollback vector. An optional
`--work-dir` chooses the temporary-store parent. No downloaded corpus or model
weights are needed.
