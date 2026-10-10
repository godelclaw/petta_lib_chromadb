"""Opt-in CeTTa collection migration with bounded shadow comparisons.

Chroma remains primary in shadow mode. Native mode requires an explicit
operator qualification receipt. A synchronized file lock coordinates all
bridge calls with exports and mode changes; raw external DB writers must be
quiescent during those operations. No personal query/document is logged.
"""
import contextlib
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import sys
import tempfile
import threading
import time


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix=".memory-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
        fd = os.open(path.parent, os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        if os.path.exists(name): os.unlink(name)


class MemoryBackend:
    def __init__(self, chroma, config_path):
        self.chroma = chroma
        self.path = Path(config_path)
        self.config = json.loads(self.path.read_text())
        self.state = Path(self.config["state_dir"])
        self.state.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.local_lock = threading.RLock()
        self.work = queue.Queue(maxsize=16)
        self.native = None
        self.native_epoch = None
        self.stats = {"pid":os.getpid(), "queued":0,"compared":0,"skipped":0,"failures":0,"generation":None}
        self.thread = None
        sys.path.insert(0, self.config["client_dir"])
        from memory_collection import NativeCollection
        self.native_type = NativeCollection

    def __getattr__(self, name):
        return getattr(self.chroma, name)

    @contextlib.contextmanager
    def guard(self):
        with self.local_lock:
            with (self.state / "bridge.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                try: yield
                finally: fcntl.flock(lock, fcntl.LOCK_UN)

    def _mode(self):
        config = json.loads(self.path.read_text())
        mode = config["mode"]
        if mode not in ("chroma", "shadow", "native"):
            raise ValueError("invalid memory mode")
        if mode == "native":
            receipt = Path(config["qualification"])
            if hashlib.sha256(receipt.read_bytes()).hexdigest() != config["qualification_sha256"]:
                raise ValueError("native qualification receipt changed")
            evidence = json.loads(receipt.read_text())
            if evidence.get("status") != "passed" or evidence.get("collection") != self.config["name"]:
                raise ValueError("native qualification is not for this collection")
        return mode

    @contextlib.contextmanager
    def migration_guard(self):
        with (self.state / "migration.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try: yield
            finally: fcntl.flock(lock, fcntl.LOCK_UN)

    def _epoch(self):
        try: return int((self.state / "generation").read_text())
        except FileNotFoundError: return 0

    def _advance(self):
        value = self._epoch()+1
        atomic_json(self.state / "generation", value)
        return value

    def _native(self):
        if self.native is None:
            self.native = self.native_type(self.config["socket"], self.config["name"],
                self.config["profile"], self.config["worker"], self.config["module"])
        return self.native

    def export(self):
        with self.guard():
            source=self._native() if self._mode()=="native" else self.chroma
            return self._epoch(), source.get(include=["embeddings","documents","metadatas"])

    def _reconcile(self, source, native):
        old = set(native.get(include=[])["ids"])
        for offset in range(0,len(source["ids"]),16):
            end=offset+16
            native.upsert(source["ids"][offset:end], source["embeddings"][offset:end],
                          source["documents"][offset:end], source["metadatas"][offset:end])
        obsolete=sorted(old-set(source["ids"]))
        for offset in range(0,len(obsolete),16): native.delete(ids=obsolete[offset:offset+16])

    def _sync(self):
        # Called under the bridge guard. Explicitly retains existing embeddings;
        # it never invokes an embedding model. The original database is intact.
        source = self.chroma.get(include=["embeddings", "documents", "metadatas"])
        native = self._native()
        self._reconcile(source,native)
        self.native_epoch=self._epoch()
        atomic_json(self.state / "synchronized.json", {"generation":self.native_epoch,
            "revision":native.space.revision,"count":native.count(),"time":time.time()})

    def synchronize(self):
        with self.migration_guard(), self.guard():
            if self._mode()=="native":raise ValueError("native is authoritative; use rollback")
            self._sync()

    def _enqueue(self, job):
        if self.thread is None:
            self.thread=threading.Thread(target=self._background,name="memory-shadow",daemon=True)
            self.thread.start()
        try:
            self.work.put_nowait(job);self.stats["queued"]+=1
        except queue.Full: self.stats["skipped"]+=1

    def _background(self):
        native=None
        while True:
            epoch,arguments,primary = self.work.get()
            try:
                # A separate disposable worker performs comparisons outside the
                # foreground lock. Slow/failing queries cannot block Chroma reads.
                with self.guard():
                    if self._mode() != "shadow" or epoch != self._epoch():
                        self.stats["skipped"]+=1;continue
                    synchronized=self.state / "synchronized.json"
                    synced=json.loads(synchronized.read_text())["generation"] if synchronized.exists() else None
                if native is None:
                    native=self.native_type(self.config["socket"],self.config["name"],
                        self.config["profile"],self.config["worker"],self.config["module"])
                if synced != epoch:
                    with self.migration_guard():
                        with self.guard():
                            if self._mode()!="shadow" or epoch!=self._epoch():
                                self.stats["skipped"]+=1;continue
                            source=self.chroma.get(include=["embeddings","documents","metadatas"])
                            atomic_json(synchronized,{"generation":None,"rebuilding":True})
                        self._reconcile(source,native)
                        with self.guard():
                            if epoch!=self._epoch() or self._mode()!="shadow":
                                self.stats["skipped"]+=1;continue
                            atomic_json(synchronized,{"generation":epoch,"revision":native.space.revision,
                                "count":native.count(),"time":time.time()})
                            self.stats["rebuilds"]=self.stats.get("rebuilds",0)+1
                start=time.perf_counter()
                actual=native.query(**arguments)
                elapsed=(time.perf_counter()-start)*1000
                with self.guard():
                    if epoch != self._epoch() or self._mode()!="shadow":
                        self.stats["skipped"]+=1;continue
                    overlaps=[len(set(first)&set(second))/max(1,len(first))
                              for first,second in zip(primary["ids"],actual["ids"])]
                    self.stats.update(compared=self.stats["compared"]+1,generation=epoch,
                        last_overlap=overlaps,last_native_ms=elapsed,last_success=time.time(),needs_sync=False)
            except Exception as exc:
                self.stats.update(failures=self.stats["failures"]+1,last_error=type(exc).__name__,last_failure=time.time())
                if native:
                    try:native.close()
                    except Exception as cleanup:self.stats["cleanup_error"]=type(cleanup).__name__
                    native=None
            finally:
                try:atomic_json(self.state / "shadow.json",self.stats)
                finally:self.work.task_done()

    def query(self, **kwargs):
        with self.guard():
            mode=self._mode()
            if mode=="native":return self._native().query(**kwargs)
            result=self.chroma.query(**kwargs)
            if mode=="shadow": self._enqueue((self._epoch(),copy.deepcopy(kwargs),copy.deepcopy(result)))
            return result

    def get(self, **kwargs):
        with self.guard():
            return (self._native() if self._mode()=="native" else self.chroma).get(**kwargs)

    def count(self):
        with self.guard():return (self._native() if self._mode()=="native" else self.chroma).count()

    def _mutate(self, operation, kwargs):
        with self.guard():
            mode=self._mode()
            previous_epoch=self._epoch()
            if mode=="native":
                result=getattr(self._native(),operation)(**kwargs)
                epoch=self._advance()
                # Native is authoritative after cutover. A failed fallback
                # mirror is recorded, never treated as native write failure.
                try:
                    path=self.state / "fallback.json"
                    old=json.loads(path.read_text()) if path.exists() else {}
                    was_current=old.get("current",False) and old.get("generation")==previous_epoch
                    getattr(self.chroma,operation)(**kwargs)
                    atomic_json(path,{"generation":epoch,"current":was_current,"time":time.time()})
                except Exception as exc:
                    atomic_json(self.state / "fallback.json",{"generation":epoch,"current":False,
                        "error":type(exc).__name__,"time":time.time()})
                return result
            result=getattr(self.chroma,operation)(**kwargs)
            epoch=self._advance()
            if mode=="shadow":
                synchronized=self.state / "synchronized.json"
                synced=json.loads(synchronized.read_text())["generation"] if synchronized.exists() else None
                if synced != previous_epoch:
                    self.stats.update(needs_sync=True)
                    atomic_json(self.state / "shadow.json",self.stats)
                    return result
                # Mutations are mirrored before the guard is released, so a
                # subsequent queued query has a single captured generation.
                try:
                    getattr(self._native(),operation)(**kwargs)
                    self.native_epoch=epoch
                    atomic_json(self.state / "synchronized.json",{"generation":epoch,
                        "revision":self.native.space.revision,"count":self.native.count(),"time":time.time()})
                except Exception as exc:
                    self.native_epoch=None
                    self.stats.update(failures=self.stats["failures"]+1,last_error=type(exc).__name__,needs_sync=True)
                    atomic_json(self.state / "shadow.json",self.stats)
            return result

    def add(self,**kwargs):return self._mutate("add",kwargs)
    def upsert(self,**kwargs):return self._mutate("upsert",kwargs)
    def update(self,**kwargs):return self._mutate("update",kwargs)
    def delete(self,**kwargs):return self._mutate("delete",kwargs)


def configured(collection, runtime_profile=None):
    path=os.environ.get("METTACLAW_MEMORY_BACKEND_CONFIG")
    if not path:
        candidate=Path(__file__).with_name("memory_backend.json")
        if not candidate.exists():return collection
        path=str(candidate)
    bridge=MemoryBackend(collection,path)
    if runtime_profile is not None:
        # Legacy source metadata may use a symbolic model name or describe an
        # embedding daemon rather than its current transport configuration.
        # Preserve that provenance; bind the reviewed runtime separately.
        expected=bridge.config.get("runtime_profile",bridge.config["profile"])
        for key,value in runtime_profile.items():
            if key.startswith("embedding_") and expected.get(key)!=value:
                raise ValueError("runtime embedding profile differs from native collection: " + key)
    return bridge
