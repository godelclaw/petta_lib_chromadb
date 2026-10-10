#!/usr/bin/env python3
"""Private export/replay and explicit native-memory activation.

Run with the same Chroma environment as the application. Exports contain
personal records and embeddings: keep them private. The bridge lock must be
shared by every writer; raw external writers must be quiescent.
"""
import argparse
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time

from memory_backend import MemoryBackend, atomic_json


def verify_rollback(source, copied):
    """Check exact records and bounded Chroma float32 round-trip error.

    Chroma's cosine index can normalize and rescale returned embeddings even
    when re-upserting its own get() result. The native source remains retained.
    """
    import numpy as np
    def keyed(data):
        return {i:(d,m,np.asarray(v,dtype=np.float32)) for i,d,m,v in
                zip(data['ids'],data['documents'],data['metadatas'],data['embeddings'])}
    first,second=keyed(source),keyed(copied)
    if first.keys()!=second.keys():raise ValueError('rollback identity mismatch')
    rounded=0;maximum=0.
    for item_id,(document,metadata,expected) in first.items():
        actual_document,actual_metadata,actual=second[item_id]
        if document!=actual_document or metadata!=actual_metadata:
            raise ValueError('rollback record mismatch')
        if expected.shape!=actual.shape or not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise ValueError('rollback vector representation mismatch')
        error=float(np.linalg.norm(actual.astype(np.float64)-expected)/np.linalg.norm(expected.astype(np.float64)))
        if not math.isfinite(error) or error>2e-6:
            raise ValueError('rollback vector error exceeds float32 round-trip bound')
        maximum=max(maximum,error);rounded+=not np.array_equal(expected,actual)
    return {'records':len(first),'rounded_vectors':rounded,'max_relative_l2_error':maximum,'limit':2e-6}


def snapshot(bridge, destination):
    from space_service_client import encode, Tensor
    from memory_collection import metadata_atom
    epoch, rows = bridge.export()
    order=sorted(range(len(rows['ids'])),key=lambda i:rows['ids'][i])
    with destination.open('xb') as stream:
        os.chmod(destination,0o600)
        header=json.dumps({'generation':epoch,'profile':bridge.config['profile'],'count':len(order)},sort_keys=True).encode()
        stream.write(b'CMEM1'+struct.pack('<I',len(header))+header)
        for i in order:
            packet=encode((rows['ids'][i],Tensor.vector(rows['embeddings'][i]),rows['documents'][i],metadata_atom(rows['metadatas'][i])))
            stream.write(struct.pack('<I',len(packet))+packet)
        stream.flush();os.fsync(stream.fileno())
    return {'generation':epoch,'records':len(order),'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()}


def read_snapshot(path):
    from space_service_client import decode, Tensor, MAX_BYTES
    with path.open('rb') as stream:
        if stream.read(5)!=b'CMEM1':raise ValueError('snapshot format')
        size=struct.unpack('<I',stream.read(4))[0]
        if size>MAX_BYTES:raise ValueError('snapshot header size')
        header=json.loads(stream.read(size))
        rows=[]
        for _ in range(header['count']):
            size=struct.unpack('<I',stream.read(4))[0]
            if size>MAX_BYTES:raise ValueError('snapshot row size')
            row=decode(stream.read(size))
            if len(row)!=4 or not isinstance(row[1],Tensor):raise ValueError('snapshot row')
            rows.append(row)
        if stream.read(1):raise ValueError('snapshot trailing bytes')
    if len({r[0] for r in rows})!=len(rows):raise ValueError('snapshot duplicate IDs')
    return header,rows


def replay(bridge, source, output, queries):
    import chromadb
    import numpy as np
    from memory_collection import metadata_python
    from space_service_client import Tensor
    receipt={'status':'failed','collection':bridge.config['name'],'checks':[],
             'snapshot_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    def check(name,value):
        receipt['checks'].append({'name':name,'passed':bool(value)})
        if not value:raise AssertionError(name)
    started=time.monotonic()
    try:
        header,rows=read_snapshot(source)
        check('profile',header['profile']==bridge.config['profile'])
        native=bridge._native()
        ids=[r[0] for r in rows]
        vectors=np.array([np.frombuffer(r[1].encoded,offset=17,dtype='<f8') for r in rows])
        docs=[r[2] for r in rows];metas=[metadata_python(r[3]) for r in rows]
        data={'ids':ids,'embeddings':vectors,'documents':docs,'metadatas':metas}
        bridge._reconcile(data,native)
        loaded=native.get(include=['embeddings','documents','metadatas'])
        check('full-record-identity',loaded['ids']==ids and loaded['documents']==docs and loaded['metadatas']==metas)
        check('all-embedding-bits',all(array('d',a).tobytes()==b.tobytes() for a,b in zip(vectors,loaded['embeddings'])))
        comparison=chromadb.EphemeralClient().create_collection('replay-'+str(os.getpid()),embedding_function=None,
            configuration={'hnsw':{'space':'cosine','num_threads':1}})
        for start in range(0,len(ids),500):
            comparison.add(ids=ids[start:start+500],embeddings=vectors[start:start+500],
                           documents=docs[start:start+500],metadatas=metas[start:start+500])
        norms=np.linalg.norm(vectors,axis=1)
        check('finite-nonzero-source',bool(np.isfinite(vectors).all() and (norms>0).all()))
        rng=np.random.default_rng(10771)
        timings=[];overlaps=[];chroma_exact=[]
        for index in range(queries):
            # Recorded document vectors and seeded perturbations exercise real
            # geometry without retaining private queries or invoking a model.
            selected=int(rng.integers(len(ids)))
            q=vectors[selected].copy()
            if index%2:q+=rng.normal(0,.015,size=len(q))
            where=None
            mask=np.ones(len(ids),dtype=bool)
            if index%4==3 and 'time' in metas[selected]:
                value=metas[selected]['time'];where={'time':value}
                mask=np.array([m.get('time')==value for m in metas])
            scores=(vectors@q)/(norms*np.linalg.norm(q))
            eligible=np.where(mask)[0];k=min(5,len(eligible))
            expected=sorted(eligible,key=lambda i:(-scores[i],ids[i]))[:k]
            boundary=scores[expected[-1]] if expected else 0
            acceptable={ids[i] for i in eligible if scores[i]>=boundary-1e-10}
            mandatory={ids[i] for i in eligible if scores[i]>boundary+1e-10}
            start=time.perf_counter()
            actual=native.query([Tensor.vector(q)],5,where=where)
            timings.append((time.perf_counter()-start)*1000)
            selected_ids=set(actual['ids'][0])
            check('exact-query-'+str(index),len(selected_ids)==k and mandatory<=selected_ids<=acceptable)
            by_id={i:1-float(scores[n]) for n,i in enumerate(ids)}
            check('distances-'+str(index),all(abs(d-by_id[i])<2e-10 for i,d in zip(actual['ids'][0],actual['distances'][0])))
            other=comparison.query(query_embeddings=[q],n_results=5,where=where)
            overlaps.append(len(selected_ids&set(other['ids'][0]))/max(1,k))
            chroma_exact.append(len(set(other['ids'][0])&acceptable)/max(1,k))
        # Lifecycle on the private native copy; restore the original bytes.
        if ids:
            item=ids[0]
            native.delete(ids=[item]);check('private-delete',native.get(ids=[item])['ids']==[])
            native.upsert([item],[vectors[0]],[docs[0]],[metas[0]])
            native.update([item],[{'migration_probe':'temporary'}])
            check('private-update',native.get(ids=[item])['metadatas'][0]['migration_probe']=='temporary')
            native.upsert([item],[vectors[0]],[docs[0]],[metas[0]])
            native.worker.process.kill();native.worker.process.wait()
            check('worker-rebuild',native.get(ids=[item])['metadatas']==[metas[0]])
        native.client.call('compact',native.space.uuid)
        native.close();bridge.native=None
        reopened=bridge._native()
        check('reopen-count',reopened.count()==len(ids))
        receipt.update(status='passed',records=len(ids),dimensions=bridge.config['profile']['embedding_output_dim'],
            queries=queries,generation=header['generation'],native_revision=reopened.space.revision,
            native_p50_ms=float(np.median(timings)),native_p95_ms=float(np.percentile(timings,95)),
            chroma_overlap_mean=float(np.mean(overlaps)),chroma_exact_recall_mean=float(np.mean(chroma_exact)),
            chroma_exact_recall_min=float(min(chroma_exact)),duration_seconds=time.monotonic()-started,
            query_kind='recorded-document-vectors-and-seeded-perturbations',seed=10771)
    finally:atomic_json(output,receipt)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--chroma',required=True)
    p.add_argument('--collection',default='memories')
    p.add_argument('action',choices=('export','replay','sync','shadow','native','rollback','status'))
    p.add_argument('--snapshot',type=Path)
    p.add_argument('--receipt',type=Path)
    p.add_argument('--queries',type=int,default=128)
    args=p.parse_args()
    import chromadb
    original=chromadb.PersistentClient(path=args.chroma).get_collection(args.collection,embedding_function=None)
    bridge=MemoryBackend(original,args.config)
    if args.action=='export':result=snapshot(bridge,args.snapshot)
    elif args.action=='replay':
        if bridge._mode()!='chroma':raise ValueError('replay requires a non-serving native candidate')
        result=replay(bridge,args.snapshot,args.receipt,args.queries)
    elif args.action=='sync':bridge.synchronize();result={'status':'synchronized'}
    elif args.action=='status':result={'mode':bridge._mode(),'generation':bridge._epoch(),'count':bridge.count()}
    elif args.action=='rollback':
        with bridge.migration_guard(),bridge.guard():
            if bridge._mode()!='native':raise ValueError('rollback requires native mode')
            source=bridge._native().get(include=['embeddings','documents','metadatas'])
            old=original.get(include=['metadatas'])
            old_metadata=dict(zip(old['ids'],old['metadatas']))
            for offset in range(0,len(source['ids']),16):
                end=offset+16
                original.upsert(ids=source['ids'][offset:end],embeddings=[list(v) for v in source['embeddings'][offset:end]],
                    documents=source['documents'][offset:end],metadatas=source['metadatas'][offset:end])
                for item_id,metadata in zip(source['ids'][offset:end],source['metadatas'][offset:end]):
                    removed={k:None for k in (old_metadata.get(item_id) or {}) if k not in metadata}
                    if removed:original.update(ids=[item_id],metadatas=[removed])
            obsolete=sorted(set(old['ids'])-set(source['ids']))
            if obsolete:original.delete(ids=obsolete)
            copied=original.get(include=['embeddings','documents','metadatas'])
            comparison=verify_rollback(source,copied)
            config=json.loads(args.config.read_text());config['mode']='chroma';atomic_json(args.config,config)
            result={'mode':'chroma','generation':bridge._epoch(),'rollback':'verified-float32-roundtrip','comparison':comparison}
    else:
        with bridge.migration_guard(),bridge.guard():
            if bridge._mode()=='native':raise ValueError('native is authoritative; use rollback first')
            evidence=json.loads(args.receipt.read_text())
            if evidence.get('status')!='passed' or evidence.get('collection')!=bridge.config['name']:
                raise ValueError('passing qualification for this collection required')
            if args.action=='shadow':bridge._sync()
            else:
                # Verify complete source equality at the activation boundary.
                bridge._sync()
                first=original.get(include=['embeddings','documents','metadatas'])
                second=bridge._native().get(include=['embeddings','documents','metadatas'])
                def keyed(data):
                    return {i:(d,m,array('d',v).tobytes()) for i,d,m,v in
                            zip(data['ids'],data['documents'],data['metadatas'],data['embeddings'])}
                if keyed(first)!=keyed(second):raise ValueError('cutover source mismatch')
                atomic_json(bridge.state/'fallback.json',{'generation':bridge._epoch(),'current':True,'time':time.time()})
            # The running agent must not depend on a scratch qualification file.
            # Retain its own private, content-addressed receipt in durable state.
            digest=hashlib.sha256(args.receipt.read_bytes()).hexdigest()
            retained=bridge.state/'qualifications'/(digest+'.json')
            if not retained.exists():atomic_json(retained,evidence)
            if json.loads(retained.read_text())!=evidence:raise ValueError('retained qualification differs')
            config=json.loads(args.config.read_text())
            config.update(mode=args.action,qualification=str(retained.resolve()),
                          qualification_sha256=hashlib.sha256(retained.read_bytes()).hexdigest())
            atomic_json(args.config,config)
            result={'mode':args.action,'generation':bridge._epoch()}
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},sort_keys=True))
    if bridge.native:bridge.native.close()

if __name__=='__main__':main()
