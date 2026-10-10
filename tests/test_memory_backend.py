import os,sys,json,time,subprocess,tempfile,argparse
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--cetta-root',type=Path,required=True)
p.add_argument('--server',type=Path,required=True)
p.add_argument('--worker',type=Path,required=True)
p.add_argument('--work-dir',type=Path)
args=p.parse_args()
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(args.cetta_root/'tools'))
from memory_backend import *
from memory_migrate import snapshot,replay,verify_rollback
from space_service_client import Client
import chromadb
root=args.cetta_root.resolve()
with tempfile.TemporaryDirectory(dir=args.work_dir,prefix='bridge-') as t:
 p=Path(t);(p/'data').mkdir(mode=0o700)
 service=subprocess.Popen([str(args.server.resolve()),'--data',str(p/'data'),'--socket',str(p/'socket')],stderr=subprocess.DEVNULL)
 try:
  for _ in range(100):
   if (p/'socket').exists():break
   time.sleep(.02)
  source=chromadb.PersistentClient(path=str(p/'chroma')).create_collection('bridge-fixture',embedding_function=None,configuration={'hnsw':{'space':'cosine','num_threads':1}})
  import numpy as np
  vectors=np.random.default_rng(988).normal(size=(257,16)).tolist()
  source.add(ids=[str(i) for i in range(257)],documents=['doc'+str(i) for i in range(257)],metadatas=[{'time':'day'+str(i%4)} for i in range(257)],embeddings=vectors)
  # The rollback allowance must reject material vector changes.
  original=source.get(include=['embeddings','documents','metadatas'])
  broken=copy.deepcopy(original);broken['embeddings'][0][0]+=0.01
  try:verify_rollback(original,broken)
  except ValueError:pass
  else:raise AssertionError('rollback accepted a corrupted embedding')
  config={'mode':'chroma','state_dir':str(p/'state'),'client_dir':str(root/'tools'),'socket':str(p/'socket'),'name':'test','profile':{'embedding_output_dim':16},'worker':str(args.worker.resolve()),'module':str(root/'lib/petta/lib_memory_collection.metta')}
  atomic_json(p/'config.json',config);b=MemoryBackend(source,p/'config.json')
  # Snapshot callers can nest an operation on this same instance; the outer
  # file lock must still exclude a writer after the inner guard returns.
  import signal
  def guard_timeout(*args):raise AssertionError('nested collection guard blocked')
  previous=signal.signal(signal.SIGALRM,guard_timeout)
  signal.alarm(5)
  try:
   with b.guard():
    with b.guard():assert b.count()==257
    probe='''import fcntl,sys
with open(sys.argv[1], 'a') as lock:
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:sys.exit(0)
 sys.exit(1)
'''
    subprocess.run([sys.executable,'-c',probe,str(b.state/'bridge.lock')],check=True)
  finally:
   signal.alarm(0);signal.signal(signal.SIGALRM,previous)
  print('PASS nested collection guard and writer exclusion')
  os.environ['METTACLAW_MEMORY_BACKEND_CONFIG']=str(p/'config.json')
  try:
   try:configured(source,runtime_profile={'embedding_output_dim':17})
   except ValueError:pass
   else:raise AssertionError('runtime profile mismatch accepted')
   config['runtime_profile']={'embedding_output_dim':16,'embedding_model':'reviewed-local-provider'}
   atomic_json(p/'config.json',config)
   assert configured(source,runtime_profile=config['runtime_profile']).config['profile']==config['profile']
  finally:del os.environ['METTACLAW_MEMORY_BACKEND_CONFIG']
  assert snapshot(b,p/'snapshot')['records']==257
  result=replay(b,p/'snapshot',p/'receipt.json',16);print({k:v for k,v in result.items() if k!='checks'})
  command=[sys.executable,'-B',str(Path(__file__).resolve().parents[1]/'memory_migrate.py'),
           '--config',str(p/'config.json'),'--chroma',str(p/'chroma'),'--collection','bridge-fixture']
  subprocess.run(command+['shadow','--receipt',str(p/'receipt.json')],check=True)
  config=json.loads((p/'config.json').read_text())
  q={'query_embeddings':[vectors[0]],'n_results':5}
  assert b.query(**q)['ids']==source.query(**q)['ids']
  b.work.join();assert b.stats['compared']==1,b.stats
  b.add(ids=['new'],embeddings=[vectors[0]],documents=['new'],metadatas=[{'time':'newday'}])
  assert b.query(**q)['ids']==source.query(**q)['ids'];b.work.join();assert b.stats['compared']==2,b.stats
  b.delete(ids=['new']);assert b.get(ids=['new'])['ids']==[]
  # Missing synchronization receipts force reconstruction from Chroma.
  atomic_json(b.state/'synchronized.json',{'generation':None})
  b.query(**q);b.work.join();assert b.stats.get('rebuilds')==1,b.stats
  # A blocked shadow worker must not block foreground Chroma queries.
  import threading
  entered=threading.Event();release=threading.Event()
  class SlowNative:
   def query(self,**kwargs):
    entered.set();assert release.wait(5);return source.query(**kwargs)
   def close(self):pass
  slow=MemoryBackend(source,p/'config.json');slow.native_type=lambda *args:SlowNative()
  slow.query(**q);assert entered.wait(2)
  start=time.monotonic()
  for _ in range(25):slow.query(**q)
  elapsed=time.monotonic()-start
  assert elapsed<2 and slow.stats['skipped']>0,(elapsed,slow.stats)
  release.set();slow.work.join()
  subprocess.run(command+['native','--receipt',str(p/'receipt.json')],check=True)
  config=json.loads((p/'config.json').read_text())
  assert Path(config['qualification']).parent==b.state/'qualifications'
  (p/'receipt.json').unlink()
  assert b._mode()=='native'
  assert b.query(**q)['ids'][0][0]=='0'
  b.add(ids=['native-new'],embeddings=[vectors[0]],documents=['native-new'],metadatas=[{'time':'newday'}]);assert source.get(ids=['native-new'])['ids']==['native-new']
  b.delete(ids=['native-new']);assert source.get(ids=['native-new'])['ids']==[]
  assert json.loads((b.state/'fallback.json').read_text())['current']
  print('PASS bridge shadow/native CRUD and replay')
  b.native.close()
  subprocess.run(command+['rollback'],check=True)
  assert json.loads((p/'config.json').read_text())['mode']=='chroma'
  print('PASS bounded shadow queue, nonblocking foreground, restart rebuilding')
 finally:
  service.terminate();service.wait()
