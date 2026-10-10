#!/usr/bin/env python3
"""Create a private Chroma-primary migration configuration from a local release."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

from memory_backend import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True)
    parser.add_argument('--profile',required=True,type=Path)
    parser.add_argument('--runtime-profile',type=Path)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('memory_backend.json'))
    parser.add_argument('--release',type=Path,default=Path.home()/'.local/lib/cetta-memory/current')
    parser.add_argument('--socket',type=Path)
    parser.add_argument('--state-dir',type=Path)
    args=parser.parse_args()
    if args.config.exists():raise ValueError('configuration exists; review changes explicitly')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',args.name) or args.name in ('.','..'):
        raise ValueError('name must be a simple collection identifier')
    profile=json.loads(args.profile.read_text())
    if not isinstance(profile.get('embedding_output_dim'),int) or profile['embedding_output_dim']<1:
        raise ValueError('profile requires positive embedding_output_dim')
    if profile.get('hnsw:space','cosine')!='cosine':raise ValueError('cosine profile required')
    release=args.release.resolve();manifest=json.loads((release/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts:raise ValueError('invalid release manifest path')
        if hashlib.sha256((release/path).read_bytes()).hexdigest()!=digest:
            raise ValueError('installed release hash mismatch')
    state=args.state_dir or Path(os.environ.get('XDG_STATE_HOME',Path.home()/'.local/state'))/'cetta-memory-client'/args.name
    socket=args.socket or Path(os.environ.get('XDG_RUNTIME_DIR',f'/run/user/{os.getuid()}'))/'cetta-memory.sock'
    config={'mode':'chroma','state_dir':str(state.resolve()),'socket':str(socket.resolve()),
            'name':args.name,'profile':profile,'client_dir':str(release/'tools'),
            'worker':str(release/'bin/cetta-value-worker'),
            'module':str(release/'lib/petta/lib_memory_collection.metta')}
    if args.runtime_profile:
        config['runtime_profile']=json.loads(args.runtime_profile.read_text())
    atomic_json(args.config,config)
    print('Created private configuration in chroma mode; replay before activation.')


if __name__=='__main__':main()
