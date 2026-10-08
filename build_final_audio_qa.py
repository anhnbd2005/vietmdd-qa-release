#!/usr/bin/env python3
"""Portable QA JSONL -> VietnameseQASLM FinalQA with WAV mapping and pair merge.

Standalone script (Python 3.9+).
Requires ffmpeg/ffprobe for pairs and durations. No pip dependencies.
Input audio: ["Med_CS-0-1.wav"] or ["Med_CS-0-1.wav","Med_CS-0-2.wav"].
Output schema: id,question,choices?,response_format,type_id,operator,audio_input,duration,text_output.

Example:
 python build_final_audio_qa.py --input /code/vimedcss/test.jsonl \\
  --output /data/FinalQA/test.jsonl --audio-root /data/ViMedCSS/audio \\
  --merged-dir /data/ViMedCSS/audio-pair-beep --expect-rows 16140
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from functools import lru_cache
from types import SimpleNamespace

import shutil
import subprocess

SCHEMA = ('id','question','choices','response_format','type_id','operator','audio_input','duration','text_output')

def within(p, root):
    try: p.relative_to(root);return True
    except ValueError:return False

def secure_path(value,root):
    p=Path(str(value))
    resolved=(p if p.is_absolute() else root/p).resolve()
    if not within(resolved,root):
        raise ValueError(f'Audio path escapes --audio-root: {value}')
    return resolved

def resolve(ref,mapping,root):
    if not isinstance(ref,str) or not ref:
        raise ValueError(f'Invalid audio reference {ref!r}')
    if ref in mapping:p=mapping[ref]
    else:
        # Only a directly usable filepath, or a filename already present beneath audio-root.
        # Never reverse-engineer a hash or select a fuzzy filename match.
        p=secure_path(ref,root)
        if not p.is_file() and '/' not in ref and '\\' not in ref:
            p=secure_path(ref+'.wav',root)
    if not p.is_file():
        raise ValueError(f'Cannot resolve {ref!r} to existing audio: {p}; supply --mapping')
    return p

def merge(a,b,out,args):
    if out.is_file():
        if out.stat().st_size==0:raise ValueError(f'Empty cached merge {out}')
        return
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('ffmpeg not found on PATH')
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out.parent,suffix='.wav',delete=False) as f:tmp=Path(f.name)
    try:
        # ffmpeg normalizes both inputs to mono PCM 16 kHz, inserts tone in between.
        filters=(f'[0:a]aresample={args.sample_rate},aformat=sample_fmts=s16:channel_layouts=mono[x];'
                 f'[1:a]aresample={args.sample_rate},aformat=sample_fmts=s16:channel_layouts=mono[y];'
                 f'[2:a]volume={args.beep_volume},aformat=sample_fmts=s16:channel_layouts=mono[z];'
                 '[x][z][y]concat=n=3:v=0:a=1[out]')
        cmd=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-y',
             '-i',str(a),'-i',str(b),'-f','lavfi','-i',
             f'sine=frequency={args.beep_hz}:duration={args.beep_seconds}:sample_rate={args.sample_rate}',
             '-filter_complex',filters,'-map','[out]','-ar',str(args.sample_rate),
             '-ac','1','-c:a','pcm_s16le',str(tmp)]
        subprocess.run(cmd,check=True)
        if tmp.stat().st_size<=44:raise RuntimeError('ffmpeg produced empty audio')
        os.replace(tmp,out)
    finally:
        tmp.unlink(missing_ok=True)

def select(row, keys):
    present = [(k,row[k]) for k in keys if k in row and row[k] is not None]
    if not present: return None
    if len(present)>1 and any(v!=present[0][1] for _,v in present[1:]):
        raise ValueError('Conflicting source fields: '+str([k for k,_ in present]))
    return present[0][1]

def duration_of(audio):
    p=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration',
                     '-of','default=noprint_wrappers=1:nokey=1',audio],
                     text=True,capture_output=True)
    if p.returncode: raise ValueError(f'ffprobe failed for {audio}: {p.stderr.strip()}')
    return float(p.stdout.strip())

def transform(row, infer_duration=False):
    if not isinstance(row,dict): raise ValueError('record must be an object')
    question=select(row,('question','prompt'))
    answer=select(row,('text_output','answer','gold_answer','gold','response','target'))
    audio=select(row,('audio_input','audio','audio_path'))
    choices=select(row,('choices','options'))
    if not isinstance(question,str) or not question.strip(): raise ValueError('missing question')
    if not isinstance(answer,str) or not answer.strip():
        raise ValueError(f'answer must be a non-empty string, got {answer!r}')
    if isinstance(audio,list):
        if len(audio)!=1: raise ValueError('audio list must contain exactly one mapped path; run pair merge first')
        audio=audio[0]
    if not isinstance(audio,str) or not audio.strip(): raise ValueError('missing mapped audio path')
    if choices is not None and (not isinstance(choices,list) or len(choices)<2 or not all(isinstance(x,str) for x in choices)):
        raise ValueError('choices must be a list of 2+ strings')
    if choices is not None and answer not in choices:
        raise ValueError(f'gold answer {answer!r} is not one of the choices; fix source representation first')
    fmt=select(row,('response_format',)) or ('mcq' if choices is not None else 'open_ended')
    if fmt=='mcq' and choices is None: raise ValueError('MCQ has no choices')
    if fmt!='mcq' and choices is not None: raise ValueError('choices exist but response_format is not mcq')
    duration=select(row,('duration','audio_duration'))
    if duration is None and infer_duration:
        duration=duration_of(audio)
    if duration is not None:
        if not isinstance(duration,(int,float)) or isinstance(duration,bool) or duration<0:
            raise ValueError('invalid duration')
    out={'id':select(row,('id','qa_id')),
         'question':question,'response_format':fmt,
         'type_id':select(row,('type_id','semantic_type')),
         'operator':select(row,('operator',)),
         'audio_input':audio,'text_output':answer}
    for field in ('id','type_id','operator'):
        if not isinstance(out[field],str) or not out[field].strip():
            raise ValueError(f'missing {field}; no implicit fabrication')
    if choices is not None:out['choices']=choices
    if duration is not None:out['duration']=duration
    return {k:out[k] for k in SCHEMA if k in out}



def parse():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    for a in ('input','output','audio-root','merged-dir'):
        p.add_argument('--'+a, type=Path, required=True)
    p.add_argument('--expect-rows',type=int,default=0)
    p.add_argument('--check-only',action='store_true',help='Validate all audio references and QA fields; do not create WAV/JSONL')
    p.add_argument('--force',action='store_true',help='Replace output JSONL (not source WAV files)')
    p.add_argument('--sample-rate',type=int,default=16000)
    p.add_argument('--beep-hz',type=int,default=1000)
    p.add_argument('--beep-seconds',type=float,default=0.25)
    p.add_argument('--beep-volume',type=float,default=0.25)
    p.add_argument('--report',type=Path)
    return p.parse_args()


@lru_cache(maxsize=200000)
def cached_duration(path:str)->float:
    return duration_of(path)


def main():
    a=parse()
    root=a.audio_root.resolve()
    merged=a.merged_dir.resolve()
    source=a.input.resolve()
    output=a.output.resolve()
    if not source.is_file(): raise ValueError(f'input not found: {source}')
    if not root.is_dir():raise ValueError(f'audio root not found: {root}')
    if source==output:raise ValueError('input and output must differ')
    if output.exists() and not a.force and not a.check_only:
        raise FileExistsError(f'output already exists: {output}; pass --force to replace')
    if a.expect_rows<0:raise ValueError('negative expected count')
    if a.sample_rate<=0 or a.beep_hz<=0 or a.beep_seconds<=0 or not (0<=a.beep_volume<=1):
        raise ValueError('invalid beep settings')
    stats={'rows':0,'single_audio_qa':0,'pairwise_qa':0,'mcq':0,'open_ended':0,
           'distinct_pair_wavs':0,'unique_original_wavs':0,'mode':'check_only' if a.check_only else 'write'}
    unique=set(); pairs=set(); temp=None; out_stream=None
    try:
        if not a.check_only:
            output.parent.mkdir(parents=True,exist_ok=True)
            out_stream=tempfile.NamedTemporaryFile('w',encoding='utf-8',newline='\n',dir=output.parent,suffix='.tmp',delete=False)
            temp=Path(out_stream.name)
        with source.open(encoding='utf-8-sig') as fh:
            for n,line in enumerate(fh,1):
                if not line.strip():raise ValueError(f'blank line {n}')
                try:
                    record=json.loads(line)
                    if not isinstance(record,dict):raise ValueError('expected JSON object')
                    refs=record.get('audio')
                    if isinstance(refs,str):refs=[refs]
                    if not isinstance(refs,list) or len(refs) not in (1,2):
                        raise ValueError('expected audio to be a list of 1 or 2 references')
                    paths=[resolve(ref,{},root) for ref in refs]
                    unique.update(map(str,paths))
                    if len(paths)==1:
                        dest=paths[0]
                        seconds=cached_duration(str(dest))
                        stats['single_audio_qa']+=1
                    else:
                        signature=json.dumps([str(paths[0]),str(paths[1]),a.sample_rate,a.beep_hz,a.beep_seconds,a.beep_volume],ensure_ascii=False)
                        key=hashlib.sha256(signature.encode('utf-8')).hexdigest()[:32]
                        dest=merged/(key+'.wav')
                        pairs.add(key)
                        if not a.check_only:
                            merge(paths[0],paths[1],dest,SimpleNamespace(sample_rate=a.sample_rate,beep_hz=a.beep_hz,
                                                                       beep_seconds=a.beep_seconds,beep_volume=a.beep_volume))
                            seconds=cached_duration(str(dest))
                        else:
                            seconds=cached_duration(str(paths[0]))+a.beep_seconds+cached_duration(str(paths[1]))
                        stats['pairwise_qa']+=1
                    adjusted=dict(record)
                    adjusted['audio']=str(dest)
                    # An existing source duration is no longer valid for two joined WAVs.
                    adjusted.pop('duration',None)
                    adjusted.pop('audio_duration',None)
                    adjusted['duration']=round(seconds,6)
                    final=transform(adjusted,infer_duration=False)
                    stats[final['response_format']]+=1 if final['response_format']=='mcq' else 0
                    if final['response_format']=='open_ended':stats['open_ended']+=1
                    if out_stream:out_stream.write(json.dumps(final,ensure_ascii=False)+'\n')
                    stats['rows']+=1
                except Exception as exc:
                    raise ValueError(f'{source}:{n}: {exc}') from exc
        if a.expect_rows and stats['rows']!=a.expect_rows:
            raise ValueError(f'expected {a.expect_rows} rows but found {stats["rows"]}')
        stats['unique_original_wavs']=len(unique)
        stats['distinct_pair_wavs']=len(pairs)
        if out_stream:out_stream.close();out_stream=None
        if temp:os.replace(temp,output);temp=None
        if not a.check_only and a.report:
            a.report.parent.mkdir(parents=True,exist_ok=True)
            a.report.write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(stats,ensure_ascii=False,indent=2))
    finally:
        if out_stream:out_stream.close()
        if temp:temp.unlink(missing_ok=True)


if __name__=='__main__':
    try:main()
    except (OSError,ValueError,RuntimeError) as exc:
        print('ERROR:',exc,file=sys.stderr)
        sys.exit(2)
