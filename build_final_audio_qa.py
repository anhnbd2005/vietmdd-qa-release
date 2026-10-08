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
import math
import wave
from pathlib import Path
import sys
import tempfile
from functools import lru_cache
from types import SimpleNamespace

import shutil
import subprocess
import time

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

def run_checked(cmd, timeout, label):
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f'{label}: exceeded {timeout}s timeout') from exc
    if result.returncode:
        raise ValueError(f'{label}: exit={result.returncode}: {result.stderr.strip()[-1200:]}')
    return result


def validate_wav(path, *, expected_rate=None):
    """Fail closed on header-only, truncated, non-PCM, or zero-frame WAVs."""
    try:
        with wave.open(str(path), 'rb') as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            if frames <= 0 or rate <= 0 or channels <= 0 or width <= 0:
                raise ValueError('no audio frames or invalid WAV format')
            if expected_rate and rate != expected_rate:
                raise ValueError(f'unexpected sample rate {rate}, expected {expected_rate}')
            # For uncompressed PCM, verify that physical data bytes exist.
            expected_bytes = frames * channels * width
            actual_bytes = 0
            while actual_bytes < expected_bytes:
                chunk = wav.readframes(min(65536, (expected_bytes - actual_bytes + channels * width - 1) // (channels * width)))
                if not chunk:
                    break
                actual_bytes += len(chunk)
            if actual_bytes != expected_bytes:
                raise ValueError(f'truncated PCM payload: {actual_bytes}/{expected_bytes} bytes')
            return frames / rate
    except (wave.Error, EOFError) as exc:
        raise ValueError(f'invalid/unreadable WAV {path}: {exc}') from exc


def merge(a,b,out,args,seconds_a,seconds_b):
    expected_seconds = seconds_a + args.beep_seconds + seconds_b
    if out.is_file():
        try:
            cached_seconds = validate_wav(out, expected_rate=args.sample_rate)
            if abs(cached_seconds - expected_seconds) > max(0.1, expected_seconds * 0.002):
                raise ValueError('cached duration mismatch')
            return
        except (OSError, ValueError):
            # Never silently overwrite an existing corrupt cache entry.
            raise ValueError(f'Invalid cached merge {out}; delete it explicitly before retrying')
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
        timeout = max(args.merge_timeout, min(args.max_merge_timeout, 30 + expected_seconds * 3))
        run_checked(cmd, timeout, f'ffmpeg merge {a.name} + {b.name}')
        produced_seconds = validate_wav(tmp, expected_rate=args.sample_rate)
        if abs(produced_seconds - expected_seconds) > max(0.1, expected_seconds * 0.002):
            raise ValueError(f'merged duration mismatch: {produced_seconds} vs {expected_seconds}')
        os.replace(tmp,out)
    finally:
        tmp.unlink(missing_ok=True)

def select(row, keys):
    present = [(k,row[k]) for k in keys if k in row and row[k] is not None]
    if not present: return None
    if len(present)>1 and any(v!=present[0][1] for _,v in present[1:]):
        raise ValueError('Conflicting source fields: '+str([k for k,_ in present]))
    return present[0][1]

def duration_of(audio, probe_timeout=15):
    if shutil.which('ffprobe') is None:
        raise RuntimeError('ffprobe not found on PATH')
    result = run_checked(
        ['ffprobe', '-v', 'error', '-show_entries',
         'stream=codec_type,sample_rate,channels,duration:format=duration',
         '-select_streams', 'a:0', '-of', 'json', str(audio)],
        probe_timeout, f'ffprobe {audio}')
    try:
        obj = json.loads(result.stdout)
        streams = obj.get('streams') or []
        if not streams or streams[0].get('codec_type') != 'audio':
            raise ValueError('no audio stream')
        duration = float((obj.get('format') or {}).get('duration') or streams[0]['duration'])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('invalid or zero audio duration')
        if int(streams[0].get('sample_rate', 0)) <= 0 or int(streams[0].get('channels', 0)) <= 0:
            raise ValueError('invalid audio stream parameters')
        return duration
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ValueError(f'Invalid audio metadata {audio}: {exc}') from exc


def decoded_duration(audio, timeout):
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('ffmpeg not found on PATH')
    run_checked(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin',
                 '-i', str(audio), '-map', '0:a:0', '-f', 'null', '-'],
                timeout, f'ffmpeg decode {audio}')


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
    p.add_argument('--probe-timeout',type=float,default=15,help='ffprobe timeout per WAV, seconds')
    p.add_argument('--merge-timeout',type=float,default=60,help='minimum ffmpeg merge timeout, seconds')
    p.add_argument('--max-merge-timeout',type=float,default=900,help='maximum dynamic merge timeout, seconds')
    p.add_argument('--deep-audio-check',action='store_true',help='ffmpeg-decode every distinct original WAV once (including --check-only)')
    p.add_argument('--progress-every',type=int,default=100,help='print progress to stderr every N QA rows (0 disables)')
    return p.parse_args()


@lru_cache(maxsize=200000)
def cached_duration(path:str, probe_timeout:float)->float:
    # WAV integrity includes verifying actual PCM data, not just a plausible header.
    pcm_seconds = validate_wav(Path(path))
    probed_seconds = duration_of(path, probe_timeout)
    if abs(pcm_seconds - probed_seconds) > max(0.1, pcm_seconds * 0.002):
        raise ValueError(f'WAV duration mismatch (PCM/ffprobe): {path}')
    return pcm_seconds


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
    if a.progress_every<0:raise ValueError('negative --progress-every')
    if a.probe_timeout<=0 or a.merge_timeout<=0 or a.max_merge_timeout<a.merge_timeout:
        raise ValueError('invalid timeout settings')
    if a.sample_rate<=0 or a.beep_hz<=0 or a.beep_seconds<=0 or not (0<=a.beep_volume<=1):
        raise ValueError('invalid beep settings')
    stats={'rows':0,'single_audio_qa':0,'pairwise_qa':0,'mcq':0,'open_ended':0,
           'distinct_pair_wavs':0,'unique_original_wavs':0,'mode':'check_only' if a.check_only else 'write'}
    unique=set(); pairs=set(); decoded=set(); temp=None; out_stream=None
    started=time.monotonic()
    def progress(current, current_audio='', force=False):
        if not force and (a.progress_every==0 or current % a.progress_every):
            return
        elapsed=max(time.monotonic()-started,0.001)
        total=f'/{a.expect_rows:,}' if a.expect_rows else ''
        pct=f' ({100*current/a.expect_rows:.1f}%)' if a.expect_rows else ''
        print(f'[PROGRESS] {current:,}{total} QA{pct} | {current/elapsed:.1f} QA/s | WAV checked={len(unique):,} | WAV decoded={len(decoded):,} | pairs={len(pairs):,} | current={current_audio}', file=sys.stderr, flush=True)
    try:
        if not a.check_only:
            output.parent.mkdir(parents=True,exist_ok=True)
            out_stream=tempfile.NamedTemporaryFile('w',encoding='utf-8',newline='\n',dir=output.parent,suffix='.tmp',delete=False)
            temp=Path(out_stream.name)
        print(f'[START] mode={stats["mode"]}, deep_audio_check={a.deep_audio_check}, expected_rows={a.expect_rows or "unknown"}, probe_timeout={a.probe_timeout}s, progress_every={a.progress_every}', file=sys.stderr, flush=True)
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
                    if n == 1:
                        print(f'[AUDIO] first={paths[0]}',file=sys.stderr,flush=True)
                    unique.update(map(str,paths))
                    durations = [cached_duration(str(p), a.probe_timeout) for p in paths]
                    if a.deep_audio_check:
                        for path, seconds in zip(paths, durations):
                            if str(path) not in decoded:
                                decoded_duration(path, max(a.merge_timeout, min(a.max_merge_timeout, 30 + 3 * seconds)))
                                decoded.add(str(path))
                    if len(paths)==1:
                        dest=paths[0]
                        seconds=durations[0]
                        stats['single_audio_qa']+=1
                    else:
                        signature=json.dumps([[str(p),p.stat().st_size,p.stat().st_mtime_ns] for p in paths] + [a.sample_rate,a.beep_hz,a.beep_seconds,a.beep_volume],ensure_ascii=False)
                        key=hashlib.sha256(signature.encode('utf-8')).hexdigest()[:32]
                        dest=merged/(key+'.wav')
                        pairs.add(key)
                        if not a.check_only:
                            merge(paths[0],paths[1],dest,a,durations[0],durations[1])
                            seconds=cached_duration(str(dest),a.probe_timeout)
                        else:
                            seconds=durations[0]+a.beep_seconds+durations[1]
                        stats['pairwise_qa']+=1
                    adjusted=dict(record)
                    adjusted['audio']=str(dest)
                    # An existing source duration is no longer valid for two joined WAVs.
                    adjusted.pop('duration',None)
                    adjusted.pop('audio_duration',None)
                    adjusted['duration']=round(seconds,6)
                    final=transform(adjusted,infer_duration=False)
                    stats[final['response_format']]+=1
                    if out_stream:out_stream.write(json.dumps(final,ensure_ascii=False)+'\n')
                    stats['rows']+=1
                    progress(stats['rows'], paths[0].name)
                except Exception as exc:
                    raise ValueError(f'{source}:{n}: {exc}') from exc
        if a.expect_rows and stats['rows']!=a.expect_rows:
            raise ValueError(f'expected {a.expect_rows} rows but found {stats["rows"]}')
        if not a.progress_every or stats['rows'] % a.progress_every:
            progress(stats['rows'], 'completed', force=True)
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
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
        print('ERROR:',exc,file=sys.stderr)
        sys.exit(2)
