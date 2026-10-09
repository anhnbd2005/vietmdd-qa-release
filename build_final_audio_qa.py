#!/usr/bin/env python3
"""Build FinalQA JSONL from one/two source WAV references, merging pairs with a beep.

Compatible CLI with the original vietmdd-qa-release/build_final_audio_qa.py.
Supports PCM integer and IEEE Float32 source WAVs. Requires ffmpeg and ffprobe.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from functools import lru_cache
from pathlib import Path

SCHEMA = ('id','question','choices','response_format','type_id','operator','audio_input','duration','text_output')


def run_checked(cmd, timeout, label):
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f'{label}: exceeded {timeout}s timeout') from exc
    if result.returncode:
        raise ValueError(f'{label}: exit={result.returncode}: {result.stderr.strip()[-1200:]}')
    return result


def within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def secure_path(value, root):
    path = Path(str(value))
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not within(resolved, root):
        raise ValueError(f'Audio path escapes --audio-root: {value}')
    return resolved


def resolve(ref, mapping, root):
    if not isinstance(ref, str) or not ref:
        raise ValueError(f'Invalid audio reference {ref!r}')
    if ref in mapping:
        path = mapping[ref]
    else:
        path = secure_path(ref, root)
        if not path.is_file() and '/' not in ref and '\\' not in ref:
            path = secure_path(ref + '.wav', root)
    if not path.is_file():
        raise ValueError(f'Cannot resolve {ref!r} to existing audio: {path}; supply --mapping')
    return path


def probe_audio(path, timeout=15):
    if shutil.which('ffprobe') is None:
        raise RuntimeError('ffprobe not found on PATH')
    result = run_checked([
        'ffprobe', '-v', 'error', '-select_streams', 'a:0',
        '-show_entries', 'stream=codec_type,codec_name,sample_rate,channels,duration:format=duration',
        '-of', 'json', str(path)
    ], timeout, f'ffprobe {path}')
    try:
        obj = json.loads(result.stdout)
        streams = obj.get('streams') or []
        if len(streams) != 1 or streams[0].get('codec_type') != 'audio':
            raise ValueError('missing audio stream')
        stream = streams[0]
        codec = stream['codec_name']
        rate = int(stream['sample_rate'])
        channels = int(stream['channels'])
        duration = float((obj.get('format') or {}).get('duration') or stream['duration'])
        if rate <= 0 or channels <= 0 or not math.isfinite(duration) or duration <= 0:
            raise ValueError('invalid audio stream parameters or duration')
        return codec, rate, channels, duration
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ValueError(f'Invalid audio metadata {path}: {exc}') from exc


def decoded_duration(audio, timeout):
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('ffmpeg not found on PATH')
    # -xerror aborts on decoding errors instead of concealing corrupt/truncated audio.
    run_checked(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-xerror', '-nostdin',
                 '-i', str(audio), '-map', '0:a:0', '-f', 'null', '-'],
                timeout, f'ffmpeg decode {audio}')


def validate_wav(path, *, expected_rate=None, probe_timeout=15):
    """Strict validation for integer PCM and IEEE Float32 WAV source audio.

    Standard PCM: read every declared PCM byte using Python wave.
    IEEE Float32: verify RIFF data bytes and fully decode with FFmpeg.
    Unexpected codecs are rejected. Merged output is required to be PCM16.
    """
    path = Path(path)
    codec, rate, channels, seconds = probe_audio(path, probe_timeout)
    if expected_rate is not None and rate != expected_rate:
        raise ValueError(f'unexpected sample rate {rate}, expected {expected_rate}')
    if expected_rate is not None and codec != 'pcm_s16le':
        raise ValueError(f'unexpected merged WAV codec {codec}, expected pcm_s16le')
    if codec in ('pcm_s16le','pcm_s24le','pcm_s32le','pcm_u8'):
        try:
            with wave.open(str(path), 'rb') as wav:
                frames = wav.getnframes()
                sample_rate = wav.getframerate()
                nchannels = wav.getnchannels()
                width = wav.getsampwidth()
                if frames <= 0 or sample_rate != rate or nchannels != channels or width <= 0:
                    raise ValueError('invalid WAV header/frames')
                expected_bytes = frames * nchannels * width
                read_bytes = 0
                while read_bytes < expected_bytes:
                    block = wav.readframes(min(65536, (expected_bytes-read_bytes + nchannels*width-1)//(nchannels*width)))
                    if not block:
                        break
                    read_bytes += len(block)
                if read_bytes != expected_bytes:
                    raise ValueError(f'truncated PCM payload: {read_bytes}/{expected_bytes} bytes')
                actual_seconds = frames / sample_rate
        except (wave.Error, EOFError) as exc:
            raise ValueError(f'invalid/unreadable WAV {path}: {exc}') from exc
    elif codec == 'pcm_f32le' and expected_rate is None:
        # Python wave before 3.15 rejects WAVE_FORMAT_IEEE_FLOAT (tag 3).
        # FFprobe confirms the codec; full FFmpeg decode checks the entire payload.
        # First check container end position to reject subtly truncated WAV files.
        import struct
        with path.open('rb') as f:
            header = f.read(12)
            if len(header) != 12 or header[:4] != b'RIFF' or header[8:] != b'WAVE':
                raise ValueError(f'invalid WAV RIFF header: {path}')
            riff_size = struct.unpack('<I', header[4:8])[0]
            if path.stat().st_size < riff_size + 8:
                raise ValueError(f'truncated RIFF file {path}: {path.stat().st_size}/{riff_size+8} bytes')
            data_bytes = 0
            while f.tell() + 8 <= riff_size + 8:
                chunk_header = f.read(8)
                if len(chunk_header) < 8:
                    raise ValueError(f'truncated WAV chunk header: {path}')
                chunk_size = struct.unpack('<I', chunk_header[4:])[0]
                if chunk_header[:4] == b'data':
                    data_bytes += chunk_size
                if f.tell() + chunk_size > path.stat().st_size:
                    raise ValueError(f'truncated WAV chunk: {path}')
                f.seek(chunk_size + chunk_size % 2, 1)
            if data_bytes <= 0 or data_bytes % (4*channels):
                raise ValueError(f'invalid Float32 payload size: {path}')
        actual_seconds = data_bytes / (4 * channels * rate)
        if abs(actual_seconds-seconds) > max(0.1, actual_seconds*0.002):
            raise ValueError(f'Float32 duration mismatch {actual_seconds} vs {seconds}: {path}')
        decoded_duration(path, max(60, min(900, 30 + actual_seconds*3)))
    else:
        raise ValueError(f'unsupported WAV codec {codec}: {path}')
    if abs(actual_seconds-seconds) > max(0.1, actual_seconds*0.002):
        raise ValueError(f'WAV duration mismatch {actual_seconds} vs {seconds}: {path}')
    return actual_seconds


@lru_cache(maxsize=200000)
def cached_duration(path, probe_timeout):
    return validate_wav(Path(path), probe_timeout=probe_timeout)


def merge(a, b, out, args, seconds_a, seconds_b):
    expected = seconds_a + args.beep_seconds + seconds_b
    if out.is_file():
        try:
            cached = validate_wav(out, expected_rate=args.sample_rate)
            if abs(cached-expected) > max(0.1, expected*0.002):
                raise ValueError('cached duration mismatch')
            return
        except (ValueError, OSError) as exc:
            raise ValueError(f'Invalid cached merge {out}; delete explicitly before retrying: {exc}') from exc
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('ffmpeg not found on PATH')
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out.parent,suffix='.wav',delete=False) as f:
        tmp = Path(f.name)
    try:
        filters = (f'[0:a]aresample={args.sample_rate},aformat=sample_fmts=s16:channel_layouts=mono[x];'
                   f'[1:a]aresample={args.sample_rate},aformat=sample_fmts=s16:channel_layouts=mono[y];'
                   f'[2:a]volume={args.beep_volume},aformat=sample_fmts=s16:channel_layouts=mono[z];'
                   '[x][z][y]concat=n=3:v=0:a=1[out]')
        cmd = ['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-y',
               '-i',str(a),'-i',str(b),'-f','lavfi','-i',
               f'sine=frequency={args.beep_hz}:duration={args.beep_seconds}:sample_rate={args.sample_rate}',
               '-filter_complex',filters,'-map','[out]','-ar',str(args.sample_rate),
               '-ac','1','-c:a','pcm_s16le',str(tmp)]
        timeout=max(args.merge_timeout,min(args.max_merge_timeout,30+expected*3))
        run_checked(cmd,timeout,f'ffmpeg merge {a.name} + {b.name}')
        actual=validate_wav(tmp,expected_rate=args.sample_rate)
        if abs(actual-expected) > max(0.1,expected*0.002):
            raise ValueError(f'merged duration mismatch {actual} vs {expected}')
        os.replace(tmp,out)
    finally:
        tmp.unlink(missing_ok=True)


def select(row,keys):
    present=[(k,row[k]) for k in keys if k in row and row[k] is not None]
    if not present:return None
    if len(present)>1 and any(v!=present[0][1] for _,v in present[1:]):
        raise ValueError('Conflicting source fields: '+str([k for k,_ in present]))
    return present[0][1]


def transform(row):
    if not isinstance(row,dict):raise ValueError('record must be an object')
    question=select(row,('question','prompt'))
    answer=select(row,('text_output','answer','gold_answer','gold','response','target'))
    audio=select(row,('audio_input','audio','audio_path'))
    choices=select(row,('choices','options'))
    if not isinstance(question,str) or not question.strip():raise ValueError('missing question')
    if not isinstance(answer,str) or not answer.strip():raise ValueError('answer must be a nonempty string')
    if isinstance(audio,list):
        if len(audio)!=1:raise ValueError('audio list must contain exactly one mapped path; run pair merge first')
        audio=audio[0]
    if not isinstance(audio,str) or not audio.strip():raise ValueError('missing mapped audio path')
    if choices is not None and (not isinstance(choices,list) or len(choices)<2 or not all(isinstance(x,str) for x in choices)):
        raise ValueError('choices must be a list of 2+ strings')
    if choices is not None and answer not in choices:raise ValueError('answer not in choices')
    fmt=select(row,('response_format',)) or ('mcq' if choices is not None else 'open_ended')
    if fmt=='mcq' and choices is None:raise ValueError('MCQ has no choices')
    if fmt!='mcq' and choices is not None:raise ValueError('choices exist but format is not mcq')
    duration=select(row,('duration','audio_duration'))
    if duration is not None and (not isinstance(duration,(int,float)) or isinstance(duration,bool) or not math.isfinite(duration) or duration<0):
        raise ValueError('invalid duration')
    result={'id':select(row,('id','qa_id')),'question':question,'response_format':fmt,
            'type_id':select(row,('type_id','semantic_type')),'operator':select(row,('operator',)),
            'audio_input':audio,'text_output':answer}
    for key in ('id','type_id','operator'):
        if not isinstance(result[key],str) or not result[key].strip():raise ValueError(f'missing {key}')
    if choices is not None:result['choices']=choices
    if duration is not None:result['duration']=duration
    return {k:result[k] for k in SCHEMA if k in result}


def parse():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('input','output','audio-root','merged-dir'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--expect-rows',type=int,default=0)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--skip-invalid-audio',action='store_true')
    parser.add_argument('--force',action='store_true')
    parser.add_argument('--sample-rate',type=int,default=16000)
    parser.add_argument('--beep-hz',type=int,default=1000)
    parser.add_argument('--beep-seconds',type=float,default=0.25)
    parser.add_argument('--beep-volume',type=float,default=0.25)
    parser.add_argument('--report',type=Path)
    parser.add_argument('--probe-timeout',type=float,default=15)
    parser.add_argument('--merge-timeout',type=float,default=60)
    parser.add_argument('--max-merge-timeout',type=float,default=900)
    parser.add_argument('--deep-audio-check',action='store_true')
    parser.add_argument('--progress-every',type=int,default=100)
    return parser.parse_args()


def main():
    args=parse()
    root=args.audio_root.resolve()
    merged=args.merged_dir.resolve()
    source=args.input.resolve()
    output=args.output.resolve()
    if not source.is_file():raise ValueError(f'input not found: {source}')
    if not root.is_dir():raise ValueError(f'audio root not found: {root}')
    if source==output:raise ValueError('input and output must differ')
    if output.exists() and not args.force and not args.check_only:raise FileExistsError(f'output exists: {output}; use --force')
    if args.expect_rows<0 or args.progress_every<0:raise ValueError('negative count')
    if args.probe_timeout<=0 or args.merge_timeout<=0 or args.max_merge_timeout<args.merge_timeout:raise ValueError('invalid timeouts')
    if args.sample_rate<=0 or args.beep_hz<=0 or args.beep_seconds<=0 or not 0<=args.beep_volume<=1:raise ValueError('invalid beep settings')
    stats={'rows':0,'input_rows':0,'skipped_rows':0,'invalid_original_wavs':0,'single_audio_qa':0,'pairwise_qa':0,
           'mcq':0,'open_ended':0,'distinct_pair_wavs':0,'unique_original_wavs':0,
           'mode':'check_only' if args.check_only else 'write'}
    unique=set();pairs=set();decoded=set();bad_audio={}; temp=None; skipped_tmp=None; out_stream=None; skipped_stream=None
    start=time.monotonic()
    def progress(n,name,force=False):
        if not force and (args.progress_every==0 or n%args.progress_every):return
        elapsed=max(time.monotonic()-start,0.001)
        total=f'/{args.expect_rows:,}' if args.expect_rows else ''
        print(f'[PROGRESS] input={n:,}{total} | kept={stats["rows"]:,} skipped={stats["skipped_rows"]:,} | '
              f'{n/elapsed:.1f} input/s | WAV checked={len(unique):,} invalid={len(bad_audio):,} '
              f'decoded={len(decoded):,} pairs={len(pairs):,} | current={name}',file=sys.stderr,flush=True)
    try:
        if not args.check_only:
            output.parent.mkdir(parents=True,exist_ok=True)
            out_stream=tempfile.NamedTemporaryFile('w',encoding='utf-8',newline='\n',dir=output.parent,suffix='.tmp',delete=False)
            temp=Path(out_stream.name)
            if args.skip_invalid_audio:
                skipped_stream=tempfile.NamedTemporaryFile('w',encoding='utf-8',newline='\n',dir=output.parent,suffix='.skipped.tmp',delete=False)
                skipped_tmp=Path(skipped_stream.name)
        print(f'[START] mode={stats["mode"]}, deep_audio_check={args.deep_audio_check}, expected_rows={args.expect_rows or "unknown"}, probe_timeout={args.probe_timeout}s, progress_every={args.progress_every}',file=sys.stderr,flush=True)
        with source.open(encoding='utf-8-sig') as handle:
            for n,line in enumerate(handle,1):
                stats['input_rows']+=1
                if not line.strip():raise ValueError(f'blank line {n}')
                try:
                    record=json.loads(line)
                    if not isinstance(record,dict):raise ValueError('expected object')
                    refs=record.get('audio')
                    if isinstance(refs,str):refs=[refs]
                    if not isinstance(refs,list) or len(refs) not in (1,2):raise ValueError('audio must be array of 1 or 2 refs')
                    paths=[];invalid=[]
                    for ref in refs:
                        try:
                            path=resolve(ref,{},root)
                            paths.append(path);unique.add(str(path))
                            if str(path) in bad_audio:invalid.append({'ref':ref,'error':bad_audio[str(path)]})
                        except (OSError,ValueError) as exc:
                            bad_audio[f'unresolved:{ref}']=str(exc)
                            invalid.append({'ref':ref,'error':str(exc)})
                    if n==1 and paths:print(f'[AUDIO] first={paths[0]}',file=sys.stderr,flush=True)
                    durations=[]
                    if not invalid:
                        for path in paths:
                            try:
                                if str(path) in bad_audio:raise ValueError(bad_audio[str(path)])
                                seconds=cached_duration(str(path),args.probe_timeout)
                                if args.deep_audio_check and str(path) not in decoded:
                                    decoded_duration(path,max(args.merge_timeout,min(args.max_merge_timeout,30+3*seconds)))
                                    decoded.add(str(path))
                                durations.append(seconds)
                            except (OSError,ValueError) as exc:
                                bad_audio[str(path)]=str(exc)
                                invalid.append({'ref':path.name,'error':str(exc)})
                    if invalid:
                        if not args.skip_invalid_audio:raise ValueError(f'invalid original audio: {invalid}')
                        stats['skipped_rows']+=1
                        if skipped_stream:
                            skipped_stream.write(json.dumps({'line':n,'id':record.get('id'),'audio':refs,'invalid_audio':invalid},ensure_ascii=False)+'\n')
                        if stats['skipped_rows']<=5:print(f'[SKIP] line={n} audio={invalid}',file=sys.stderr,flush=True)
                        progress(n,str(refs[0]));continue
                    if len(paths)==1:
                        dest=paths[0]; seconds=durations[0];stats['single_audio_qa']+=1
                    else:
                        signature=json.dumps([[str(p),p.stat().st_size,p.stat().st_mtime_ns] for p in paths]+[
                            args.sample_rate,args.beep_hz,args.beep_seconds,args.beep_volume],ensure_ascii=False)
                        key=hashlib.sha256(signature.encode('utf-8')).hexdigest()[:32]
                        dest=merged/(key+'.wav');pairs.add(key)
                        if not args.check_only:
                            merge(paths[0],paths[1],dest,args,durations[0],durations[1])
                            seconds=cached_duration(str(dest),args.probe_timeout)
                        else:
                            seconds=durations[0]+args.beep_seconds+durations[1]
                        stats['pairwise_qa']+=1
                    adjusted=dict(record)
                    adjusted['audio']=str(dest)
                    adjusted.pop('duration',None);adjusted.pop('audio_duration',None)
                    adjusted['duration']=round(seconds,6)
                    final=transform(adjusted)
                    stats[final['response_format']]+=1
                    if out_stream:out_stream.write(json.dumps(final,ensure_ascii=False)+'\n')
                    stats['rows']+=1
                    progress(n,paths[0].name)
                except Exception as exc:
                    raise ValueError(f'{source}:{n}: {exc}') from exc
        if args.expect_rows and stats['input_rows']!=args.expect_rows:
            raise ValueError(f'expected {args.expect_rows} input rows but found {stats["input_rows"]}')
        progress(stats['input_rows'],'completed',force=True)
        stats['invalid_original_wavs']=len(bad_audio)
        stats['unique_original_wavs']=len(unique)
        stats['distinct_pair_wavs']=len(pairs)
        if out_stream:out_stream.close();out_stream=None
        if skipped_stream:skipped_stream.close();skipped_stream=None
        if temp:os.replace(temp,output);temp=None
        if skipped_tmp:os.replace(skipped_tmp,output.with_suffix('.skipped.jsonl'));skipped_tmp=None
        if not args.check_only and args.report:
            args.report.parent.mkdir(parents=True,exist_ok=True)
            args.report.write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        if args.skip_invalid_audio and not args.check_only:
            print(f'[DONE] output={output} skipped_report={output.with_suffix(".skipped.jsonl")}',file=sys.stderr,flush=True)
        print(json.dumps(stats,ensure_ascii=False,indent=2))
    finally:
        if out_stream:out_stream.close()
        if skipped_stream:skipped_stream.close()
        if temp:temp.unlink(missing_ok=True)
        if skipped_tmp:skipped_tmp.unlink(missing_ok=True)

if __name__=='__main__':
    try:main()
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
        print('ERROR:',exc,file=sys.stderr)
        sys.exit(2)
