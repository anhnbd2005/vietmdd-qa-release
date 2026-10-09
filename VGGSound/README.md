# VGGSound

**VGGSound** — About 200,000 YouTube clips of 10 s (~550 hours) across 309 sound classes.

| Info | Value |
|---|---|
| **Source** | `Loie/VGGSound` |
| **License** | CC BY 4.0 for annotations; audio/video follow the licence of each YouTube video |
| **Language** | Vietnamese (QA and labels) |
| **Paper** | *VGGSound: A Large-Scale Audio-Visual Dataset* (2020), arXiv:2004.14368 |
| **Raw data** | `/home/voice/data/voice/VGGSound` |
| **Manifests** | `/home/voice/data/voice/datasets/VGGSound/metadata` |
| **Audio** | `/home/voice/data/voice/datasets/VGGSound/audio` (16 kHz mono FLAC) |
| **QA directory** | `/home/voice/data/voice/VietnameseQASLM/VGGSound/FinalQA` |
| **Checked** | 2026-10-07, by `src/finalize.py` |

## Splits

| Split | Manifest output | Audio | Hours | QA |
|---|---|---:|---:|---:|
| `train` | `metadata/trainset.jsonl` | 182,615 | 506.24 | 182,615 |
| `test` | `metadata/testset.jsonl` | 15,341 | 42.40 | 15,341 |
| **Total** | | **197,956** | **548.64** | **197,956** |

## Columns

Source metadata:

`audio_input`, `duration`, `label`, `video_id`, `start_second`, `split`, `id`, `source`.

Final QA:

`id`, `question`, `type_id`, `operator`, `audio_input`, `duration`, `text_output`.

## Published vs. Measured

| | Published by the authors | On this server |
|---|---|---|
| Samples | ~200,000 clips / 309 classes | 197,956 clips |
| Duration | ~550 hours | 548.64 hours |

**Notes:** 197,956 of the 199,467 CSV rows (99.2%) have audio. The `vggsound_08.tar.gz` archive was corrupted at the Hugging Face source; 8,780 clips were recovered, and approximately 291 clips were missing from archive 19. Audio was extracted from mirrored MP4 files and stored as 16 kHz mono FLAC.

## Download & Preprocess

```bash
python3 /home/voice/code/VDT_02/anhhd10/slm_datasets/src/data/vggsound/preprocess.py
python3 /home/voice/code/VDT_02/anhhd10/slm_datasets/src/finalize.py --dataset VGGSound
```

## Exploitable Tasks

| Task | Description | Train QA | Test QA | Total QA |
|---|---|---:|---:|---:|
| **Sound Event Classification** | Identify the main sound event using the Vietnamese `label` (309 classes) | 182,615 | 15,341 | **197,956** |

**Question:** "Sự kiện âm thanh chính được ghi lại trong đoạn âm thanh là gì?"

**Generation:** 1 QA/audio, deterministic label-based generation, no LLM.

### QA Example

```json
{
  "id": "qa_a20df09fbf5056e0315f",
  "question": "Sự kiện âm thanh chính được ghi lại trong đoạn âm thanh là gì?",
  "type_id": "vggsound_event_classification",
  "operator": "DIRECT",
  "audio_input": "/home/voice/data/voice/datasets/VGGSound/audio/train/00/--0PQM4-hqg_000030.flac",
  "duration": 10.008,
  "text_output": "thác nước chảy róc rách"
}
```

## QA Output

**Output directory:** `/home/voice/data/voice/VietnameseQASLM/VGGSound/FinalQA/`

| Split | QA file | Number of QA |
|---|---|---:|
| Train | `FinalQA/train.jsonl` | 182,615 |
| Test | `FinalQA/test.jsonl` | 15,341 |
| **Total** | | **197,956** |

Directory structure:

```text
/home/voice/data/voice/VietnameseQASLM/VGGSound/
├── README.md
└── FinalQA/
    ├── train.jsonl
    └── test.jsonl
```

**Total: 197,956 QA across one task.**