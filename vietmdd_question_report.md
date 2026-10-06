# VietMDD QA Release Report

## Release summary

- Dataset: VietMDD
- Active release: `mixed_format_v1`
- Model-facing file: `qa_model_facing.jsonl`
- Total QA rows: **21,927**
- Active QA types: **6**
- MCQ types: **5**
- Open-ended types: **1**
- Total MCQ rows: **18,746**
- Total open-ended rows: **3,181**
- Unique VietMDD audio items: **3,181**

## Question distribution

| # | type_id | Format | Rows |
|---|---|---:|---:|
| 1 | `vietmdd_direct_observed_text` | OPEN_ENDED | 3,181 |
| 2 | `vietmdd_target_match_observed_text` | MCQ | 6,362 |
| 3 | `vietmdd_spoken_content_matches_reference` | MCQ | 3,181 |
| 4 | `vietmdd_equality_observed_text` | MCQ | 3,011 |
| 5 | `vietmdd_selection_observed_text` | MCQ | 3,181 |
| 6 | `vietmdd_composite_transcribe_pair_equality` | STRUCTURED MCQ | 3,011 |
|  | **TOTAL** |  | **21,927** |

## Response-format policy

### Open-ended

Only `vietmdd_direct_observed_text` is open-ended.

The gold answer is the exact observed transcription.

### MCQ

`vietmdd_target_match_observed_text`

- Choices: `["Có", "Không"]`
- Rows: 6,362

`vietmdd_spoken_content_matches_reference`

- Choices: `["Có", "Không"]`
- Rows: 3,181

`vietmdd_equality_observed_text`

- Choices: `["Có", "Không"]`
- Rows: 3,011

`vietmdd_selection_observed_text`

- Choices:
  - `Đoạn âm thanh thứ nhất`
  - `Đoạn âm thanh thứ hai`
- Rows: 3,181

`vietmdd_composite_transcribe_pair_equality`

- Structured MCQ
- 4 complete candidate choices per row
- Each choice represents transcript A, transcript B, and their equality relation
- Rows: 3,011

## Audio mapping

The VietMDD physical audio collection contains **3,181 WAV files**.

Expected server dataset root:

```text
/home/voice/data/voice/datasets/VietMDD
```

Physical audio paths follow:

```text
/home/voice/data/voice/datasets/VietMDD/audio/<numeric_id>.wav
```

The helper script `map_vietmdd_audio_paths.py` converts logical references such as:

```text
vietmdd_audio_1071
```

to:

```text
/home/voice/data/voice/datasets/VietMDD/audio/1071.wav
```

The script validates physical existence, preserves multi-audio ordering, checks one-to-one mapping, and does not overwrite the input JSONL.

## Integrity notes

- Total active rows: **21,927**
- Source/output row identity: **1:1**
- Active semantic type IDs: unchanged
- Historical VietMDD release: preserved
- Final response-format policy: **5 MCQ + 1 open-ended**
