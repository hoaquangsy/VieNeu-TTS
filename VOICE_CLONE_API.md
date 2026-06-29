# VieNeu Voice Clone API

API local de cac tool khac goi HTTP vao VieNeu-TTS.

## AutoVid contract - 2026-06-07

AutoVid production TTS now calls VieNeu v3 Turbo GPU only.

AutoVid defaults:

```json
{
  "model": "vieneu-v3-turbo-gpu",
  "mode": "v3_turbo_gpu",
  "device": "cuda",
  "timeline_mode": "full_then_align",
  "voice": "chill",
  "voice_type": "ref_audio",
  "ref_audio": "examples/audio_ref/example_ngoc_huyen.wav"
}
```

The old v2 preset `Nguyễn Huyền Trang` is not a v3 voice. AutoVid maps legacy labels to the v3 ref-audio preset `chill` and sends `preset_voice=null` with `ref_audio=examples/audio_ref/example_ngoc_huyen.wav`.

If v3 fails, AutoVid fails fast. It does not fallback to `vieneu-v2-cpu`.

VieNeu API may still expose v2 models for separate manual tests, but AutoVid should not call them.

## Chay API

```powershell
cd D:\VieNeu-TTS
.\run_voice_clone_api.bat
```

API mac dinh chay o:

```text
http://127.0.0.1:8002
```

Khi chay truc tiep trong workspace, uu tien:

```powershell
.\.venv\Scripts\python.exe apps\voice_clone_api.py
```

## Endpoint

### 1. Health check

```http
GET /health
```

### 2. Liet ke model API ho tro

```http
GET /models
```

Model hien co:

```text
vieneu-v2-cpu        VieNeu-TTS-v2 standard CPU, mac dinh cho AutoVid aligned TTS
vieneu-v2-gpu        VieNeu-TTS-v2 standard GPU, thu nghiem chat luong cao neu co CUDA/VRAM du
vieneu-v2-turbo-cpu  VieNeu-TTS-v2-Turbo CPU
```

### 3. Liet ke preset voices

```http
GET /voices?model=vieneu-v2-cpu
```

Co the truyen `mode`, `device`, `backbone_device`, `codec_device` neu can debug:

```http
GET /voices?mode=turbo&device=cpu
```

### 4. Clone voice va tra thang file WAV

```http
POST /clone
Content-Type: multipart/form-data
```

Form fields:

```text
text            bat buoc
ref_audio       bat buoc
ref_text        bat buoc neu mode=standard
model           vieneu-v2-cpu | vieneu-v2-turbo-cpu
mode            turbo | turbo_gpu | standard
device          mac dinh cpu
backbone_device mac dinh cpu
codec_device    mac dinh cpu
emotion         natural | storytelling
temperature
top_k
max_chars
max_tokens
apply_watermark true | false
hf_token
```

### 5. Clone voice va tra JSON duong dan file

```http
POST /clone-json
Content-Type: multipart/form-data
```

Response:

```json
{
  "status": "ok",
  "output_path": "D:\\VieNeu-TTS\\outputs\\api\\clone_xxxxxxxx.wav",
  "model": "vieneu-v2-cpu",
  "mode": "standard",
  "sample_rate": 24000
}
```

### 6. Synthesize bang preset voice

```http
POST /synthesize
Content-Type: application/json
```

Body:

```json
{
  "text": "Xin chao, day la ban API.",
  "preset_voice": "voice_id",
  "model": "vieneu-v2-cpu",
  "mode": "standard",
  "device": "cpu"
}
```

### 7. Synthesize full narration va tra timeline aligned

Dung cho AutoVid hoac cac tool can tao mot file WAV dai, roi can moc thoi gian theo scene/segment.

```http
POST /synthesize-aligned
Content-Type: application/json
```

Body toi thieu:

```json
{
  "request_id": "autovid_20260605154345_2fcf3c6080bf",
  "preset_voice": "Ly",
  "model": "vieneu-v2-cpu",
  "mode": "standard",
  "timeline_mode": "full_then_align",
  "alignment_method": "whisper",
  "speed": 1.0,
  "emotion": "storytelling",
  "segments": [
    { "id": "intro", "text": "Cau mo dau." },
    { "id": "chapter-1-scene-1", "text": "Noi dung can thuyet minh." }
  ]
}
```

Gia tri quan trong:

```text
timeline_mode        full_then_align | full_then_align_verified
alignment_method     whisper | faster_whisper | openai_whisper
max_full_align_words mac dinh 1300
max_chars            mac dinh 256, la upper bound cho sentence-aware chunking
max_tokens           chi dung cho turbo
```

`full_then_align_verified` khong con la duong chay chinh. Neu client cu gui gia tri nay, API map ve `full_then_align` va warning:

```text
verified mode disabled; using full_then_align
```

Trong `/synthesize-aligned`, API se:

```text
1. Ghep tat ca segments thanh mot full narration.
2. Tach sentence-aware chunks de TTS on dinh hon.
3. Tao cac chunk audio bang dung text goc, roi join thanh mot WAV duy nhat.
4. Chay Whisper local chi de lay word timestamps.
5. Tra timeline segment va ghi artifacts ra outputs\api.
```

Artifacts duoc ghi theo `request_id`:

```text
outputs\api\full_<request_id>.wav
outputs\api\full_<request_id>.txt
outputs\api\full_<request_id>.timings.json
```

Response chinh:

```json
{
  "status": "ok",
  "request_id": "autovid_...",
  "audio_path": "D:\\VieNeu-TTS\\outputs\\api\\full_autovid_....wav",
  "text_path": "D:\\VieNeu-TTS\\outputs\\api\\full_autovid_....txt",
  "timings_path": "D:\\VieNeu-TTS\\outputs\\api\\full_autovid_....timings.json",
  "duration": 319.39,
  "model": "vieneu-v2-cpu",
  "mode": "standard",
  "timeline_mode": "full_then_align",
  "requested_timeline_mode": "full_then_align",
  "alignment_method": "whisper",
  "fallback_used": false,
  "segments": [
    {
      "id": "intro",
      "start": 0,
      "end": 36.62,
      "duration": 36.62,
      "alignment_confidence": 0.84
    }
  ],
  "warnings": [],
  "chunkingStrategy": "sentence-aware",
  "verificationDisabled": true,
  "patching": { "enabled": false }
}
```

Response va file `.timings.json` co them block `performance` de audit bottleneck:

```json
{
  "performance": {
    "requestId": "autovid_...",
    "mode": "standard",
    "model": "vieneu-v2-cpu",
    "timelineMode": "full_then_align",
    "segmentCount": 20,
    "chunkCount": 49,
    "audioDurationSeconds": 309.83,
    "totalDurationMs": 301932,
    "requestParseValidationMs": 1,
    "modelLoadVoiceMs": 8580,
    "preprocessMs": 0,
    "chunkingMs": 1,
    "ttsSynthMs": 211851,
    "concatExportMs": 398,
    "whisperAlignMs": 81099,
    "artifactWriteMs": 2,
    "cpuTuning": {
      "cpuCount": 24,
      "ttsThreads": 8,
      "torchNumThreads": 8,
      "onnxIntraOpThreads": 8,
      "ggufThreads": 8,
      "cpuAvgPercent": 850.62,
      "cpuPeakPercent": 1344.5,
      "memoryPeakMb": 1496.18
    },
    "chunks": []
  }
}
```

Env profiling/tuning:

```text
VIE_TTS_CHUNK_CONCURRENCY=1       mac dinh, an toan cho standard GGUF
VIE_TTS_CPU_THREADS=8             tuy chon; truyen vao GGUF, ONNX codec va torch CPU threads
VIE_TTS_ALIGNMENT_MODEL=base      mac dinh neu khong set VIENEU_WHISPER_MODEL
VIE_TTS_ALIGNMENT_DEVICE=cpu      cpu | cuda
VIE_TTS_ALIGNMENT_COMPUTE_TYPE=int8 | float16
VIE_TTS_ALIGNMENT_CPU_THREADS=8   tuy chon cho faster-whisper CPU
```

Voi `mode=standard`, API tu khoa chunk concurrency ve `1` vi backend GGUF khong thread-safe.

Neu request bi timeout o client nhung API van dang chay, dung `request_id` de kiem tra file `.timings.json` trong `outputs\api`.

## Checkpoint hien tai cho AutoVid aligned TTS

### Trang thai on dinh

`/synthesize-aligned` dang dung mode on dinh:

```text
timeline_mode=full_then_align
mode=standard
model=vieneu-v2-cpu
```

API nhan nhieu `segments` tu AutoVid, ghep thanh full narration text, synth thanh mot full WAV, chay Whisper alignment de tra timing tung segment.

Output cuoi cung van la mot bo artifact duy nhat:

```text
D:\VieNeu-TTS\outputs\api\full_<request_id>.wav
D:\VieNeu-TTS\outputs\api\full_<request_id>.txt
D:\VieNeu-TTS\outputs\api\full_<request_id>.timings.json
```

Khong dung:

```text
full_then_align_verified lam duong chay chinh
pronunciation verifier de fail job
patch audio
alias/rewrite text
scene-level output
chunk_align toan bo
```

### Request ID deterministic artifacts

Voice API ho tro `request_id` tu AutoVid. Artifact duoc tao deterministic theo request id:

```text
D:\VieNeu-TTS\outputs\api\full_<request_id>.wav
D:\VieNeu-TTS\outputs\api\full_<request_id>.txt
D:\VieNeu-TTS\outputs\api\full_<request_id>.timings.json
```

Muc tieu:

```text
AutoVid khong recover bang latest artifact.
AutoVid chi recover/copy dung artifact theo request_id.
Tranh copy nham audio tu request khac.
```

### Sentence-aware chunking

`/synthesize-aligned` trong `full_then_align` da dung sentence-aware chunking:

```text
Khong doi script/text.
Khong alias/rewrite.
.txt khop full narration text nhan tu request.
Chunk theo cau.
Cau risky giu rieng chunk.
Cau safe pack toi da 2 cau/chunk.
Khong gom bua cau risky voi cau sau.
max_chars chi la upper bound, khong phai ly do de gom cau sau.
maxCharsStrategy=sentence-length-adaptive
chunkingStrategy=sentence-aware
textChangedForTts=false
pronunciationAliasApplied=false
```

Test Voice 20 segments:

```text
txt_equal_request=true
chunkingStrategy=sentence-aware
maxCharsStrategy=sentence-length-adaptive
maxCharsUsed=256
longestSentenceChars=283
chunkCount=56
riskySentenceSingleChunkCount=41
maxSentencesPerChunk=2
```

Output test:

```text
D:\VieNeu-TTS\outputs\api\full_audit_sentence_chunking_autovid20_final_1780728505.wav
D:\VieNeu-TTS\outputs\api\full_audit_sentence_chunking_autovid20_final_1780728505.txt
D:\VieNeu-TTS\outputs\api\full_audit_sentence_chunking_autovid20_final_1780728505.timings.json
```

### Performance profiling

`/synthesize-aligned` da ghi performance profiling vao response va `.timings.json`.

Baseline replay tu AutoVid:

```text
request_id=autovid_20260606154052_6d1e4cf01307_profile_c1
segments=20
chunks=49
audioDurationSeconds=309.83
```

Performance baseline:

```json
{
  "concurrency": 1,
  "segmentCount": 20,
  "chunkCount": 49,
  "audioDurationSeconds": 309.83,
  "totalDurationMs": 301932,
  "modelLoadVoiceMs": 8580,
  "preprocessMs": 0,
  "chunkingMs": 1,
  "ttsSynthMs": 211851,
  "concatExportMs": 398,
  "whisperAlignMs": 81099,
  "artifactWriteMs": 2
}
```

Bottleneck:

```text
Chinh: TTS synth chunks, khoang 211.9s, khoang 70% request.
Phu: Whisper alignment, khoang 81.1s.
Khong phai concat/export: khoang 0.4s.
Khong phai chunking/preprocess.
```

Chunk lau nhat:

```text
chunk 1: 11968ms, charCount=313, risk=high
preview: "No khong den bang tieng coi, ma bang nhung viec rat nho..."
```

### Chunk concurrency decision

Da thu:

```text
VIE_TTS_CHUNK_CONCURRENCY=2
```

Voi standard GGUF, backend crash:

```text
GGML_ASSERT(i1 >= 0 && i1 < ne1) failed
```

Quyet dinh:

```text
Khong bat chunk concurrency > 1 cho standard GGUF.
VIE_TTS_CHUNK_CONCURRENCY=1 la default an toan.
Neu user set concurrency > 1 trong standard mode, API force ve 1.
Warning key: standard_gguf_concurrency_forced_to_1
```

Khong dung chunk concurrency de tang toc vi co rui ro crash, rui ro mat tu nhien giua chunks, va output can giu full narration on dinh.

### CPU/thread tuning

Da audit CPU/thread utilization nhung van giu chunk concurrency = 1.

Files lien quan:

```text
apps/voice_clone_api.py: CPU/memory sampler, cpuTuning, env thread config, CPU log
src/vieneu/standard.py: truyen cpu_threads vao llama.cpp n_threads / n_threads_batch
src/vieneu/utils.py: set ONNX intra_op_num_threads / inter_op_num_threads
VOICE_CLONE_API.md: document cpuTuning va env moi
```

Benchmark cung request, 20 segments, 49 chunks:

```json
[
  {
    "threads": "current",
    "ttsSynthMs": 200557,
    "whisperAlignMs": 45620,
    "totalDurationMs": 251317,
    "cpuAvgPercent": 1224.37,
    "cpuPeakPercent": 2056.2
  },
  {
    "threads": 4,
    "ttsSynthMs": 217579,
    "whisperAlignMs": 41689,
    "totalDurationMs": 264548,
    "cpuAvgPercent": 417.25,
    "cpuPeakPercent": 635.9
  },
  {
    "threads": 8,
    "ttsSynthMs": 201646,
    "whisperAlignMs": 37641,
    "totalDurationMs": 244278,
    "cpuAvgPercent": 850.62,
    "cpuPeakPercent": 1344.5
  },
  {
    "threads": 12,
    "ttsSynthMs": 203269,
    "whisperAlignMs": 42037,
    "totalDurationMs": 250266,
    "cpuAvgPercent": 1256.0,
    "cpuPeakPercent": 1853.8
  }
]
```

Fastest full run tren may nay:

```text
VIE_TTS_CPU_THREADS=8
VIE_TTS_ALIGNMENT_CPU_THREADS=8
VIE_TTS_ALIGNMENT_MODEL=base
VIE_TTS_ALIGNMENT_COMPUTE_TYPE=int8
VIE_TTS_CHUNK_CONCURRENCY=1
```

Result:

```text
total khoang 244.3s
baseline khoang 251.3s
cai thien nhe khoang 7s
```

Conclusion:

```text
CPU khong underutilized nang nhu tuong.
Current process avg CPU khoang 1224%, tuc khoang 12.2 logical cores tren may 24 logical cores.
Peak khoang 20.5 cores.
Thread tuning chi cai thien nhe, khong dang tiep tuc toi uu sau.
Khong hard-code default trong code; chi cho phep set bang env neu muon test.
```

### Whisper alignment tuning

Whisper alignment la bottleneck phu. Khong tat Whisper alignment vi AutoVid can segment timing chinh xac.

Ket qua test:

```json
[
  {
    "model": "base",
    "threads": 4,
    "whisperAlignMs": 52234
  },
  {
    "model": "base",
    "threads": 8,
    "whisperAlignMs": 46316
  },
  {
    "model": "base",
    "threads": 12,
    "whisperAlignMs": 43359
  },
  {
    "model": "small",
    "threads": 4,
    "whisperAlignMs": 111715
  },
  {
    "model": "small",
    "threads": 8,
    "whisperAlignMs": 104958
  },
  {
    "model": "small",
    "threads": 12,
    "whisperAlignMs": 102711
  }
]
```

Conclusion:

```text
base/int8 nhanh hon small/int8 rat nhieu.
Khong nen dung small neu muc tieu la toc do.
```

Config khuyen nghi neu can set env:

```text
VIE_TTS_ALIGNMENT_MODEL=base
VIE_TTS_ALIGNMENT_COMPUTE_TYPE=int8
VIE_TTS_ALIGNMENT_CPU_THREADS=8
VIE_TTS_ALIGNMENT_DEVICE=cpu neu khong co CUDA
```

### Env/config hien tai

Env moi / lien quan:

```text
VIE_TTS_CHUNK_CONCURRENCY=1
VIE_TTS_CPU_THREADS=8
VIE_TTS_ALIGNMENT_CPU_THREADS=8
VIE_TTS_ALIGNMENT_MODEL=base
VIE_TTS_ALIGNMENT_DEVICE=cpu|cuda
VIE_TTS_ALIGNMENT_COMPUTE_TYPE=int8|float16
```

Current recommendation:

```text
VIE_TTS_CHUNK_CONCURRENCY=1
VIE_TTS_ALIGNMENT_MODEL=base
VIE_TTS_ALIGNMENT_COMPUTE_TYPE=int8
```

Optional machine-specific tuning:

```text
VIE_TTS_CPU_THREADS=8
VIE_TTS_ALIGNMENT_CPU_THREADS=8
```

Luu y:

```text
Thread tuning cai thien it.
Khong tiep tuc toi uu performance luc nay.
Uu tien on dinh chat luong.
```

### Profiling report fields

Timings JSON co the ghi performance block:

```json
{
  "performance": {
    "requestId": "...",
    "mode": "standard",
    "model": "vieneu-v2-cpu",
    "timelineMode": "full_then_align",
    "segmentCount": 20,
    "chunkCount": 49,
    "audioDurationSeconds": 309.83,
    "totalDurationMs": 301932,
    "preprocessMs": 0,
    "chunkingMs": 1,
    "ttsSynthMs": 211851,
    "concatExportMs": 398,
    "whisperAlignMs": 81099,
    "artifactWriteMs": 2,
    "chunks": [],
    "cpuTuning": {
      "cpuCount": 0,
      "ttsThreads": 0,
      "alignmentCpuThreads": 0,
      "cpuAvgPercent": 0,
      "cpuPeakPercent": 0,
      "memoryPeakMb": 0
    }
  }
}
```

### Quality checks from benchmark

Baseline quality checks:

```text
.txt khop full narration tu response segments: true
Whisper fallback: false
WAV duration: 309.83s
Peak: 0.9343
Clipping ratio: 0.0
Max low-RMS run: 0.9s
```

Khong thay clipping/silence dai bat thuong trong kiem tu dong. Human listening van la approval gate cuoi.

Artifacts nhanh nhat:

```text
D:\VieNeu-TTS\outputs\api\full_autovid_20260606154052_6d1e4cf01307_threads_8.wav
D:\VieNeu-TTS\outputs\api\full_autovid_20260606154052_6d1e4cf01307_threads_8.timings.json
```

Baseline profiling artifacts:

```text
D:\VieNeu-TTS\outputs\api\full_autovid_20260606154052_6d1e4cf01307_profile_c1.wav
D:\VieNeu-TTS\outputs\api\full_autovid_20260606154052_6d1e4cf01307_profile_c1.txt
D:\VieNeu-TTS\outputs\api\full_autovid_20260606154052_6d1e4cf01307_profile_c1.timings.json
```

### Checks gan nhat

```text
py_compile apps/voice_clone_api.py: pass
pytest tests/test_utils.py -q: 11 passed
git diff --check: clean except Git CRLF warnings
```

### Current decision

```text
Khong toi uu performance Voice them luc nay.
Khong bat chunk concurrency > 1.
Khong hard-code thread defaults.
Giu on dinh: standard, full_then_align, sentence-aware chunking, chunk concurrency 1, Whisper alignment base/int8.
AutoVid da chay Voice song song voi visual, nen workflow da toi uu du.
```

Neu can nhanh hon ro ret trong tuong lai, can can nhac TTS engine khac, GPU backend, cache chunk/audio, hoac chap nhan trade-off chat luong/turbo.

## Vi du curl

### Clone voice tra file WAV

```powershell
curl.exe -X POST "http://127.0.0.1:8002/clone" ^
  -F "text=Xin chao, day la ban clone tu API." ^
  -F "mode=turbo" ^
  -F "device=cpu" ^
  -F "ref_audio=@D:\VieNeu-TTS\examples\audio_ref\example.wav" ^
  --output D:\VieNeu-TTS\outputs\api_result.wav
```

### Clone voice tra JSON

```powershell
curl.exe -X POST "http://127.0.0.1:8002/clone-json" ^
  -F "text=Xin chao, day la ban clone tu API." ^
  -F "mode=turbo" ^
  -F "device=cpu" ^
  -F "ref_audio=@D:\VieNeu-TTS\examples\audio_ref\example.wav"
```

### AutoVid aligned TTS smoke test

```powershell
$body = @{
  request_id = "smoke_aligned"
  preset_voice = "Ly"
  model = "vieneu-v2-cpu"
  mode = "standard"
  timeline_mode = "full_then_align"
  alignment_method = "whisper"
  segments = @(
    @{ id = "intro"; text = "Xin chao, day la doan mo dau." },
    @{ id = "scene-1"; text = "Day la doan thu hai de kiem tra moc thoi gian." }
  )
} | ConvertTo-Json -Depth 6

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8002/synthesize-aligned" `
  -ContentType "application/json; charset=utf-8" `
  -Body $body
```

## Kiem tra tien do khi AutoVid goi sang

```powershell
# API con song khong
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8002/health

# Co request dang noi vao API khong
Get-NetTCPConnection -LocalPort 8002 -ErrorAction SilentlyContinue

# File moi nhat ma Voice API da tao
Get-ChildItem D:\VieNeu-TTS\outputs\api -File |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 10 Name,LastWriteTime,Length
```

Neu chi thay port `8002` o trang thai `Listen`, khong co connection `Established`, va `outputs\api` khong co file moi, thi AutoVid chua vao buoc TTS hoac da qua buoc TTS roi.
