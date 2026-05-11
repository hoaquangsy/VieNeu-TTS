# VieNeu Voice Clone API

API local de cac tool khac goi HTTP vao VieNeu-TTS.

## Chay API

```powershell
cd D:\VieNeu-TTS
.\run_voice_clone_api.bat
```

API mac dinh chay o:

```text
http://127.0.0.1:8002
```

## Endpoint

### 1. Health check

```http
GET /health
```

### 2. Liet ke preset voices

```http
GET /voices?mode=turbo&device=cpu
```

### 3. Clone voice va tra thang file WAV

```http
POST /clone
Content-Type: multipart/form-data
```

Form fields:

```text
text            bat buoc
ref_audio       bat buoc
ref_text        bat buoc neu mode=standard
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
```

### 4. Clone voice va tra JSON duong dan file

```http
POST /clone-json
Content-Type: multipart/form-data
```

Response:

```json
{
  "status": "ok",
  "output_path": "D:\\VieNeu-TTS\\outputs\\api\\clone_xxxxxxxx.wav",
  "mode": "turbo",
  "sample_rate": 24000
}
```

### 5. Synthesize bang preset voice

```http
POST /synthesize
Content-Type: application/json
```

Body:

```json
{
  "text": "Xin chao, day la ban API.",
  "preset_voice": "voice_id",
  "mode": "turbo",
  "device": "cpu"
}
```

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
