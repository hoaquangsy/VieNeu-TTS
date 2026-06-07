"""
VieNeu-TTS v3 Turbo — inference engine (PyTorch).
=================================================
Synthesizes 48 kHz Vietnamese speech with instant voice cloning from a short
reference clip. This is an inference engine: it loads a checkpoint plus the MOSS
audio codec and turns text into a waveform.

Quick start:
    from vieneu._v3_turbo_engine import VieNeuTTSv3Turbo
    tts = VieNeuTTSv3Turbo(checkpoint_path="pnnbao-ump/VieNeu-TTS-v3-Turbo")
    ref = tts._encode_ref("reference_voice.wav")      # encode the voice once
    wav = tts.infer("Xin chào Việt Nam", ref_codes=ref)   # -> float32 @ 48 kHz
    # or stream it chunk by chunk:
    for chunk in tts.infer_stream("...", ref_codes=ref):
        play(chunk)

Credits
-------
- Architecture: VieNeu-TTS v3 Turbo is designed and trained from scratch on
  ~10,000 hours of English–Vietnamese speech by Phạm Nguyễn Ngọc Bảo
  — https://github.com/pnnbao97
- Phonemizer: sea-g2p — https://github.com/pnnbao97/sea-g2p
- Audio codec: MOSS-Audio-Tokenizer-Nano (OpenMOSS-Team).
"""
from __future__ import annotations
import re
import threading
import time
from pathlib import Path
from typing import Generator, List, Optional, Union
import numpy as np
import torch
_STREAM_LEADIN_FRAMES = 4
from .configuration_v3_turbo import VieNeuV3TurboConfig
from .hub_load_v3_turbo import load_v3_turbo_checkpoint
from .modeling_v3_turbo import VieNeuV3TurboForTTS, _sample_token

# ── Inline non-verbal cues (emotion tokens) ───────────────────────────────────
# The emotion checkpoint embeds three non-verbal cues directly in the phoneme
# stream as special tokens. In the text they appear as bracketed tags; when
# phonemizing we leave them as the matching <|emotion_k|> token rather than
# spelling the bracketed words out. Spacing matches the training data exactly.
#   [chuckle]/[cười] -> <|emotion_1|>, [sigh]/[thở dài] -> <|emotion_2|>,
#   [clear throat]/[hắng giọng] -> <|emotion_3|>
_EMOTION_TAG_TO_K = {
    'chuckle': 1, 'cười': 1, 'cuoi': 1,
    'sigh': 2, 'thở dài': 2, 'tho dai': 2,
    'clear throat': 3, 'hắng giọng': 3, 'hang giong': 3,
}
_EMOTION_SPLIT_RE = re.compile(r'(\[[^\]]+\]|<\|emotion_\d+\|>)')
_ATTACHING_PUNCT = set('.,!?;:…)]}"\'’”')


def _emotion_tag_token(tag: str) -> Optional[str]:
    t = tag.strip()
    if t.startswith('<|'):
        return t
    inner = t[1:-1].strip().lower()
    k = _EMOTION_TAG_TO_K.get(inner)
    return f'<|emotion_{k}|>' if k is not None else None


def _phonemize_with_emotions(text: str, pipeline) -> str:
    """Phonemize ``text`` with ``pipeline.run`` while keeping inline cues as
    ``<|emotion_k|>`` tokens (see module note)."""
    if '[' not in text and '<|emotion_' not in text:
        return pipeline.run(text)
    out = ''
    for i, part in enumerate(_EMOTION_SPLIT_RE.split(text)):
        token = _emotion_tag_token(part) if i % 2 == 1 else None
        if token is not None:
            out = (out + ' ' + token) if out else token
            continue
        ph = pipeline.run(part) if part and part.strip() else ''
        if not ph:
            continue
        if not out:
            out = ph
        elif ph[0] in _ATTACHING_PUNCT:
            out += ph
        else:
            out += ' ' + ph
    return out

class VieNeuTTSv3Turbo:
    """High-level text-to-speech interface (48 kHz, instant voice cloning).

    Per utterance the engine: encodes the reference voice -> builds the prompt ->
    generates audio tokens frame by frame -> decodes them with the MOSS codec ->
    waveform. Use :meth:`infer` for one-shot synthesis or :meth:`infer_stream`
    to receive audio in chunks. Calls are thread-safe (guarded by an internal lock).
    """

    SAMPLE_RATE = 48000

    def __init__(self, checkpoint_path: str='pnnbao-ump/VieNeu-TTS-v3-Turbo', tokenizer_path: Optional[str]=None, moss_tokenizer_path: str='OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano', device: str='auto', dtype: str='auto', compile_acoustic: bool=False):
        """Load the v3 Turbo checkpoint and the MOSS audio codec.

        Args:
            checkpoint_path: HF repo id or local directory of the v3 Turbo model.
            tokenizer_path: text tokenizer (defaults to ``checkpoint_path``).
            moss_tokenizer_path: MOSS-Audio-Tokenizer-Nano, used to encode the
                reference voice and decode generated tokens back to audio.
            device: ``"auto"`` | ``"cpu"`` | ``"cuda"``.
            dtype: ``"auto"`` | ``"float32"`` | ``"bfloat16"`` | ``"float16"``.
        """
        self._lock = threading.RLock()
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)
        from transformers import AutoTokenizer
        tok_path = tokenizer_path or checkpoint_path
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
        self.config = VieNeuV3TurboConfig.from_pretrained(checkpoint_path)
        self.model = load_v3_turbo_checkpoint(checkpoint_path, device=self.device, dtype=self.dtype).eval()
        from transformers import AutoModel
        self.audio_tokenizer = AutoModel.from_pretrained(moss_tokenizer_path, trust_remote_code=True).to(self.device).eval()
        self.default_emotion = '<|emotion_0|>'
        if compile_acoustic:
            # Compile the acoustic decoder's cached step (the inner-loop hot path).
            # Needs a C compiler (MSVC `cl` on Windows / gcc on Linux). Warm it up now
            # so we can fall back to eager if the toolchain is missing.
            dec = self.model.acoustic_decoder
            _orig_step = dec.cached_step
            try:
                dec.cached_step = torch.compile(_orig_step, dynamic=True)
                _H = self.config.hidden_size
                _L = len(dec.layers)
                _dt = next(dec.parameters()).dtype
                with torch.no_grad():
                    _x = torch.zeros(1, 2, _H, dtype=_dt, device=self.device)
                    dec.cached_step(_x, torch.tensor([0, 1], device=self.device), [None] * _L, [None] * _L)
            except Exception as e:  # noqa: BLE001
                dec.cached_step = _orig_step
                print(f"[v3turbo] torch.compile unavailable, using eager "
                      f"({type(e).__name__}). Install a C compiler to enable it.")

    def _leading_token_id(self, emotion: str, voice_token_id: Optional[int]) -> int:
        """Resolve the leading text-block token.

        For built-in default voices the emotion checkpoint expects that speaker's
        reserved token (``voice_token_id``, ids 13..42). When cloning a voice
        (``voice_token_id is None``) we fall back to the emotion tag, matching the
        original v3 Turbo behaviour.
        """
        if voice_token_id is not None:
            return int(voice_token_id)
        return self.config.emotion_0_token_id if emotion == 'natural' else self.config.emotion_4_token_id

    def infer(self, text: str, ref_audio: Optional[str]=None, ref_codes: Optional[np.ndarray]=None, ref_text: Optional[str]=None, phonemes: Optional[str]=None, ref_phonemes: Optional[str]=None, emotion: str='natural', voice_token_id: Optional[int]=None, temperature: float=0.8, top_k: int=25, top_p: float=0.95, max_new_frames: int=300, repetition_penalty: float=1.2) -> np.ndarray:
        """Synthesize one utterance into a float32, 48 kHz mono waveform.

        Args:
            text: the sentence to read. Ignored if ``phonemes`` is supplied.
            ref_audio: path to a reference wav to clone the voice from.
            ref_codes: pre-encoded reference voice (see :meth:`_encode_ref`);
                reuse this across calls to avoid re-encoding the same voice.
            phonemes: SEA-G2P phoneme string, if you phonemized the text yourself.
            emotion: ``"natural"`` or ``"storytelling"``.
            voice_token_id: a speaker reserved token id (13..42) for a built-in
                default voice. When set it overrides ``emotion`` as the leading
                token. Leave ``None`` for voice cloning (uses the emotion tag).
            temperature, top_k, top_p, repetition_penalty: sampling controls.
            max_new_frames: hard cap on generated frames (1 frame = 80 ms @ 12.5 Hz).

        Returns:
            ``np.ndarray`` (float32) at 48 kHz.
        """
        codes = self._generate_codes(text, ref_audio, ref_codes, ref_text, emotion, temperature, top_k, top_p, max_new_frames, phonemes=phonemes, ref_phonemes=ref_phonemes, repetition_penalty=repetition_penalty, voice_token_id=voice_token_id)
        return self._decode_codes(codes)

    def infer_stream(self, text: str, ref_audio: Optional[str]=None, ref_codes: Optional[np.ndarray]=None, ref_text: Optional[str]=None, emotion: str='natural', voice_token_id: Optional[int]=None, temperature: float=0.8, top_k: int=25, top_p: float=0.95, max_new_frames: int=300, chunk_frames: int=25, repetition_penalty: float=1.2) -> Generator[np.ndarray, None, None]:
        """Like :meth:`infer` but yields the waveform in chunks for low latency.

        ``chunk_frames`` is how many frames to accumulate before decoding and
        emitting one chunk (smaller = lower latency, more codec calls). Yields
        consecutive float32 48 kHz arrays that concatenate into the full clip.
        ``voice_token_id`` selects a built-in default voice (see :meth:`infer`).
        """
        if ref_codes is None and ref_audio is not None:
            ref_codes = self._encode_ref(ref_audio)
        emotion_id = self._leading_token_id(emotion, voice_token_id)
        prompt_2d = self._build_prompt_2d(text, ref_codes, ref_text, emotion_id)
        with self._lock:
            yield from self._stream_generate(prompt_2d, temperature, top_k, top_p, max_new_frames, chunk_frames, repetition_penalty=repetition_penalty)

    def infer_batch(self, texts: List[str], **kwargs) -> List[np.ndarray]:
        """Synthesize several utterances sequentially (one waveform per text).
        Accepts the same keyword arguments as :meth:`infer`.
        """
        return [self.infer(t, **kwargs) for t in texts]

    @staticmethod
    def _prepare_gen_slot_row(slot_row: torch.Tensor, frame_codes: Optional[torch.Tensor], sgs_id: int, audio_pad: int) -> None:
        slot_row[:, :, 0] = sgs_id
        if frame_codes is None:
            slot_row[:, :, 1:] = audio_pad
        else:
            slot_row[:, 0, 1:] = frame_codes.to(slot_row.device)

    @torch.no_grad()
    def _generate_codes(self, text: str, ref_audio, ref_codes, ref_text, emotion, temperature, top_k, top_p, max_new_frames, phonemes: Optional[str]=None, ref_phonemes: Optional[str]=None, repetition_penalty: float=1.2, voice_token_id: Optional[int]=None) -> torch.LongTensor:
        emotion_id = self._leading_token_id(emotion, voice_token_id)
        if ref_codes is None and ref_audio is not None:
            ref_codes = self._encode_ref(ref_audio)
        prompt_2d = self._build_prompt_2d(text, ref_codes, ref_text, emotion_id, phonemes=phonemes, ref_phonemes=ref_phonemes)
        input_2d = prompt_2d.unsqueeze(0).to(self.device)
        prefill_embeds = self.model._build_inputs_embeds(input_2d)
        prefill_out = self.model.semantic_backbone(inputs_embeds=prefill_embeds, use_cache=True, return_dict=True)
        past_kv = prefill_out.past_key_values
        h = prefill_out.last_hidden_state[:, -1]
        all_codes: List[torch.LongTensor] = []
        eos_id = self.config.speech_generation_end_token_id
        sgs_id = self.config.speech_generation_start_token_id
        n_vq = self.config.n_vq
        audio_pad = self.config.audio_pad_token_id
        hist = [set() for _ in range(n_vq)] if repetition_penalty != 1.0 else None
        for _ in range(max_new_frames):
            frame_codes, last_local_out = self.model.decode_one_frame(h, text_token_id=torch.tensor([sgs_id], device=self.device), temperature=temperature, top_k=top_k, audio_top_p=top_p, repetition_penalty=repetition_penalty, history_by_channel=hist)
            all_codes.append(frame_codes.cpu())
            text_logits = self.model.text_lm_head(last_local_out[0, 0]).float()
            next_text = int(text_logits.argmax().item())
            if next_text == eos_id:
                break
            slot_row = torch.full((1, 1, n_vq + 1), audio_pad, dtype=torch.long, device=self.device)
            self._prepare_gen_slot_row(slot_row, frame_codes=frame_codes, sgs_id=sgs_id, audio_pad=audio_pad)
            slot_embed = self.model._build_inputs_embeds(slot_row)
            step_out = self.model.semantic_backbone(inputs_embeds=slot_embed, past_key_values=past_kv, use_cache=True, return_dict=True)
            past_kv = step_out.past_key_values
            h = step_out.last_hidden_state[:, 0]
        if not all_codes:
            return torch.zeros(0, self.config.n_vq, dtype=torch.long)
        return torch.stack(all_codes)

    @torch.no_grad()
    def _stream_generate(self, prompt_2d, temperature, top_k, top_p, max_new_frames, chunk_frames, repetition_penalty: float=1.2) -> Generator[np.ndarray, None, None]:
        input_2d = prompt_2d.unsqueeze(0).to(self.device)
        prefill_embeds = self.model._build_inputs_embeds(input_2d)
        prefill_out = self.model.semantic_backbone(inputs_embeds=prefill_embeds, use_cache=True, return_dict=True)
        past_kv = prefill_out.past_key_values
        h = prefill_out.last_hidden_state[:, -1]
        eos_id = self.config.speech_generation_end_token_id
        sgs_id = self.config.speech_generation_start_token_id
        n_vq = self.config.n_vq
        audio_pad = self.config.audio_pad_token_id
        buffer: List[torch.LongTensor] = []
        hist = [set() for _ in range(n_vq)] if repetition_penalty != 1.0 else None
        sr = self.SAMPLE_RATE
        first_decode = True
        emitted_samples = 0
        t_first: Optional[float] = None

        def _target_frames() -> int:
            cap = max(1, chunk_frames)
            if t_first is None:
                return min(cap, _STREAM_LEADIN_FRAMES)
            lead = emitted_samples / sr - (time.perf_counter() - t_first)
            if lead < 0.2:
                return min(cap, _STREAM_LEADIN_FRAMES)
            if lead < 0.55:
                return min(cap, 6)
            if lead < 1.1:
                return min(cap, 8)
            return cap
        try:
            for _ in range(max_new_frames):
                frame_codes, last_local_out = self.model.decode_one_frame(h, text_token_id=torch.tensor([sgs_id], device=self.device), temperature=temperature, top_k=top_k, audio_top_p=top_p, repetition_penalty=repetition_penalty, history_by_channel=hist)
                buffer.append(frame_codes.cpu())
                text_logits = self.model.text_lm_head(last_local_out[0, 0]).float()
                next_text = int(text_logits.argmax().item())
                if next_text == eos_id:
                    break
                slot_row = torch.full((1, 1, n_vq + 1), audio_pad, dtype=torch.long, device=self.device)
                self._prepare_gen_slot_row(slot_row, frame_codes=frame_codes, sgs_id=sgs_id, audio_pad=audio_pad)
                slot_embed = self.model._build_inputs_embeds(slot_row)
                step_out = self.model.semantic_backbone(inputs_embeds=slot_embed, past_key_values=past_kv, use_cache=True, return_dict=True)
                past_kv = step_out.past_key_values
                h = step_out.last_hidden_state[:, 0]
                if len(buffer) >= _target_frames():
                    wav = self._decode_codes_stream(torch.stack(buffer), reset=first_decode)
                    first_decode = False
                    if t_first is None and len(wav):
                        t_first = time.perf_counter()
                    emitted_samples += len(wav)
                    buffer = []
                    yield wav
            if buffer:
                wav = self._decode_codes_stream(torch.stack(buffer), reset=first_decode)
                first_decode = False
                yield wav
        finally:
            if not first_decode:
                self._reset_stream_session()

    @staticmethod
    def _moss_codes_to_Tnq(audio_codes: torch.Tensor, n_vq: int) -> torch.Tensor:
        if audio_codes.ndim != 3:
            raise ValueError(f'audio_codes must be 3D, got {tuple(audio_codes.shape)}')
        if audio_codes.shape[0] == n_vq:
            return audio_codes[:, 0, :].permute(1, 0)
        if audio_codes.shape[1] == n_vq:
            return audio_codes[0].permute(1, 0)
        raise ValueError(f'unexpected audio_codes shape {tuple(audio_codes.shape)} for n_vq={n_vq}')

    @staticmethod
    def _Tnq_to_moss_codes(codes_Tnq: torch.Tensor) -> torch.Tensor:
        return codes_Tnq.permute(1, 0).unsqueeze(1)

    def _encode_ref(self, ref_audio_path: str) -> np.ndarray:
        """Encode a reference wav into MOSS voice codes of shape ``(T, n_vq)``.

        Pass the result as ``ref_codes`` to :meth:`infer` / :meth:`infer_stream`
        to clone that voice. Encoding once and reusing the codes keeps the voice
        identical across calls and skips re-encoding the same clip.
        """
        import torchaudio
        wav_t, sr = torchaudio.load(ref_audio_path)
        if sr != self.SAMPLE_RATE:
            wav_t = torchaudio.functional.resample(wav_t, sr, self.SAMPLE_RATE)
        n_ch = int(getattr(self.audio_tokenizer.config, 'number_channels', 2))
        if wav_t.shape[0] == 1:
            wav_t = wav_t.repeat(n_ch, 1)
        else:
            wav_t = wav_t[:n_ch]
        wav_t = wav_t.unsqueeze(0).to(self.device)
        with torch.no_grad():
            enc = self.audio_tokenizer.encode(wav_t, return_dict=True)
        return self._moss_codes_to_Tnq(enc.audio_codes, self.config.n_vq).cpu().numpy()

    @torch.no_grad()
    def _decode_codes(self, codes: torch.LongTensor) -> np.ndarray:
        c = self._Tnq_to_moss_codes(codes).to(self.device)
        dec = self.audio_tokenizer.decode(c, return_dict=True)
        wav_np = dec.audio[0].mean(0).cpu().float().numpy()
        return wav_np

    @torch.no_grad()
    def _decode_codes_stream(self, codes: torch.LongTensor, *, reset: bool) -> np.ndarray:
        c2d = codes.permute(1, 0).to(self.device)
        out = self.audio_tokenizer.batch_decode([c2d], num_quantizers=self.config.n_vq, streaming=True, reset_stream=reset)
        audio = out.audio
        if getattr(out, 'audio_lengths', None) is not None:
            length = int(out.audio_lengths[0].item())
            audio = audio[:, :, :length]
        return audio[0].mean(0).cpu().float().numpy()

    def _reset_stream_session(self) -> None:
        reset = getattr(self.audio_tokenizer, '_reset_batch_decode_streaming_state', None)
        if callable(reset):
            try:
                reset()
            except Exception:
                pass

    def _get_sea_pipeline(self):
        """Lazily build and cache the sea-g2p Vietnamese pipeline (reused per call)."""
        pipe = getattr(self, '_sea_pipeline', None)
        if pipe is None:
            from sea_g2p import SEAPipeline
            pipe = SEAPipeline(lang='vi')
            self._sea_pipeline = pipe
        return pipe

    def _build_prompt_2d(self, text: str, ref_codes: Optional[np.ndarray], ref_text: Optional[str], emotion_token_id: int, phonemes: Optional[str]=None, ref_phonemes: Optional[str]=None) -> torch.LongTensor:
        from .prompt_v3_turbo import build_prompt_2d
        if phonemes is None:
            # Keep inline cues [cười]/[thở dài]/[hắng giọng] as <|emotion_1/2/3|>.
            text_phones = _phonemize_with_emotions(text, self._get_sea_pipeline())
        else:
            text_phones = phonemes
        if ref_phonemes is not None:
            ref_phones = ref_phonemes
        elif ref_text:
            from sea_g2p import G2P
            ref_phones = G2P().convert(ref_text)
        else:
            ref_phones = ''
        ref_tensor: Optional[torch.LongTensor] = None
        if ref_codes is not None:
            ref_tensor = torch.as_tensor(ref_codes, dtype=torch.long)
        return build_prompt_2d(text_phones, ref_tensor, self.tokenizer, self.config, emotion_token_id=emotion_token_id, ref_phonemes=ref_phones)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)

    def _resolve_dtype(self, dtype: str) -> torch.dtype:
        if dtype == 'auto':
            if self.device.type == 'cuda':
                return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.float32
        return {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}[dtype]
