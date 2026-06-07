"""
Batched serving engine for v3 Turbo (static batching).

Ties the batched backbone + batched acoustic frame generation into one loop that
advances B requests together:

    prefill(prompts) -> loop { acoustic frame (B) -> EOS check -> backbone step (B) }

Finished requests are masked out (their last frame is kept) and the batch keeps
running until all finish or ``max_new_frames``. Each request's codes are decoded
to a waveform at the end. Pure PyTorch, runs on CUDA (or CPU).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from .batched_acoustic import generate_frame_batched
from .batched_backbone import BatchedBackbone


class V3TurboBatchEngine:
    def __init__(self, tts):
        # tts: VieNeuTTSv3Turbo (provides prompt building, embeddings, heads, codec)
        self.tts = tts
        self.model = tts.model
        self.config = tts.config
        self.bb = BatchedBackbone(tts.model)
        self._graphs = {}  # (B, temp, top_k, top_p) -> CudaGraphedFrame (acoustic step)

    def _get_graph(self, B, temperature, top_k, top_p):
        key = (B, round(temperature, 4), top_k, round(top_p, 4))
        if key not in self._graphs:
            from .cudagraph import CudaGraphedFrame
            self._graphs[key] = CudaGraphedFrame(
                self.model, B, temperature=temperature, top_k=top_k, top_p=top_p
            )
        return self._graphs[key]

    @torch.no_grad()
    def _prompt_embeds(self, req) -> torch.Tensor:
        cfg = self.config
        # Built-in default voices carry a speaker reserved token (``voice_token_id``)
        # that overrides the emotion tag; cloned voices leave it None.
        voice_token_id = req.get("voice_token_id")
        if voice_token_id is not None:
            emo = int(voice_token_id)
        else:
            emo = cfg.emotion_0_token_id if req.get("emotion", "natural") == "natural" else cfg.emotion_4_token_id
        phonemes = req.get("phonemes")
        if phonemes is None:
            from vieneu_utils.phonemize_text import phonemize_text_with_emotions
            phonemes = phonemize_text_with_emotions(req["text"])
        ref_codes = req.get("ref_codes")
        prompt_2d = self.tts._build_prompt_2d("", ref_codes, None, emo, phonemes=phonemes)
        return self.model._build_inputs_embeds(prompt_2d.unsqueeze(0).to(self.tts.device))[0]  # (T, H)

    @torch.no_grad()
    def generate_batch(
        self,
        requests: List[dict],
        *,
        temperature: float = 0.8,
        top_k: int = 25,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        max_new_frames: int = 300,
        use_cudagraph: bool = False,
    ) -> List[np.ndarray]:
        """Generate a waveform for each request in one batched run.

        ``temperature`` is a scalar applied to every codebook. ``repetition_penalty``
        (default 1.2, matching the single-path engine) down-weights codes already
        produced per row/codebook to avoid repetition artifacts.

        ``use_cudagraph=True`` captures (and caches) a CUDA graph of the per-frame
        acoustic step for this batch size — big per-step speedup, reused across calls.
        It is ignored when ``repetition_penalty != 1.0`` (the penalty needs dynamic
        per-row history, which a static graph cannot hold).
        """
        cfg = self.config
        n_vq = cfg.n_vq
        eos_id = cfg.speech_generation_end_token_id
        sgs = cfg.speech_generation_start_token_id
        pad = cfg.audio_pad_token_id
        dev = self.tts.device

        embeds_list = [self._prompt_embeds(r) for r in requests]
        B = len(requests)

        # Per-row, per-codebook history for the repetition penalty (matches the
        # single-path decode_one_frame). None when the penalty is disabled.
        history = ([[set() for _ in range(n_vq)] for _ in range(B)]
                   if repetition_penalty != 1.0 else None)
        # CUDA graph bakes in a static step → incompatible with dynamic rep-penalty.
        use_graph = use_cudagraph and dev.type == "cuda" and repetition_penalty == 1.0
        graphed = self._get_graph(B, temperature, top_k, top_p) if use_graph else None

        h, cache, mask, pos = self.bb.prefill(embeds_list)   # h: (B, H)
        finished = [False] * B
        codes_per_req: List[List[torch.Tensor]] = [[] for _ in range(B)]

        for _ in range(max_new_frames):
            if graphed is not None:
                codes, is_eos = graphed.run(h)               # acoustic frame via CUDA graph
            else:
                codes, prefill_out = generate_frame_batched(
                    self.model, h, temperature=temperature, top_k=top_k, top_p=top_p,
                    repetition_penalty=repetition_penalty, history=history,
                )
                is_eos = (self.model.text_lm_head(prefill_out[:, 0]).float().argmax(-1) == eos_id)
            for b in range(B):
                if not finished[b]:
                    codes_per_req[b].append(codes[b])   # include the EOS frame (matches single path)
                    if bool(is_eos[b]):
                        finished[b] = True
            if all(finished):
                break

            # Feed the generated frame back as the next backbone input.
            slot = torch.full((B, 1, n_vq + 1), pad, dtype=torch.long, device=dev)
            slot[:, :, 0] = sgs
            slot[:, 0, 1:] = codes
            se = self.model._build_inputs_embeds(slot)
            h, cache, mask, pos = self.bb.decode_step(se, cache, mask, pos)

        wavs: List[np.ndarray] = []
        for b in range(B):
            if codes_per_req[b]:
                c = torch.stack(codes_per_req[b]).cpu()      # (T, n_vq)
                wavs.append(self.tts._decode_codes(c))
            else:
                wavs.append(np.zeros(0, dtype=np.float32))
        return wavs
