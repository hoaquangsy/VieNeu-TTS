import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from vieneu.utils import _linear_overlap_add, extract_speech_ids
from vieneu_utils.core_utils import split_text_into_chunks, split_into_chunks_v2, join_audio_chunks
from apps.voice_clone_api import _build_sentence_aware_chunk_plan, _chunk_audio_qa

# --- Text Utils Tests ---

def test_split_text_into_chunks():
    text = "Đây là một câu ngắn. Đây là một câu dài hơn một chút để kiểm tra xem nó có bị chia ra không nếu chúng ta đặt giới hạn ký tự thấp."
    chunks = split_text_into_chunks(text, max_chars=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 50

def test_split_text_paragraphs():
    text = "Đoạn 1.\n\nĐoạn 2."
    chunks = split_text_into_chunks(text, max_chars=100)
    assert len(chunks) == 2
    assert "Đoạn 1" in chunks[0]
    assert "Đoạn 2" in chunks[1]

def test_split_into_chunks_v2_uses_soft_continuations_for_long_sentences():
    text = "Đây là một câu dài cần được chia nhỏ để kiểm tra hành vi cắt câu, tránh dấu kết thúc giữa chừng và không gây ngắt mạnh ở các đoạn giữa."
    chunks = split_into_chunks_v2(text, max_chunk_size=40, min_chunk_size=8)
    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        assert not chunk.is_sentence_end
        assert chunk.text.endswith(",")
    assert chunks[-1].is_sentence_end

# --- Audio Utils Tests ---

def test_linear_overlap_add():
    # Create two overlapping frames
    frame_len = 100
    stride = 50
    frame1 = np.ones(frame_len, dtype=np.float32)
    frame2 = np.ones(frame_len, dtype=np.float32)

    frames = [frame1, frame2]
    out = _linear_overlap_add(frames, stride)

    # Total length should be stride * (len(frames) - 1) + frame_len = 50 * 1 + 100 = 150
    assert out.shape == (150,)
    assert np.any(out != 0)
    # With all ones and linear OLA, the result should be close to 1.0 where it overlaps
    assert np.allclose(out[50:100], 1.0)

def test_linear_overlap_add_empty():
    assert _linear_overlap_add([], 50).shape == (0,)

def test_join_audio_chunks_simple():
    chunks = [np.ones(100), np.zeros(100)]
    joined = join_audio_chunks(chunks, sr=16000)
    assert joined.shape == (200,)
    assert np.array_equal(joined[:100], np.ones(100))
    assert np.array_equal(joined[100:], np.zeros(100))

def test_join_audio_chunks_silence():
    chunks = [np.ones(100), np.ones(100)]
    sr = 16000
    silence_p = 0.1 # 0.1s * 16000 = 1600 samples
    joined = join_audio_chunks(chunks, sr=sr, silence_p=silence_p)
    assert joined.shape == (100 + 1600 + 100,)
    assert np.all(joined[100:1700] == 0)

def test_join_audio_chunks_crossfade():
    chunks = [np.ones(1000), np.zeros(1000)]
    sr = 16000
    crossfade_p = 0.01 # 0.01s * 16000 = 160 samples
    joined = join_audio_chunks(chunks, sr=sr, crossfade_p=crossfade_p)
    # Length should be 1000 + 1000 - 160 = 1840
    assert joined.shape == (1840,)
    assert joined[0] == 1.0
    assert joined[-1] == 0.0
    # Mid-point of crossfade should be 0.5
    assert np.allclose(joined[1000 - 80], 0.5, atol=0.01)

def test_join_audio_chunks_empty():
    assert join_audio_chunks([], 16000).shape == (0,)

def test_join_audio_chunks_single():
    chunk = np.ones(100)
    assert np.array_equal(join_audio_chunks([chunk], 16000), chunk)

def test_extract_speech_ids():
    codes_str = "<|speech_100|><|speech_101|><|speech_102|>"
    assert extract_speech_ids(codes_str) == [100, 101, 102]
    assert extract_speech_ids("no speech here") == []
    assert extract_speech_ids("<|speech_abc|>") == []

def test_v3_chunk_plan_merges_micro_sentences():
    plan = _build_sentence_aware_chunk_plan(
        [
            {
                "id": "intro",
                "text": (
                    "Các bạn thử nghĩ. "
                    "Điều đáng sợ là chuyện này không đến bằng tiếng nổ. "
                    "Nghe thì đơn giản. "
                    "Nó chỉ giống như thêm một công cụ vào công việc."
                ),
            }
        ],
        requested_max_chars=220,
        min_chunk_chars=40,
        max_chunk_chars=220,
    )
    chunks = plan["ttsChunks"]
    assert "Các bạn thử nghĩ." not in chunks
    assert "Nghe thì đơn giản." not in chunks
    assert chunks[0] == "Các bạn thử nghĩ. Điều đáng sợ là chuyện này không đến bằng tiếng nổ."
    assert chunks[1] == "Nghe thì đơn giản. Nó chỉ giống như thêm một công cụ vào công việc."
    assert all(len(chunk) <= 220 for chunk in chunks)

def test_v3_chunk_audio_qa_rejects_silent_micro_chunk():
    audio = np.zeros(48000 * 8, dtype=np.float32)
    qa = _chunk_audio_qa(audio, 48000, "Nghe thì đơn giản.")
    assert qa["valid"] is False
    assert qa["reason"] == "suspected_silent_chunk"
    assert qa["longestSilenceSeconds"] >= 8.0

def test_v3_chunk_audio_qa_accepts_non_silent_chunk():
    samples = np.linspace(0, 1, 48000, dtype=np.float32)
    audio = 0.2 * np.sin(2 * np.pi * 220 * samples).astype(np.float32)
    qa = _chunk_audio_qa(audio, 48000, "Nghe thì đơn giản. Nó chỉ giống như thêm một công cụ vào công việc.")
    assert qa["valid"] is True
