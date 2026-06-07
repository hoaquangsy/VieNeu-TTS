from __future__ import annotations
from transformers import PretrainedConfig
from typing import Optional

class VieNeuV3TurboConfig(PretrainedConfig):
    model_type = 'vieneu_v3_turbo'

    def __init__(self, text_vocab_size: int=389, audio_vocab_size: int=1024, n_vq: int=16, hidden_size: int=768, num_hidden_layers: int=12, num_attention_heads: int=12, num_key_value_heads: int=4, head_dim: Optional[int]=None, intermediate_size: int=3072, max_position_embeddings: int=2048, rope_theta: float=1000000.0, local_rope_theta: float=10000.0, rms_norm_eps: float=1e-06, attention_dropout: float=0.0, tie_word_embeddings: bool=False, local_num_hidden_layers: int=2, local_num_attention_heads: int=8, local_intermediate_size: int=2048, pad_token_id: int=0, bos_token_id: int=1, eos_token_id: int=2, text_prompt_start_token_id: int=3, text_prompt_end_token_id: int=4, speech_generation_start_token_id: int=5, speech_generation_end_token_id: int=6, audio_ref_slot_token_id: int=7, emotion_0_token_id: int=8, emotion_1_token_id: int=9, emotion_2_token_id: int=10, emotion_3_token_id: int=11, emotion_4_token_id: int=12, unk_token_id: int=13, audio_tokenizer_pretrained_name_or_path: str='OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano', audio_sample_rate: int=48000, audio_pad_token_id: int=1024, **kwargs):
        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.text_vocab_size = text_vocab_size
        self.audio_vocab_size = audio_vocab_size
        self.n_vq = n_vq
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.local_rope_theta = local_rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.attention_dropout = attention_dropout
        self.local_num_hidden_layers = local_num_hidden_layers
        self.local_num_attention_heads = local_num_attention_heads
        self.local_intermediate_size = local_intermediate_size
        self.text_prompt_start_token_id = text_prompt_start_token_id
        self.text_prompt_end_token_id = text_prompt_end_token_id
        self.speech_generation_start_token_id = speech_generation_start_token_id
        self.speech_generation_end_token_id = speech_generation_end_token_id
        self.audio_ref_slot_token_id = audio_ref_slot_token_id
        self.emotion_0_token_id = emotion_0_token_id
        self.emotion_1_token_id = emotion_1_token_id
        self.emotion_2_token_id = emotion_2_token_id
        self.emotion_3_token_id = emotion_3_token_id
        self.emotion_4_token_id = emotion_4_token_id
        self.unk_token_id = unk_token_id
        self.audio_tokenizer_pretrained_name_or_path = audio_tokenizer_pretrained_name_or_path
        self.audio_sample_rate = audio_sample_rate
        self.audio_pad_token_id = audio_pad_token_id
