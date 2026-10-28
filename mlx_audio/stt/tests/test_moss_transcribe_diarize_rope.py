import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_audio.convert import build_quant_predicate
from mlx_audio.stt.models.moss_transcribe_diarize.config import (
    AudioConfig,
    ModelConfig,
    TextConfig,
)
from mlx_audio.stt.models.moss_transcribe_diarize.moss_transcribe_diarize import Model
from mlx_audio.stt.models.qwen3_asr.qwen3_asr import TextModel
from mlx_audio.utils import apply_quantization


@pytest.fixture
def inverse_frequencies():
    # Qwen3 inv_freq for head_dim=128, theta=1e6 after the published MOSS
    # runner's model.to(bfloat16), stored here as BF16 bits. No Torch dependency.
    # fmt: off
    bits = np.array([
        16256, 16206, 16166, 16134, 16088, 16046, 16012, 15970,
        15926, 15891, 15852, 15807, 15770, 15736, 15687, 15649,
        15618, 15569, 15528, 15496, 15450, 15408, 15374, 15333,
        15288, 15252, 15215, 15169, 15131, 15098, 15050, 15011,
        14979, 14931, 14890, 14857, 14813, 14770, 14736, 14695,
        14650, 14614, 14578, 14531, 14493, 14461, 14412, 14373,
        14341, 14294, 14252, 14219, 14176, 14132, 14097, 14058,
        14013, 13976, 13941, 13893, 13855, 13824, 13775, 13735,
    ], dtype=np.uint32)
    # fmt: on
    return (bits << 16).view(np.float32)


@pytest.fixture
def config():
    return ModelConfig(
        text_config=TextConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
        ),
        audio_config=AudioConfig(
            num_mel_bins=8,
            d_model=8,
            encoder_layers=1,
            encoder_attention_heads=2,
            encoder_ffn_dim=16,
            max_source_positions=16,
        ),
    )


@pytest.fixture
def model(config):
    return Model(config)


def test_rotary_frequencies_match_bf16_reference(model, inverse_frequencies):
    for layer in model.layers:
        rope = layer.self_attn.rope
        assert rope._freqs.dtype == mx.float32
        np.testing.assert_array_equal(
            np.array(rope._freqs), np.reciprocal(inverse_frequencies)
        )


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("length", [1, 4])
@pytest.mark.parametrize("offset", [0, 131068])
def test_rotary_basis(model, inverse_frequencies, dtype, length, offset):
    values = np.zeros((1, length, 16, 128), dtype=np.float32)
    values[:, :, :8, :64] = 1
    values[:, :, 8:, 64:] = 1
    # Preserve the transposed query layout used by TextAttention.
    x = mx.array(values, dtype=dtype).transpose(0, 2, 1, 3)
    actual = model.layers[0].self_attn.rope(x, offset=offset)

    positions = np.arange(offset, offset + length, dtype=np.float32)
    angles = positions[:, None] * inverse_frequencies
    cos, sin = np.cos(angles), np.sin(angles)
    expected = np.empty((1, 16, length, 128), dtype=np.float32)
    expected[:, :8, :, :64] = cos
    expected[:, :8, :, 64:] = sin
    expected[:, 8:, :, :64] = -sin
    expected[:, 8:, :, 64:] = cos
    actual = np.array(actual.astype(mx.float32))

    # Bounds native-kernel trigonometry/rounding error, not Torch eager parity.
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0.02)
    if offset == 0:
        np.testing.assert_array_equal(
            actual[:, :, 0], np.array(x.astype(mx.float32))[:, :, 0]
        )


def test_rotary_replacement_is_moss_local(config, model):
    from mlx_audio.stt.models.moss_transcribe_diarize.moss_transcribe_diarize import (
        MossRoPE,
    )

    assert all(isinstance(layer.self_attn.rope, MossRoPE) for layer in model.layers)
    qwen = TextModel(config.text_config)
    assert all(type(layer.self_attn.rope) is nn.RoPE for layer in qwen.layers)

    # Scaling remains on the existing path until separately supported.
    config.text_config.rope_scaling = {"rope_type": "linear", "factor": 2.0}
    scaled = Model(config)
    assert all(type(layer.self_attn.rope) is nn.RoPE for layer in scaled.layers)


def test_rotary_buffer_is_excluded_from_weights(config, model, tmp_path):
    original = TextModel(config.text_config)
    weights = tree_flatten(original.parameters())
    decoder = model.model.language_model
    assert dict(tree_flatten(decoder.parameters())).keys() == dict(weights).keys()
    decoder.load_weights(weights, strict=True)

    filename = str(tmp_path / "model.safetensors")
    model.save_weights(filename)
    saved = mx.load(filename)
    assert not any("rope" in key for key in saved)
    Model(config).load_weights(filename, strict=True)


def test_weight_cast_preserves_rotary_denominators(model):
    before = [np.array(layer.self_attn.rope._freqs) for layer in model.layers]
    model.set_dtype(mx.bfloat16)
    assert model.layers[0].self_attn.q_proj.weight.dtype == mx.bfloat16
    for layer, expected in zip(model.layers, before):
        assert layer.self_attn.rope._freqs.dtype == mx.float32
        np.testing.assert_array_equal(np.array(layer.self_attn.rope._freqs), expected)


def test_quantization_skips_rotary_buffer(config, model):
    ropes = [layer.self_attn.rope for layer in model.layers]
    assert all(not hasattr(rope, "to_quantized") for rope in ropes)
    nn.quantize(
        model, group_size=64, bits=4, class_predicate=build_quant_predicate(model)
    )
    assert isinstance(model.layers[0].self_attn.q_proj, nn.QuantizedLinear)
    assert isinstance(model.model.vq_adaptor.layers.layers[0], nn.Linear)
    for layer, rope in zip(model.layers, ropes):
        assert layer.self_attn.rope is rope
        assert rope._freqs.dtype == mx.float32

    weights = dict(tree_flatten(model.parameters()))
    loaded = Model(config)
    apply_quantization(
        loaded,
        {"quantization": {"group_size": 64, "bits": 4}},
        weights,
        loaded.model_quant_predicate,
    )
    loaded.load_weights(list(weights.items()), strict=True)
    assert isinstance(loaded.layers[0].self_attn.q_proj, nn.QuantizedLinear)
    assert loaded.layers[0].self_attn.rope._freqs.dtype == mx.float32
