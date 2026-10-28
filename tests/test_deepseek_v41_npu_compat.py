# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mcore_bridge.model.gpts.deepseek_v4 import DeepseekV4Bridge
from mcore_bridge.model.mm_gpts.deepseek_v41 import DeepseekV41Bridge


def test_engram_block_scales():
    weight = torch.ones((64, 64), dtype=torch.float8_e4m3fn)
    scales = torch.tensor([[1., 2.], [4., 8.]])
    actual = DeepseekV41Bridge._dequantize_engram_rows(weight, scales)
    expected = scales.repeat_interleave(32, dim=0).repeat_interleave(32, dim=1)
    torch.testing.assert_close(actual, expected, check_dtype=False)


def test_engram_row_scales():
    weight = torch.ones((64, 64), dtype=torch.float8_e4m3fn)
    scales = torch.arange(1, 129, dtype=torch.float32).reshape(64, 2)
    actual = DeepseekV41Bridge._dequantize_engram_rows(weight, scales)
    torch.testing.assert_close(actual, scales.repeat_interleave(32, dim=1))


@pytest.mark.parametrize('language_model_only', [True, False])
def test_image_marker_loading(language_model_only):
    bridge = object.__new__(DeepseekV41Bridge)
    bridge.config = SimpleNamespace(language_model_only=language_model_only)
    bridge._set_state_dict = Mock()
    state_dict = {}
    with patch.object(DeepseekV4Bridge, '_convert_pre_process', return_value={}):
        bridge._convert_pre_process(None, state_dict, '', True)
    if language_model_only:
        bridge._set_state_dict.assert_not_called()
    else:
        assert [call.args[1] for call in bridge._set_state_dict.call_args_list
                ] == ['visual.image_start', 'visual.image_end', 'visual.image_newline']
