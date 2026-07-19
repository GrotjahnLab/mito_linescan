"""Regression test for SCI-2.

The 3D STED pipeline assumes a fixed channel order (0=mtDNA, 1=mito, 2=septin).
Per the Phase-1 decision we keep that assumption but add a guard that logs the
assumed mapping with per-channel ranges and warns when a channel looks
implausible (empty / degenerate) — the signature of a mislabeled stack.
"""

import numpy as np

from mito_linescan.sted_colocalization_3d import validate_channel_order


def test_plausible_channels_log_mapping_without_warnings():
    rng = np.random.default_rng(0)
    channels = [rng.random((4, 4, 4)) * (i + 1) for i in range(3)]
    logs = []

    warnings = validate_channel_order(channels, log=logs.append)

    assert warnings == []
    # The assumed mapping is logged for the operator to eyeball.
    assert any("channel 0 -> mtDNA" in line for line in logs)
    assert any("channel 2 -> septin" in line for line in logs)


def test_empty_channel_triggers_warning():
    channels = [
        np.zeros((4, 4, 4)),                          # channel 0: empty
        np.linspace(0, 1, 64).reshape(4, 4, 4),       # channel 1: fine
        np.linspace(0, 2, 64).reshape(4, 4, 4),       # channel 2: fine
    ]
    logs = []

    warnings = validate_channel_order(channels, log=logs.append)

    assert any("channel 0" in w for w in warnings), warnings
    assert not any("channel 1" in w for w in warnings)
    assert not any("channel 2" in w for w in warnings)
    assert any("WARNING" in line for line in logs)
