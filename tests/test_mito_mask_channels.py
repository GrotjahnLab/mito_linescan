"""Regression test for BP-3.

`mito_mask.main` accepts --mito-channel / --target-channel (wired from the
draw_mask config), but the loop body hardcoded `mito_channel = 1` /
`target_channel = 0`, silently ignoring the config and swapping the channels.
Drive main with the GUI and TIFF IO stubbed and assert the channels actually
fed to the mask drawer come from the config, not the hardcoded pair.
"""

import numpy as np

import mito_linescan.mito_mask as mito_mask


def test_draw_mask_uses_config_channels(monkeypatch, tmp_path):
    # A 3-channel image where channel c is uniformly filled with value c.
    image = np.stack([np.full((6, 6), c, dtype=np.uint8) for c in range(3)])

    captured = {}

    def fake_draw(mito_image, scan_image):
        captured["mito"] = np.asarray(mito_image).copy()
        captured["target"] = np.asarray(scan_image).copy()
        return np.ones(mito_image.shape, dtype=bool), False

    monkeypatch.setattr(mito_mask.tf, "imread", lambda *_a, **_k: image)
    monkeypatch.setattr(mito_mask.tf, "imwrite", lambda *_a, **_k: None)
    monkeypatch.setattr(mito_mask, "draw_mitochondria", fake_draw)

    in_dir = tmp_path / "in"
    in_dir.mkdir()
    (in_dir / "img.tif").write_bytes(b"")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # Config asks for mito=0, target=2. The old hardcoded pair was mito=1/target=0.
    mito_mask.main.callback(
        input_directory=str(in_dir),
        manual_mask_directory=str(out_dir),
        target_channel=2,
        mito_channel=0,
        outliers_csv="",
    )

    assert np.all(captured["mito"] == 0), "mito channel should be config channel 0"
    assert np.all(captured["target"] == 2), "target channel should be config channel 2"
