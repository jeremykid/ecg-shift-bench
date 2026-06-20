"""Tests for ECG preprocessing utilities."""

import numpy as np
import pytest

from ecgshiftbench.utils.preprocessing import (
    bandpass_filter,
    normalize_signal,
    pad_or_truncate,
    remove_baseline_wander,
    resample_signal,
)


@pytest.fixture
def ecg_signal_2d():
    """Random 12-lead ECG signal of 10 seconds at 500 Hz."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((12, 5000)).astype(np.float32)


@pytest.fixture
def ecg_signal_1d():
    """Random single-lead ECG signal."""
    rng = np.random.default_rng(1)
    return rng.standard_normal(5000).astype(np.float32)


class TestBandpassFilter:
    def test_output_shape_2d(self, ecg_signal_2d):
        filtered = bandpass_filter(ecg_signal_2d, lowcut=0.5, highcut=40.0, fs=500.0)
        assert filtered.shape == ecg_signal_2d.shape

    def test_output_shape_1d(self, ecg_signal_1d):
        filtered = bandpass_filter(ecg_signal_1d, lowcut=0.5, highcut=40.0, fs=500.0)
        assert filtered.shape == ecg_signal_1d.shape

    def test_output_dtype(self, ecg_signal_2d):
        filtered = bandpass_filter(ecg_signal_2d)
        assert filtered.dtype == np.float32

    def test_attenuates_dc(self, ecg_signal_2d):
        """DC component should be strongly attenuated after bandpass."""
        dc_signal = ecg_signal_2d + 1000.0
        filtered = bandpass_filter(dc_signal)
        # Mean should be much closer to zero after filtering
        assert abs(filtered.mean()) < 1.0


class TestRemoveBaselineWander:
    def test_output_shape(self, ecg_signal_2d):
        out = remove_baseline_wander(ecg_signal_2d, fs=500.0)
        assert out.shape == ecg_signal_2d.shape

    def test_output_dtype(self, ecg_signal_2d):
        out = remove_baseline_wander(ecg_signal_2d)
        assert out.dtype == np.float32

    def test_reduces_dc(self, ecg_signal_2d):
        dc_signal = ecg_signal_2d + 500.0
        out = remove_baseline_wander(dc_signal)
        assert abs(out.mean()) < abs(dc_signal.mean())


class TestNormalizeSignal:
    def test_zscore_2d_per_lead(self, ecg_signal_2d):
        normed = normalize_signal(ecg_signal_2d, method="zscore", per_lead=True)
        assert normed.shape == ecg_signal_2d.shape
        # Each lead should have ~0 mean and ~1 std
        for i in range(normed.shape[0]):
            assert abs(normed[i].mean()) < 1e-3
            assert abs(normed[i].std() - 1.0) < 1e-3

    def test_zscore_1d(self, ecg_signal_1d):
        normed = normalize_signal(ecg_signal_1d, method="zscore")
        assert abs(normed.mean()) < 1e-3
        assert abs(normed.std() - 1.0) < 1e-3

    def test_minmax_2d(self, ecg_signal_2d):
        normed = normalize_signal(ecg_signal_2d, method="minmax", per_lead=True)
        for i in range(normed.shape[0]):
            assert normed[i].min() >= 0.0
            assert normed[i].max() <= 1.0 + 1e-6

    def test_invalid_method(self, ecg_signal_2d):
        with pytest.raises(ValueError, match="method must be"):
            normalize_signal(ecg_signal_2d, method="invalid")

    def test_constant_lead(self):
        """Constant lead should not cause division by zero."""
        signal = np.zeros((12, 100), dtype=np.float32)
        normed = normalize_signal(signal, method="zscore")
        assert np.all(normed == 0.0)


class TestResampleSignal:
    def test_downsample_2d(self, ecg_signal_2d):
        resampled = resample_signal(ecg_signal_2d, orig_fs=500.0, target_fs=250.0)
        assert resampled.shape == (12, 2500)

    def test_upsample_1d(self, ecg_signal_1d):
        resampled = resample_signal(ecg_signal_1d, orig_fs=500.0, target_fs=1000.0)
        assert resampled.shape == (10000,)

    def test_no_change_same_rate(self, ecg_signal_2d):
        resampled = resample_signal(ecg_signal_2d, orig_fs=500.0, target_fs=500.0)
        np.testing.assert_array_equal(resampled, ecg_signal_2d)

    def test_output_dtype(self, ecg_signal_2d):
        resampled = resample_signal(ecg_signal_2d, orig_fs=500.0, target_fs=250.0)
        assert resampled.dtype == np.float32


class TestPadOrTruncate:
    def test_pad_2d(self, ecg_signal_2d):
        padded = pad_or_truncate(ecg_signal_2d, target_length=6000)
        assert padded.shape == (12, 6000)
        # Padded region should be zero
        assert np.all(padded[:, 5000:] == 0.0)

    def test_truncate_2d(self, ecg_signal_2d):
        truncated = pad_or_truncate(ecg_signal_2d, target_length=4000)
        assert truncated.shape == (12, 4000)
        np.testing.assert_array_equal(truncated, ecg_signal_2d[:, :4000])

    def test_pad_1d(self, ecg_signal_1d):
        padded = pad_or_truncate(ecg_signal_1d, target_length=6000)
        assert padded.shape == (6000,)
        assert np.all(padded[5000:] == 0.0)

    def test_exact_length(self, ecg_signal_2d):
        out = pad_or_truncate(ecg_signal_2d, target_length=5000)
        np.testing.assert_array_equal(out, ecg_signal_2d)

    def test_output_dtype(self, ecg_signal_2d):
        out = pad_or_truncate(ecg_signal_2d, target_length=4000)
        assert out.dtype == np.float32
