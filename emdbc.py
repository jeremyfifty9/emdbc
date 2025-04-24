"""
emdbc.py
Utilities for Empirical Mode Decomposition–based bias correction.

Authors
-------
Arkaprabha Ganguli
    Mathematics & Computer Science Division,
    Argonne National Laboratory, Lemont IL USA

Jeremy Feinstein
    Environmental Science Division,
    Argonne National Laboratory, Lemont IL USA
"""

__author__ = "Arkaprabha Ganguli, Jeremy Feinstein"

import os
from PyEMD import EEMD
import numpy as np
from scipy.fft import fft, fftfreq
import hashlib
from pathlib import Path
from sklearn.linear_model import QuantileRegressor
from scipy.signal import butter, filtfilt
import numpy as np

# Bandpass filter function
def bandpass_filter(data, low_cutoff, high_cutoff, fs=1.0, order=4):
    """
    Applies a bandpass filter to the data using the specified cutoffs, sample rate, and filter order.
    
    Args:
        data: Input signal.
        low_cutoff: Low frequency cutoff.
        high_cutoff: High frequency cutoff.
        fs: Sample rate (default: 1.0).
        order: Order of the filter (default: 4).
    
    Returns:
        Filtered signal.
    """
    nyquist = 0.5 * fs
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def compute_fft(imf):
    """
    Computes the FFT of the IMF and returns the frequency and magnitude.
    
    Args:
        imf: IMF signal.
    
    Returns:
        fft_freq: Frequencies corresponding to FFT.
        fft_magnitude: Magnitudes of the FFT.
    """
    sample_spacing = 1

    N = len(imf)
    fft_values = fft(imf)
    fft_freq = fftfreq(N, sample_spacing)[:N//2]  # Take the positive frequencies
    fft_magnitude = 2.0/N * np.abs(fft_values[:N//2])  # Magnitude of the FFT

    return fft_freq, fft_magnitude

def compute_eIMFs(s, noise):
    """
    Computes the Empirical Mode Decomposition (EMD) of a signal with varying noise.
    
    Args:
        s: Signal to decompose.
        noise: Initial noise level.
        max_retries: Maximum number of retries if EMD fails (default: 100).
    
    Returns:
        noise: The final noise value used.
        eIMFs: Empirical Mode Functions (IMFs).
    """
    max_retries = 100
    
    # Set numpy random seed
    np.random.seed(42)
    
    for retry_no in range(max_retries):
        eemd = EEMD(trials=250, noise_width=noise, parallel=True, processes=os.cpu_count())
        eemd.noise_seed(42+retry_no)

        eIMFs = eemd(s).astype(np.float32)
        nIMF = len(eIMFs[:-1])
        max_amp_freq = []
        
        for i in range(nIMF):
            fft_freq, fft_magnitude = compute_fft(eIMFs[i])
            max_amp_freq.append(fft_freq[np.argmax(fft_magnitude)])
        
        # Check the condition
        diff = np.diff(max_amp_freq)

        if np.sum(diff >= 0) == 0 and np.min(np.abs(diff / max_amp_freq[:-1])) > 0.2 and np.max(np.abs(diff / max_amp_freq[:-1])) < 0.8:
            return noise, eIMFs  # Exit the loop and return eIMFs if the condition is satisfied
        
        # Increment noise for the next iteration
        noise += np.random.uniform(0.00001, 0.0001)
    
    # Return nan if the maximum number of retries is exhausted
    return np.nan, np.nan

def calculate_disjoint_thresholds(imfs, filter_configs, fs=1.0):
    """
    Calculates disjoint thresholds for each IMF based on the provided filter configurations.
    
    Args:
        imfs: IMFs to be analyzed.
        filter_configs: Filter configurations for each timescale.
        fs: Sampling frequency (default: 1.0).
    
    Returns:
        Thresholds for each timescale.
    """
    num_imfs = imfs.shape[0]
    timescale_correlations = np.zeros((len(filter_configs), num_imfs))

    # Compute correlations for each IMF with each timescale
    for i, (label, (low_cutoff, high_cutoff)) in enumerate(filter_configs.items()):
        filtered_signal = bandpass_filter(imfs.sum(axis=0), low_cutoff, high_cutoff, fs)
        timescale_correlations[i] = [
            np.abs(np.corrcoef(imfs[j], filtered_signal)[0, 1]) for j in range(num_imfs)
        ]

    # Assign timescale to each IMF
    assigned_imfs = set()
    thresholds = {label: [] for label in filter_configs}
    for _ in range(num_imfs):
        max_corr_idx = np.unravel_index(
            np.argmax(timescale_correlations, axis=None), timescale_correlations.shape
        )
        timescale_idx, imf_idx = max_corr_idx

        if imf_idx not in assigned_imfs:  # Assign IMF if not already assigned
            label = list(filter_configs.keys())[timescale_idx]
            thresholds[label].append((imf_idx, timescale_correlations[timescale_idx, imf_idx]))
            assigned_imfs.add(imf_idx)

        # Remove assigned IMF from further consideration
        timescale_correlations[:, imf_idx] = -np.inf

    return {label: [idx for idx, _ in imfs] for label, imfs in thresholds.items()}

def qdm(obs, model_hist, model_fut, quantiles=np.linspace(0, 1, 101)):
    """
    Applies Quantile Delta Mapping (QDM) bias correction.
    
    Args:
        obs: Historical observations (array-like).
        model_hist: Historical model simulations (array-like).
        model_fut: Future model simulations (array-like).
        quantiles: Quantiles for QDM (default: np.linspace(0, 1, 101)).
    
    Returns:
        Bias-corrected historical and future model simulations.
    """

    # Calculate quantiles of historical observations, historical model, and future model data
    obs_quantiles = np.quantile(obs, quantiles)
    model_hist_quantiles = np.quantile(model_hist, quantiles)
    model_fut_quantiles = np.quantile(model_fut, quantiles)
    
    # Perform quantile mapping for future model data using historical observations
    corrected_hist = np.interp(model_hist, model_hist_quantiles, obs_quantiles)
    
    # Calculate the absolute change in quantiles (delta) between model_hist and model_fut
    delta_quantiles = model_fut - np.interp(model_fut, model_fut_quantiles, model_hist_quantiles)
    
    # Adjust future projections by adding the quantile delta to the mapped observed quantiles
    corrected_fut = np.interp(model_fut, model_fut_quantiles, obs_quantiles)+delta_quantiles 
    
    return corrected_hist, corrected_fut

def bc_qr_multiquantile(obs, model_hist, model_fut, lambda_reg=0.01, quantiles=np.arange(0.05, 1, 0.01)):
    """
    Applies bias correction using Quantile Regression for multiple quantiles.
    
    Args:
        obs: Historical observations (array-like).
        model_hist: Historical model simulations (array-like).
        model_fut: Future model simulations (array-like).
        lambda_reg: Regularization parameter (default: 0.01).
        quantiles: List of quantiles for bias correction (default: [0.05, 0.06, ..., 0.99]).
    
    Returns:
        Averaged bias-corrected historical and future model simulations.
    """
    # Create time indices and day numbers for historical data
    n_hist = len(obs)
    day_number_hist = np.tile(np.arange(1, 366), n_hist // 365 + 1)[:n_hist]

    n_model_hist = len(model_hist)
    bias = model_hist - obs[:n_model_hist]
    
    # Prepare the predictors for historical data
    X_hist = np.column_stack([day_number_hist[:n_model_hist], model_hist[:n_model_hist]])

    # Initialize arrays to accumulate corrected data for averaging
    corrected_hist_total = np.zeros_like(model_hist)
    corrected_fut_total = np.zeros_like(model_fut)

    # Loop over each quantile and perform quantile regression
    for quantile in quantiles:
        quantile_reg = QuantileRegressor(quantile=quantile, alpha=lambda_reg)
        quantile_reg.fit(X_hist, bias)

        # Predict the bias for historical data
        predicted_bias_hist = quantile_reg.predict(X_hist)
        corrected_hist = model_hist - predicted_bias_hist
        
        # Prepare predictors and predict the bias for future data
        n_fut = len(model_fut)
        day_number_fut = np.tile(np.arange(1, 366), n_fut // 365 + 1)[:n_fut]
        X_fut = np.column_stack([day_number_fut, model_fut])
        predicted_bias_fut = quantile_reg.predict(X_fut)
        corrected_fut = model_fut - predicted_bias_fut
        
        # Accumulate the corrected data for each quantile
        corrected_hist_total += corrected_hist
        corrected_fut_total += corrected_fut

    # Average the corrected data across the specified quantiles
    corrected_hist_avg = corrected_hist_total / len(quantiles)
    corrected_fut_avg = corrected_fut_total / len(quantiles)

    return corrected_hist_avg, corrected_fut_avg

def Bias_Correction_EMD_disjoint(original_series, hist_series, fut_series, noise, filter_configs):
    """
    Perform bias correction using Empirical Mode Decomposition (EMD) and adaptive disjoint IMF selection.
    
    Args:
        original_series: Original observed series.
        hist_series: Historical modeled series.
        fut_series: Future modeled series.
        noise: Noise for EMD ensemble.
        filter_configs: Dictionary of filter configurations for each timescale.
    
    Returns:
        Corrected historical and future series.
    """
    # Step 1: Compute IMFs
    noise_original, eIMFs_original = compute_eIMFs(original_series.reshape(-1), noise)
    noise_hist, eIMFs_hist = compute_eIMFs(hist_series.reshape(-1), noise)

    if len(hist_series) == len(fut_series) and (hist_series == fut_series).all():
        eIMFs_fut = eIMFs_hist
    else:
        noise_fut, eIMFs_fut = compute_eIMFs(fut_series.reshape(-1), noise)

    # If eIMFs failed to compute, return nan
    if np.any(np.isnan(eIMFs_original)) or np.any(np.isnan(eIMFs_hist)) or np.any(np.isnan(eIMFs_fut)):
        return False, False

    imfs_original, res_original = eIMFs_original[:-1], eIMFs_original[-1]
    imfs_hist, res_hist = eIMFs_hist[:-1], eIMFs_hist[-1]
    imfs_fut, res_fut = eIMFs_fut[:-1], eIMFs_fut[-1]

    thresholds_original = calculate_disjoint_thresholds(imfs_original, filter_configs)
    thresholds_hist = calculate_disjoint_thresholds(imfs_hist, filter_configs)
    thresholds_fut = calculate_disjoint_thresholds(imfs_fut, filter_configs)

    biweekly_original = np.sum(imfs_original[thresholds_original['Biweekly']], axis=0)
    seasonal_original = np.sum(imfs_original[thresholds_original['Seasonal']], axis=0)
    annual_original = np.sum(imfs_original[thresholds_original['Annual']], axis=0)
        
    biweekly_hist = np.sum(imfs_hist[thresholds_hist['Biweekly']], axis=0)
    seasonal_hist = np.sum(imfs_hist[thresholds_hist['Seasonal']], axis=0)
    annual_hist = np.sum(imfs_hist[thresholds_hist['Annual']], axis=0)

    biweekly_fut = np.sum(imfs_fut[thresholds_fut['Biweekly']], axis=0)
    seasonal_fut = np.sum(imfs_fut[thresholds_fut['Seasonal']], axis=0)
    annual_fut = np.sum(imfs_fut[thresholds_fut['Annual']], axis=0)

    biweekly_corrected = qdm(biweekly_original, biweekly_hist, biweekly_fut)
    seasonal_corrected = bc_qr_multiquantile(seasonal_original, seasonal_hist, seasonal_fut)
    annual_corrected = bc_qr_multiquantile(annual_original, annual_hist, annual_fut)
    res_corrected = qdm(res_original, res_hist, res_fut)

    all_corrected_hist = (
        biweekly_corrected[0] + seasonal_corrected[0] + annual_corrected[0] + res_corrected[0]
    )
    all_corrected_fut = (
        biweekly_corrected[1] + seasonal_corrected[1] + annual_corrected[1] + res_corrected[1]
    )

    return all_corrected_hist, all_corrected_fut
