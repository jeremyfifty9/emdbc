```
    ______                    ______ __  __ _____  ____   _____
   / \  / \                  |  ____|  \/  |  __ \|  _ \ / ____|
  /   /\   \                 | |__  | \  / | |  | | |_) | |
 /   /  \   \                |  __| | |\/| | |  | |  _ <| |
 \--/----\--/                | |____| |  | | |__| | |_) | |____
  \/______\/                 |______|_|  |_|_____/|____/ \_____|
```

# EMDBC - Empirical Mode Based Bias Correction

This repository contains the Python implementation for **Empirical Mode Decomposition Based Bias Correction (EMDBC)**.

Please refer to the manuscript (citation below) for detailed methodology and theoretical background.

---

## File Structure

```
├── emdbc.py
├── analysis.ipynb
└── validation_area_indices.txt

EMDBC/
├── emdbc.py            # Python script containing the core EMDBC implementation
│
├── analysis.ipynb      # Jupyter notebook with a sample application of the EMDBC function
│
└── sample_data/        # Contains a single sample temperature time series: observed, historical model (control), and future model (projected)
    ├── observed.npy    # Observed temperature time series (Livneh)
    ├── ycontrol.npy    # Historical modeled temperature time series (WRF-CCSM, 1995–2004)
    └── yproject.npy    # Future modeled temperature time series (WRF-CCSM, 2045–2054)
```

---

## Requirements

To run this code, ensure Python 3.11.9 is installed with the following packages:

- `numpy==1.26.4`
- `scipy==1.13.1`
- `pandas==2.2.3`
- `matplotlib==3.10.0`
- `emd-signal==1.6.0`
- `scikit-learn==1.6.0`
- `jupyterlab==4.3.5`

Alternatively, install via Anaconda:

    conda env create -f environment.yml

And activate:

    conda activate emdbc_env_py3_11_9

---

## Usage

To apply bias correction using EMDBC, apply the `Bias_Correction_EMD_disjoint` method in `emdbc.py`. Please adjust the `parallel` and `processes` setting in the `EEMD` call according to your hardware.

### `Bias_Correction_EMD_disjoint(original_series, hist_series, fut_series, noise, filter_configs)`

Perform bias correction on historical and future modeled time series using Empirical Mode Decomposition (EMD) and adaptive disjoint IMF selection.

#### Parameters:
- **original_series** (`array-like`):  
  The observed time series to be used as the historical reference for bias correction.
  
- **hist_series** (`array-like`):  
  The historical modeled time series to be used as the historical reference for bias correction.
  
- **fut_series** (`array-like`):  
  The future modeled time series to be corrected.
  
- **noise** (`float`):  
  The initial noise level to use for the ensemble EMD.
  
- **filter_configs** (`dict`):  
  A dictionary that maps timescales (biweekly, seasonal, annual) to frequency bands (low and high cutoff values) for bandpass filtering:
```
filter_configs = {
    'Biweekly': (1/30, 1/3),
    'Seasonal': (1/180, 1/30),
    'Annual': (1/(365*2.5), 1/180)
}
```

#### Returns:
- **tuple** (`array-like`, `array-like`):  
  The corrected historical and future time series.

#### Example:
 - Refer to `analysis.ipynb`


---

## Citation

Please cite the manuscript when using this code in your research.

Preprint citation:
```
@Article{egusphere-2025-1112,
  AUTHOR = {Ganguli, A. and Feinstein, J. and Raji, I. and Akinsanola, A. and Aghili, C. and Jung, C. and Branham, J. and Wall, T. and Huang, W. and Kotamarthi, R.},
  TITLE = {Bias Correcting Regional Scale Earth Systems Model Projections: Novel Approach using Empirical Mode Decomposition},
  JOURNAL = {EGUsphere},
  VOLUME = {2025},
  YEAR = {2025},
  PAGES = {1--23},
  URL = {https://egusphere.copernicus.org/preprints/2025/egusphere-2025-1112/},
  DOI = {10.5194/egusphere-2025-1112}
}
```

## Acknowledgements
This  material  is  based  upon  work  supported  by  Laboratory  Directed  Research  and  Development (LDRD) funding from Argonne National Laboratory, provided by the Director, Office of Science, of the U.S. Department of Energy under Contract No. DE-AC02-06CH11357.