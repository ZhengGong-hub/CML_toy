# Causal Machine Learning Analysis of Training Programs

This project analyzes the effectiveness of training programs using causal machine learning techniques, focusing on employment and earnings outcomes.

## Project Structure

```
.
├── src/
│   ├── main_data_preprocess.py    # Data preprocessing
│   ├── plot_ptype.py             # Program type visualizations
│   ├── plot_by_region.py         # Regional analysis
│   ├── plot_by_nan.py            # Missing value analysis
│   ├── propensity_score.py       # Propensity score matching
│   ├── sample_statistics.py      # Statistical analysis
│   └── treatment_effect.py       # Treatment effect estimation
├── output_data/                  # Analysis outputs
└── CML_public/                   # Raw data
```

## Key Decisions

### 1. Sample Selection
- Focused on training programs (PTYPE 1, 2)
- Excluded employment programs (PTYPE 3, 4)
- Removed cancelled programs due to unknown cancellation mechanisms
- Filtered age range to 30-50 years
- Removed duplicates and vocational degree level 2

### 2. Missing Value Treatment
- Imputed missing values for regional sector shares
- Dropped remaining observations with missing values
- Analyzed missing value patterns for transparency

### 3. Identification Strategy
- Used propensity score matching
- Ensured common support (dropped P(D|X) near 0 or 1)
- Maintained regional information for heterogeneity analysis

## Setup and Run

1. Environment setup:
```bash
uv venv .venv
source .venv/bin/activate  # Unix/macOS
# or
.venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
uv pip install -r requirements.txt
```

3. Run analysis:
```bash
uv run src/main.py
```

## Outputs

Generated in `output_data/`:
- `nan_percentage.txt`: Missing value analysis
- `sample_sizes.txt`: Sample size changes
- `distribution_comparison.txt`: Pre/post processing comparisons
- Visualization files (PNG)

Generated in `output_treatment_effect/`:
Generated in `output_treatment_effect_placebo/`:

## Limitations

1. Sample Size
   - Final sample ~30% of original data
   - Distribution checks ensure representativeness

2. Regional Analysis
   - Regional program effectiveness differences
   - Sectoral composition variations
   - Local labor market conditions

3. Identification
   - Relies on propensity score matching
   - Requires strong ignorability
   - Common support constraints

## Future Improvements

1. Consider alternative methods for handling cancelled programs
2. Implement more sophisticated imputation techniques
3. Add sensitivity analysis for key assumptions
4. Explore heterogeneous treatment effects