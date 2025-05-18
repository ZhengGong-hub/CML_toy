# Simple Python Project with UV

This is a simple Python project that demonstrates the use of `uv` for virtual environment management and package installation.

## Running the Project

To run the project, simply execute:
uv run src/main.py

# about the rationale of some key steps
## reasons for drop samples that related to a cancelled program
Without understanding the reasons for program cancellations, we could be introducing significant bias and noise into our analysis by including them.

### Potential Sources of Noise in Cancellations:
- Administrative errors
- Individual decisions to drop out
- Program capacity issues
- Changes in eligibility
- Personal circumstances
- Quality of caseworker decisions

### Problems with Including Cancelled Programs:
- We don't know if the cancellation was random or systematic
- The cancellation might be correlated with unobserved characteristics
- It could introduce selection bias if certain types of people are more likely to have their programs cancelled
- The "intent-to-treat" analysis would be invalid if the cancellation mechanism is not random

### Check distribution after preprocess
Analyze whether the remaining 30% of the data maintains similar distributions for critical variables compared to the original dataset. Significant deviations might indicate that the sample is no longer representative.

### We do not account for heterogeneity of argriculture, production, service sector
I do not know why we would care.

### for identification strategy, we need common support identified and drop the samples that P(D|X) too close to 0 or 1.