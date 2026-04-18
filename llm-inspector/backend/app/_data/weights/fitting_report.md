# Capability Weight Fitting Report

- Source: **snapshot:HELM-v1.10+LMSYS**
- Rows: 8
- R^2: 0.9576

## Fitted weights (normalized to sum=1)

| Dimension | Weight |
| --- | ---: |
| reasoning | 0.0000 |
| adversarial | 0.0968 |
| instruction | 0.2492 |
| coding | 0.2571 |
| safety | 0.0190 |
| protocol | 0.0690 |
| knowledge | 0.0781 |
| tool_use | 0.2307 |

## Method

Non-negative least squares (scipy.optimize.nnls), weights renormalized to 1.
See Lawson & Hanson (1974) Solving Least Squares Problems, SIAM.