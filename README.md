# RFmicroscope
Provide several tools to understand a given random forest, based on sklearn

## To-Do plans:

### Summarize the training process:
- observation - feature splitting matrix (What observations drive a certain feature split)


### Summarize of a given RF:
- Tabulate the RF into depth - node - (feature, observation) tensor
- Distance matrix of leaves
- leaf info: diameter

### Population level:
- Get important features used by RF.
- Get important feature pairs used by RF.
- Get important feature triplets used by RF.
- Get frequent feature interactions used by RF.
- Cluster the samples into different groups.

### Individual level:
- "Explain" why RF has a particular prediction for a sample.
- How the prediction of sample A will change when
  - A could be perturbed by random noise.
  - A could be perturbed by adversarial noise.
  - some adversarial samples can be added into RF
- Frequent features/interactions for a particular sample.
