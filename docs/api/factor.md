# Factors

Factors define energy contributions over groups of nodes. `WeightedFactor`
stores a weight tensor; `FactorSamplingProgram` pairs a factor with its
conditional sampler for use during block Gibbs sweeps.

::: hamon.AbstractFactor
    options:
        members:
            - to_interaction_groups

::: hamon.WeightedFactor
    options:
        members: false
        inherited_members: false

::: hamon.FactorSamplingProgram
    options:
        members:
            - __init__
