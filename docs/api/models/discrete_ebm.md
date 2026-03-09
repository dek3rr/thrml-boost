# Discrete Energy-Based Models

This module contains implementations of discrete energy-based models.

::: hamon.models.DiscreteEBMFactor
    options:
        members:
            - energy
            - to_interaction_groups

::: hamon.models.DiscreteEBMInteraction
    options:
        members: False

::: hamon.models.SquareDiscreteEBMFactor
    options:
        members:
            - to_interaction_groups

::: hamon.models.SpinEBMFactor
    options:
        members: false
        inherited_members: false

::: hamon.models.CategoricalEBMFactor
    options:
        members: false
        inherited_members: false

::: hamon.models.SquareCategoricalEBMFactor
    options:
        members: false
        inherited_members: false

::: hamon.models.SpinGibbsConditional
    options:
        members:
            - compute_parameters

::: hamon.models.CategoricalGibbsConditional
    options:
        members:
            - compute_parameters