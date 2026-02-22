# Discrete Energy-Based Models

This module contains implementations of discrete energy-based models.

::: thrml_boost.models.DiscreteEBMFactor
    options:
        members:
            - energy
            - to_interaction_groups

::: thrml_boost.models.DiscreteEBMInteraction
    options:
        members: False

::: thrml_boost.models.SquareDiscreteEBMFactor
    options:
        members:
            - to_interaction_groups

::: thrml_boost.models.SpinEBMFactor
    options:
        members: false
        inherited_members: false

::: thrml_boost.models.CategoricalEBMFactor
    options:
        members: false
        inherited_members: false

::: thrml_boost.models.SquareCategoricalEBMFactor
    options:
        members: false
        inherited_members: false

::: thrml_boost.models.SpinGibbsConditional
    options:
        members:
            - compute_parameters

::: thrml_boost.models.CategoricalGibbsConditional
    options:
        members:
            - compute_parameters