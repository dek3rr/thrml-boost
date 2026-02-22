# Discrete Energy-Based Models

This module contains implementations of discrete energy-based models.

::: thrml-boost.models.DiscreteEBMFactor
    options:
        members:
            - energy
            - to_interaction_groups

::: thrml-boost.models.DiscreteEBMInteraction
    options:
        members: False

::: thrml-boost.models.SquareDiscreteEBMFactor
    options:
        members:
            - to_interaction_groups

::: thrml-boost.models.SpinEBMFactor
    options:
        members: false
        inherited_members: false

::: thrml-boost.models.CategoricalEBMFactor
    options:
        members: false
        inherited_members: false

::: thrml-boost.models.SquareCategoricalEBMFactor
    options:
        members: false
        inherited_members: false

::: thrml-boost.models.SpinGibbsConditional
    options:
        members:
            - compute_parameters

::: thrml-boost.models.CategoricalGibbsConditional
    options:
        members:
            - compute_parameters