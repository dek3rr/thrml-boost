# Observers

Observers collect statistics during sampling. `StateObserver` records raw
states; `MomentAccumulatorObserver` computes running means and variances
without storing every sample.

::: hamon.AbstractObserver
    options:
        members:
            - init

::: hamon.StateObserver
    options:
        members:
            - __init__

::: hamon.MomentAccumulatorObserver
    options:
        members:
            - __init__
