# v1.2.0 (2026-08-XX)

- Added a easy to use target visibility notebook ("target_visibility.ipynb") which can accept multiple targets including just names of targets.
    - The target visibility notebook accepts planet names ending in a lowercase letter ("TRAPPIST-1 b", "WASP-160 B b"). Their ephemerides are pulled from the NASA Exoplanet Archive, and transits (blue) and secondary eclipses (purple) are overlaid on the visibility timeline with the observable fraction of each event reported. Unresolvable stars, unknown planets and non-transiting planets each raise a warning and degrade gracefully.
- Added `use_dynamic_earthlimb` (default False) to `Visibility`. When True the boresight Earth limb keep-out follows the piecewise DPC wedge curve as a function of the Earth illumination angle at the nearest limb point, instead of the fixed day/night limb limits. Star tracker keep-outs are unaffected.
