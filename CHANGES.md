# v1.2.0 (2026-08-XX)

- Added a easy to use target visibility notebook ("target_visibility.ipynb") which can accept multiple targets including just names of targets.
- Added `use_dynamic_earthlimb` (default False) to `Visibility`. When True the boresight Earth limb keep-out follows the piecewise DPC wedge curve as a function of the Earth illumination angle at the nearest limb point, instead of the fixed day/night limb limits. Star tracker keep-outs are unaffected.
