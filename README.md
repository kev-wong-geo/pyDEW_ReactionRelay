# pyDEW_ReactionRelay
Fluid-rock reaction relay for pyDEW v2.18. Last updated 10 Nov 2026.

## Description
This code utilises the pyDEW Python 3 wrapper for DEW-EQ3/6 (https://github.com/simonwmatthews/pyDEW) to chain together relayed EQ3/6 reactions following a specified and discretised pressure-temperature pathway. The fluid composition, fluid speciation, and mineral composition are calculated at full reaction progress at each step and saved to an Excel File. These results can then be subject to further fluid input.

Includes four files:
* demo.ipynb : jupyter notebook with commented code for the relay function and setup for a heterogeneous solid at isobaric, isothermal, and variable PT conditions.
* pyDEW_functions : supplementary functions for pyDEW. Includes conversions between pandas.Series and reaction inputs, mass, volume and density calculations, and element conservation functionality.
* relay_functions : structural functions for the relayed equations.
* EQ3_pickups_GS.xlsx : original data collected by Guillaume Siron, used as a basefile for the slab fluid compositions. See Siron et al. (2025).

### Reference
Siron, G., Blanchard, M., Aufort, J., Williams, S. E., & Vitale Brovarone, A. (2025). Decoupling of High-Pressure H2 Production From Serpentinization and Magnetite in Subduction Zones. Journal of Geophysical Research: Solid Earth, 130(9), e2024JB030567. https://doi.org/10.1029/2024JB030567
