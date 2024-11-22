# pyDEW_ReactionRelay
Time-decompression relay for pyDEW. Work in progress. Last updated 22/11/2024.

## Description
This code utilises the pyDEW Python 3 wrapper for DEW-EQ3/6 (https://github.com/simonwmatthews/pyDEW) to chain together relayed EQ3/6 reactions following a specified and descretised pressure-temperature pathway. The fluid composition, fluid speciation, and mineral composition are calculated at full reaction progress at each step and saved to an Excel File. These results can then be subject to further fluid input.

Includes four files:
* TDR_demo.ipynb : jupyter notebook with commented code for the relay function and setup for a heterogeneous solid at isobaric and isothermal conditions.
* pyDEW_functions : supplementary functions for pyDEW. Includes conversions between pandas.Series and reaction inputs, mass, volume and density calculations, and element conservation functionality.
* relay_functions : structural functions for the relayed equations.
* EQ3_pickups_GS.xlsx : original data collected by Guillaume Siron, used as a basefile for the slab fluid compositions. See doi.org/10.21203/rs.3.rs-4973494/v1.
