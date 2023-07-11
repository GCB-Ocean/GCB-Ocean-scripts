# GCB-Ocean-scripts

Scripts used for analysis of the ocean carbon sink in the Global Carbon Budget

Currently, here deposited is the file used for evaluation of fugacity of CO2 (fCO2) in seawater Global Ocean Biogeochemical Models (GOBMs) and surface ocean fCO2 observation-based data-products in comparison to fCO2 from the Surface Ocean CO2 Atlas (SOCAT, https://www.socat.info/).

For more details, please see section 3.5.5, Figure B2, and Appendix C3.3 in Friedlingstein et al. (2022): Global Carbon Budget 2022, https://doi.org/10.5194/essd-14-4811-2022.

Script by Luke Gregor, ETH.

Repository maintained by Judith Hauck, AWI.

## Change log

- 2023-07-11: Luke Gregor
  - made it easier to change the SOCAT url.
  - Updated demo notebook to show how to perform analysis on only 1990-202X.
  - Improved logging to make dataset end year more explicit.

## Requirements

- Python 3.8 +
- numpy
- pandas
- xarray
- netCDF4
- matplotlib
- pooch (for downloading files)
- cartopy (for vizualisation, not required)
