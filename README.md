# CREST-S: Channel-Refined Water Classification for River Width Extraction from Sentinel-2
 
Code and data supporting:
 
> Khatri, S., Feng, D., & Gupta, A. (2026). *Accurate River Width Extraction from Sentinel-2 Using CREST: A Channel-Refined Classification Framework with Snow and Turbidity Screening.
 
## Overview
 
River width is a key input to hydrological and land-surface models, but satellite-based width extraction is confounded by land covers that spectrally resemble water — snow, turbid sediment, and dense riparian vegetation. This repository accompanies a study that benchmarks four Sentinel-2-compatible water classification algorithms (Zou multi-condition classifier, WI2015, MuWI, and an ultra-blue MNDWI variant), each with and without a snow-and-sediment (SS) environmental guard, against independent river width references.
 
**Key result:** the Zou classifier combined with the SS guard (CREST-S(Zou)) achieved the best overall performance (NSE = 0.504, R² = 0.594, RMSE = 117.9 m) against USGS gauge widths for channels ≥30 m across the conterminous United States (2017–2024), validated against 655,593 field measurements at 5,750 stations, with a secondary check against the RiverScope PlanetScope-derived benchmark.

## Data Sources
 
- **USGS streamflow gauge measurements** — in-situ channel width, U.S. Geological Survey
- **MERIT Hydro** — hydrologically conditioned DEM and river centerline network (Yamazaki et al. 2019)
- **GRWL** — Global River Widths from Landsat, centerline geometry and transect angles
- **RiverScope** — PlanetScope-derived river masking benchmark (Daroya et al. 2025)
- **GloRivSed** — suspended sediment concentration estimates (Prajapati et al. 2026)
- **Sentinel-2 L2A** — Copernicus surface reflectance imagery, via Google Earth Engine

## Contact
 
Suraj Khatri — khatrisj@mail.uc.edu
Department of Chemical and Environmental Engineering, University of Cincinnati
