# The Angiogenic Growth Of Cities

This repository contains code and data associated with the paper:

Capel-Timms, I. et al. (2024) The Angiogenic Growth of Cities

## Data
All data needed are provided in the ./Data directory for the respective case studies. 

All historical and projected residential population data and respective GIS shapefiles were retrieved at the finest possible spatial unit for Greater London (Parish-level, Enumeration District or Output Area) (1 - 5) and Metropolitan Sydney (Mesh Block) (6). These were converted from the irregular administrative units to a 1 km $\times$ 1 km on the `EPSG:27700` (British National Grid) `EPSG:8058` (NSW Lambert) coordinate reference systems using spatial overlays via the `geopandas (v 0.9.0)` package. The observed data spatial units are coarser for earlier years than for later years. Data used for creating the Greater London observed network are from Levinson (7) and Brown (8), whilst the Sydney tram (9) and train (10) networks were collated by (11). These include the location and opening year of stations and the opening year of the connections between them. Node accessibility $A_{v}$ (12) is calculated during pre-processing. Observed and simulated networks are handled as `networkx (v 2.4) Graph` objects. All edges *E* are assumed to be bi-directional, and changes between subnetworks or lines (e.g. National Rail, Underground) are assumed to be instant (i.e. no travel time in between). The *V* and *E* within the `networkx` graph are converted to vector objects then to raster objects for use within the model.

### Pre-processing of population data
Pre-processing of population data generally had to be done on a case-by-case basis, but `irreg_to_reg.py` provides a generic script for translating irregular population data to a regular grid. 

### Pre-processing of network data
Original network was provided as SHP files by (7, 10), with additional data from (8) for the London case. The network SHP files comprised railline and rail station data, with each attribute detailing the following:
*Railines*
The complex "real-life" network as reduced to single straight lines between each station, with ends snapped (manually) to the stations. The following attributes are necessary:
- Line ID
- IDs/names of stations *i* and *j* at each end of the line
- Opening year of line
- Closing year of line

*Rail stations*
- Station ID
- Station name
- Opening year of station
- Closing year of station



1. GreatBritainHistoricalGISProject2017
2. ONS17
3.  Satchell2018
4.   UKDataService2018
5.   Casweb
6.   AustralianBureauofStatistics2016
7.   Levinson2008
8.   Brown2012
9.   Keenan1979
10.   LahoorpoorLevinson2019
11.   Lahoorpoor2019
12.   Wang2009
