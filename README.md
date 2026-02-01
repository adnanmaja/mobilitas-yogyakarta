# Simulating Urban Traffic Flow and Congestion in Daerah Istimewa Yogyakarta (DIY)
## Overview & Motivation
I was lying in bed during winter break, watching random YouTube videos, when a vlog about Tokyo caught my attention. What stood out wasn’t the city itself, but how the creator moved through it almost entirely by train. That made me wonder what daily mobility would look like if my hometown, Yogyakarta, had similarly accessible public transportation.

As I fell into urban planning content, a simple question stuck with me: instead of just wondering, why not try to model it myself? Before evaluating public transit, I needed to understand the baseline, which is road traffic. That curiosity became the starting point for this project.

Anyway, its a computational simulation modeling origin-destination traffic patterns, route assignment, and congestion levels across Yogyakarta's road network. The model generates visual heatmaps of traffic flow and congestion, which can hypothetically be used to identify chronic choke points and explore hypothetical transport or infrastructure scenarios before real-world implementation.

## Project Scope
This is a proof-of-concept, city-scale traffic simulation developed as a personal project. It is not intended for operational forecasting or policy evaluation, but as a sandbox for experimenting with:
- Land-use–driven trip generation
- Multi-purpose travel demand
- Gravity-based trip distribution
- Congestion-aware route assignment
- Feedback between demand, congestion, and travel cost

## Methodology & Pipeline
### A. Spatial Framework & Intensity Surfaces
The Special Region of Yogyakarta (DIY) is discretized into a 1 km × 1 km grid, balancing spatial detail with computational tractability. <br>
Each grid cell is characterized by continuous intensity measures:
- **Residential Intensity**<br>
Derived from WorldPop population raster data, representing trip production potential.
- **Employment Intensity**<br>
Approximated using a combination of:
    - GHSL built-up non-residential volume
    - VIIRS night-time light intensity
- **Amenity Intensity**<br>
Overture Maps Foundation POIs, disaggregated into purpose-specific layers: <br>
    - Residential-level intensities: leisure, essential services, and HBNW commercial (department stores, salons, barbershops)
    - Non-home-based intensities: NHB commercial (cafés, coffee shops) and places of worship<br>
    
    These are then grouped into two composite measures used downstream in trip distribution:
    - amenity_hbnw — aggregates leisure, essential services, and HBNW commercial
    - amenity_nhb — aggregates NHB commercial and places of worship

These layers form the spatial basis for trip generation across different trip purposes.

### B. Trip Purpose Segmentation & Temporal Weighting
Travel demand is decomposed into three OD matrices, each representing a distinct behavioral class:
1. **Home-Based Work (HBW)**<br>
Residential → employment-driven trips
2. **Home-Based Non-Work (HBNW)**<br>
Residential → amenity (hbnw) driven trips
3. **Non-Home-Based (NHB)**<br>
Trips not anchored at home, drawn to amenity_nhb destinations (NHB commercial, places of worship)

Each OD matrix is further adjusted using time-of-day weights, allowing demand composition to shift between peak and off-peak conditions. <br>
The final demand matrix is obtained by a weighted combination of all trip types.

### C. Trip Distribution: Purpose-Specific Gravity Models
Trips between grid cells `i` and `j` are generated using gravity models, with distinct distance-decay parameters per trip type. <br>

**General form:** <br>
$T_{ij}^{(p)} = k_p \frac{O_i^{\alpha_p} D_j^{\beta_p}}{d_{ij}^{\gamma_p}}$ 
<br>
**Where**:
- $T_{ij}^{(p)}$ = number of trips from zone $i$ to zone $j$ for trip purpose $p$
- $p \in {\text{HBW}, \text{HBNW}, \text{NHB}}$
- $O_i$ = origin intensity of zone $i$
- $D_j$ = destination intensity of zone $j$
- $d_{ij}$ = distance between zones $i$ and $j$
- $k_p$ = scaling constant for trip purpose $p$
- $\alpha_p$ = origin elasticity parameter
- $\beta_p$ = destination elasticity parameter
- $\gamma_p$ = distance decay parameter for trip purpose $p$
<br>

Distance-decay values are calibrated using findings from (Devi et al., 2019), reflecting observed differences in trip-length tolerance across purposes (e.g., work trips tolerate longer distances than discretionary trips).

**Output** <br>
An Origin-Destination (OD) matrix containing estimated trip volumes between all grid pairs.

### D. Route Assignment & Network Handling
- **Road Network** : OpenStreetMap route network via ``osmnx``
- **Routing** : Each OD pair's trip volume is assigned to a path on the network using Dijkstra's shortest-path algorithm
- **Initial Cost** : Free-flow travel time, adjusted by relative road class capacity
- **Assignment Method** : 
    - OD flows are distributed across multiple shortest paths (top-k routing)
    - Routing costs are updated iteratively as congestion evolves

### E. Congestion Modeling
Congestion is modeled using a Bureau of Public Roads (BPR) function: <br>
$t = t_0 \left( 1 + \alpha \left( \frac{v}{c} \right)^{\beta} \right)$
<br>
Where: <br>
- $t$ = congested travel time
- $t_0$ = free-flow travel time
- $v$ = traffic volume on the segment or corridor
- $c$ = effective capacity
- $v/c$ = volume-to-capacity ratio
- $\alpha, \beta$ = BPR calibration parameters

Key extensions:
- **Relative Capacity by Road Type**<br>
Capacities are defined in relative terms (e.g., primary > secondary > tertiary), rather than absolute veh/hr.
- **Utilization Ratios by Trip Type**<br>
Different trip purposes contribute differently to effective congestion.
- **Spatial Smoothing**<br>
Congestion is smoothed across neighboring edges to reduce artificial discontinuities and better represent spillback effects.
- **Corridor-Level Congestion**<br>
Congestion is aggregated and applied at corridor scale rather than strictly per-edge, improving network realism.

### F. Congestion Feedback Loop
The model implements a static iterative assignment, introducing feedback between congestion and routing:
1. Assign OD flows to the network
2. Compute congestion using BPR
3. Update edge travel times
4. Re-route OD flows using updated costs
5. Repeat for 5–10 iterations until changes stabilize

This captures first-order congestion feedback without full dynamic traffic simulation.


## Results & Vsiualization
The resulting data have been visualized into an interactive map using maplibre. [Check it out!](https://adnanmaja.github.io/mobilitas-yogyakarta) <br>
Additionally, static figures can be found at ```data/figures```

## Limitations & Assumptions
- **Calibration** : While distance-decay parameters are literature-informed, most other parameters remain heuristic.
- **Static  Demand** : No within-period demand dynamics or departure-time choice modeling.
- **Behaviour** : The model uses a simple user-equilibrium (all drivers choose the perceived shortest path). It does not account for driver learning, real-time information, or stochastic variations.
- **Capacity Representation**: Relative rather than measured capacities; no lane-level or signal modeling.
- **Grid & Data Resolution** : 1 km grid limits fine-grained neighborhood analysis.

## Technical Stack
- **Languages** : Python, Javascript
- **Core Libraries** : pandas, numpy, geopandas, osmnx, scipy
- **Data Sourcers**:
    - [WorldPop](https://www.worldpop.org/)
    - [GHSL](https://human-settlement.emergency.copernicus.eu/) ©  European Commission
    - [VIIRS Nighttime Lights (Earth Observation Group)](https://eogdata.mines.edu/products/vnl/)
    - [Overture Maps](https://overturemaps.org/)
    - [OpenStreetMap](https://www.openstreetmap.org/about)
- **Basemap & Rendering**: [MapLibre GL JS](https://maplibre.org/) with styles hosted by [MapTiler](https://www.maptiler.com/)<br>
 © MapTiler © OpenStreetMap contributors.

## Future Work
- More robust calibration using observed traffic or mobile-phone data
- Stochastic and multi-class route choice models
- Explicit public transport and mode choice integration
- Scenario testing for land-use or infrastructure changes
- Transition toward quasi-dynamic or mesoscopic simulation

## Acknowledgements
Devi, M. K., et al. (2019). Travel Behavior Pattern in Yogyakarta Urbanized Area. Proceedings of the Eastern Asia Society for Transportation Studies, 12.
https://www.researchgate.net/publication/338865315
