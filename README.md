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
    - `amenity_hbnw`: aggregates leisure, essential services, and HBNW commercial
    - `amenity_nhb`: aggregates NHB commercial and places of worship

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

General form: <br>
```math
T_{ij}^{(p)} = k_p \frac{O_i^{\alpha_p} D_j^{\beta_p}}{d_{ij}^{\gamma_p}}
```
<br>
Where:

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

**Boundary Effects and External Trip Leakage** <br>
The DIY administrative boundary does not represent a closed travel system. In early iterations, the absence of destinations outside the modeled area led to artificial concentration of long-distance trips at boundary-adjacent zones and excessive loading of radial corridors.

To mitigate this edge effect, the model introduces external trip leakage near the study-area boundary. Grid cells within a 3 km buffer of the boundary probabilistically leak a fraction of generated trips to an implicit external system. Leakage rates are purpose-specific, with higher leakage for Home-Based Work (HBW) trips and lower leakage for discretionary travel.

This approach does not model specific external destinations but serves as a boundary condition that prevents unrealistic accumulation of demand at the study edge while preserving internal spatial structure.

**Output** <br>
An Origin-Destination (OD) matrix containing estimated trip volumes between all grid pairs.

### D. Route Assignment & Network Handling
**Road Network** <br>
OpenStreetMap route network via ``osmnx``

**Routing** <br>
Each OD pair's trip volume is assigned to a path on the network using Dijkstra's shortest-path algorithm

**Initial Cost** <br> 
Free-flow generalized cost defined as road length multiplied by a mode- and road-class-specific constant. During routing, this base cost is augmented by stochastic perception noise and turn penalties as described above.

**Behavioral Route Choice Adjustments** <br>
- **Perception Noise (Stochastic Cost Perturbation)**<br>
    To avoid deterministic shortest-path collapse—where all trips select a single identical route due to perfectly perceived costs—the model introduces small stochastic perturbations to edge costs during routing.
    <br>
    For each origin–destination (OD) routing instance, base edge costs are perturbed as:

    ```math
    w_e' = w_e \left( 1 + \varepsilon_e \right), \qquad \varepsilon_e \sim \mathcal{U}(-\delta_m, +\delta_m) 
    ```

    Where:<br>
    - $w_e$ is the base edge cost (road-length multiplied by a mode and road-class-specific constant),
    - $\delta_m$ is a mode-specific perception noise parameter.

- **Turn Penalties at Intersections**<br>
    To better represent intersection delay and maneuvering friction, an additional cost is applied to turning movements during routing. When consecutive edges are non-collinear, a fixed turn penalty is added to the perceived travel cost.

    This discourages unrealistic zig-zag routing and helps preserve through-movement continuity on major arterials, where in reality drivers tend to remain on the same corridor unless a clear advantage exists. Turn penalties are mode-specific and applied during shortest-path computation.

**Assignment Method:** <br>
- **Destination sampling**<br>
To maintain computational tractability, the model does not route all possible origin–destination (OD) pairs. Instead, for each origin zone and distance band (near, medium, far), a limited number of destinations are selected for routing. <br><br>
Unlike a deterministic “top-k” approach, destinations are sampled probabilistically, with selection probabilities proportional to their gravity-model trip weights. This preserves the dominance of highly attractive destinations while avoiding winner-take-all artifacts in sparse regional contexts.<br><br>
The number of sampled destinations per distance band is fixed:
    - Near: 12
    - Medium: 8
    - Far: 4

- **Mode-specific routing**: Cars and motorbikes are routed separately with different road type preferences
- **Static assignment**: Initial routing does not incorporate congestion feedback (see section F for congestion iteration)

**Destination Sampling Strategy**<br>


### E. Congestion Modeling
Congestion is modeled using a Bureau of Public Roads (BPR) function: <br>
```math
t = t_0 \left( 1 + \alpha \left( \frac{v}{c} \right)^{\beta} \right)
```
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


### F. Congestion Feedback Loop
The model implements a static iterative assignment, introducing feedback between congestion and routing:
1. Assign OD flows to the network
2. Compute congestion using BPR
3. Update edge travel times
4. Re-route OD flows using updated costs
5. Repeat for 5–10 iterations until changes stabilize

To achieve a stable user equilibrium and prevent oscillations between iterations, the model implements the standard Method of Successive Averages (MSA)

### G. Mode Choice & Multi-Class Traffic Assignment
To better reflect observed travel behavior in Yogyakarta, the model extends beyond a single homogeneous traffic stream by explicitly representing mode choice and multi-class traffic flow, focusing on cars and motorbikes (with public transport planned for future iterations).

#### G.1 Purpose-Preserving Mode Choice
Rather than collapsing all trips into a single OD matrix prior to assignment, trip purposes are preserved through the mode choice stage:
- Home-Based Work (HBW)
- Home-Based Non-Work (HBNW)
- Non-Home-Based (NHB)
For each purpose-specific OD matrix, trips are split into modes based primarily on trip distance, reflecting empirically observed behavior in Indonesian cities (e.g., motorbike dominance for short–medium trips, higher car usage for longer HBW trips). <br>

After mode splitting, OD matrices are recombined by mode:
- `OD_car`
- `OD_motorbike`

Trip purposes are not carried further into the assignment stage.

#### G.2 Destination Sampling with Distance Stratification
To maintain computational tractability, OD assignment does not load all destination pairs. However, rather than truncating destinations solely by trip volume, the model applies a distance-stratified destination sampling scheme:

For each origin and mode:
- Destinations are divided into three distance bands:
    - Near (≤ ~3 km)
    - Medium (~3–8 km)
    - Far (> ~8 km)
- Within each band, destinations are ranked by OD flow magnitude 
- A fixed number of destinations is retained per band (e.g., 10 per band)

#### G.3 Mode-Specific Network Accessibility
Cars and motorbikes are routed on the same street network, but with mode-specific road type weights: <br>

- Cars are preferentially routed onto higher-class roads (primary, secondary)
- Motorbikes are less constrained by road hierarchy and can utilize narrower streets and local shortcuts unavailable or unattractive to cars

#### G.4 Shared Congestion with Passenger Car Units (PCU)
Congestion is modeled as a shared physical state of the roadway, rather than separately per mode. <br>
Traffic volumes are converted into an effective flow using Passenger Car Units (PCU): <br>
```math
v_{\text{eff}} = 1.0 \cdot v_{\text{car}} + \phi \cdot v_{\text{motorbike}}
```
<br>
Where:

- $v_{\text{eff}}$ : effective traffic volume used for congestion calculation (in Passenger Car Units)
- $v_{\text{car}}$ : assigned car traffic volume on a network segment
- $v_{\text{motorbike}}$ : assigned motorbike traffic volume on a network segment
- $\phi$ : motorbike passenger car unit (PCU) factor, representing relative road space consumption compared to a car

#### G.5 Mode-Specific Congestion Response 
Congested travel times are updated using BPR-style functions with mode-specific parameters, allowing motorbikes to remain competitive even under high traffic volumes.

```math
t_m = t_0 \left( 1 + \alpha_m \left( \frac{v_{\text{eff}}}{c} \right)^{\beta_m} \right)
```
<br>
Where:

- $t_m$ : congested travel time experienced by mode $m$
- $t_0$ : free-flow travel time
- $c$ : effective road capacity
- $\alpha_m, \beta_m$ : mode-specific congestion sensitivity parameters
While congestion is computed from the shared effective flow, experienced travel times differ by mode:
- Cars experience full congestion effects
- Motorbikes experience reduced delay due to filtering, queue bypassing, and maneuverability 

#### G.6 Iterative Multi-Class Assignment
The assignment process is implemented as a static iterative loop:
1. Route car OD flows using car travel times
2. Route motorbike OD flows using motorbike travel times
3. Aggregate flows into effective congestion volumes
4. Update mode-specific travel times
5. Repeat until convergence

## Results & Visualization
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

BPS "Provinsi Daerah Istimewa Yogyakarta Dalam Angka 2025"