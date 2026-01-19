# Simulating Urban Traffic Flow and Congestion in Daerah Istimewa Yogyakarta (DIY)
## Overview & Motivation
I was lying in bed during winter break, watching random YouTube videos, when a vlog about Tokyo caught my attention. What stood out wasn’t the city itself, but how the creator moved through it almost entirely by train. That made me wonder what daily mobility would look like if my hometown, Yogyakarta, had similarly accessible public transportation.

As I fell into urban planning content, a simple question stuck with me: instead of just wondering, why not try to model it myself? Before evaluating public transit, I needed to understand the baseline, which is road traffic. That curiosity became the starting point for this project.

Anyway, its a computational simulation modeling origin-destination traffic patterns, route assignment, and congestion levels across Yogyakarta's road network. The model generates visual heatmaps of traffic flow and congestion, which can hypothetically be used to identify chronic choke points and explore hypothetical transport or infrastructure scenarios before real-world implementation.

## Project Scope
This project is an exploratory, proof-of-concept simulation developed as a personal winter break project. The goal is not predictive accuracy, but to understand how population distribution, destinations, and road networks interact to produce congestion patterns at a city scale. 

## Methodology & Pipeline
### A. Data Preparation & Gridding
The region, Special Region of Yogyakarta was divided into a grid of 1km2. This resolution was chosen as a compromise between spatial detail (capturing neighborhood-level variations) and computational feasibility for a desktop-based simulation.
- **Origin points (O)** : Origin points for each grid was derived 70% from WorldPop population raster (100m resolution), representing trip origins from residential areas. The other 30% are derived from OpenStreetMap (OSM) road data (excluding major highways), approximating trips originating from local streets
- **Destination points (D)** : A composite score from: OSM point-of-interest tags (45%), road density (20%), intersections (15%), and manually added special locations e.g., Malioboro (20%). Here, i defined three distinct attractiveness scenarios: 
    * Peak Rush Hour: Weights biased towards commercial/business/education tags.
    * Chill Hour: Weights biased towards residential/local amenities.
    * Weekends: Weights biased towards recreational/tourist/shopping tags.

### B. Trip Generation & Distribution
Trips between an origin grid `i` and a destination grid `j` are calculated based on the attractiveness of `j` and the impedance (discouraging effect) of distance.
- **Formula** : ``T_ij = k * (O_i^α * D_j^β) / (d_ij^γ)``
- **Parameters** : ``α=1.0, β=1.0, γ=2.0`` These parameters control the sensitivity to origin mass, destination mass, and distance decay. The values were chosen as standard defaults for an initial proof-of-concept simulation, following common practice in rudimentary gravity models.
- **Output** :  An Origin-Destination (OD) matrix containing estimated trip volumes between all grid pairs.

### C. Route Assignment & Network Handling
- **Network** : OpenStreetMap route network via ``osmnx``
- **Routing** : Each OD pair's trip volume is assigned to a path on the network using Dijkstra's shortest-path algorithm
- **Cost Function** : The algorithm prioritizes higher-capacity roads (e.g., primary > secondary > tertiary) by assigning them lower traversal costs, encouraging logical route choices. OD demands are split accross top-k shortest path (k=30)
- **Output** : Aggregate flow volume for every edge (road segment) in the network.

### D. Congestion Modeling
well its pretty much is this set of concepts, of which i cant explain:
- After assigning trips to the road network, each road segment accumulates a total traffic flow. Since real-world road capacity data is unavailable, the model assumes that the observed flow on each segment represents a proxy for near-capacity conditions.
- Traffic congestion is estimated using a volume-to-capacity style ratio, where higher relative flow implies higher congestion. This ratio is normalized across the network and mapped to discrete congestion levels for visualization.
- To better reflect real-world variability such as driver behavior, traffic signals, and unmodeled disruptions, controlled random noise is added to the congestion values. This prevents unrealistically uniform patterns and produces more organic-looking congestion distributions.

## Results & Vsiualization
The resulting data have been visualized into an interactive map using mapbox. Here's the link: didntexistyet.com <br>
Additionally, figures can be found at ```data/figures```

## Limitations & Assumptions
- **Parameters** : The gravity model parameters ``(α, β, γ)``, attractiveness weights, and road capacity values are uncalibrated estimates. A real application would require calibration against sensor or survey data.
- **Demand** : Trip generation is static and based on population/POI density, not dynamic time-of-day demands.
- **Behaviour** : The model uses a simple user-equilibrium (all drivers choose the perceived shortest path). It does not account for driver learning, real-time information, or stochastic variations.
- **Grid & Data Resolution** : The 1km grid and OSM data completeness impose a limit on spatial precision.

## Technical Implementations
- **Languages & Libraries** : Python, pandas, numpy, geopandas, osmnx, scipy
- **Data Sources** : WorldPop, OpenStreetMap.

## Future Work
- Improving congestion models with feedback loop
- Better routing with more random and organic route choices
- Determine the impact of the current existing transit routes
- Explore future public transit scenario potentials
