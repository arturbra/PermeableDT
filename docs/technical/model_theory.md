# Model Theory and Mathematical Background

This document provides the mathematical foundation and theoretical background for the permeabledt permeable pavement model.

## Table of Contents

1. [Model Overview](#model-overview)
2. [Three-Zone Conceptual Model](#three-zone-conceptual-model)
3. [Mathematical Formulation](#mathematical-formulation)
4. [Governing Equations](#governing-equations)
5. [Numerical Implementation](#numerical-implementation)
6. [Parameter Definitions](#parameter-definitions)
7. [Water Balance](#water-balance)
8. [Model Assumptions](#model-assumptions)
9. [References](#references)

## Model Overview

The PermeableDT model simulates water flow through permeable pavement systems using a physically-based, three-zone approach. The model captures the key hydrological processes in green infrastructure systems while maintaining computational efficiency for real-time applications.
The model is based on the MicroPollutants In Rain GardEn (MPiRe) developed by Randelovic et. al (2016):

### Reference:
Randelovic, A., Zhang, K., Jacimovic, N., McCarthy, D., Deletic, A., 2016. Stormwater biofilter treatment model (MPiRe) for selected micropollutants. Water Research 89, 180–191. doi:10.1016/j.watres.2015.11.046.

### Key Features
- **Three-zone conceptual model**: Ponding, unsaturated, and saturated zones
- **Process-based simulation**: Physical equations for each zone
- **Water balance conservation**: Mass conservation throughout the system
- **Variable time stepping**: Adaptive or fixed time steps
- **Uncertainty quantification**: Compatible with particle filtering

## Three-Zone Conceptual Model

The permeable pavement system is represented as three vertically-stacked zones:

```
┌─────────────────────────────────┐
│         PONDING ZONE            │  ← Surface water storage
│  - Surface storage              │
│  - Weir overflow                │
│  - Infiltration to below        │
├─────────────────────────────────┤
│      UNSATURATED ZONE           │  ← Vadose zone
│  - Moisture storage             │
│  - Evapotranspiration           │
│  - Unsaturated flow             │
├─────────────────────────────────┤
│       SATURATED ZONE            │  ← Groundwater zone
│  - Groundwater storage          │
│  - Pipe drainage                │
│  - Lateral infiltration         │
└─────────────────────────────────┘
```

### Zone Characteristics

#### Ponding Zone (Surface)
- **Function**: Surface water storage and overflow
- **Key Processes**: Rainfall input, evaporation, overflow, infiltration
- **Primary Equation**: Water balance with weir overflow

#### Unsaturated Zone (Vadose)
- **Function**: Moisture storage and transmission
- **Key Processes**: Evapotranspiration, vertical flow, moisture storage
- **Primary Equation**: Richards' equation (simplified)

#### Saturated Zone (Groundwater)
- **Function**: Groundwater storage and drainage
- **Key Processes**: Pipe drainage, lateral infiltration, storage
- **Primary Equation**: Water balance with pipe flow

## Mathematical Formulation

### 1. Ponding Zone

The ponding zone water balance is given by:

```
dVp/dt = Qin + Qrain - Qet - Qover - Qinfp
```

Where:
- **Vp** = Ponding volume [m³]
- **Qin** = External inflow [m³/s]
- **Qrain** = Rainfall input [m³/s]
- **Qet** = Evaporation from ponding [m³/s]
- **Qover** = Overflow [m³/s]
- **Qinfp** = Infiltration from ponding [m³/s]

#### Overflow Calculation

Overflow occurs when ponding depth exceeds the overflow height:

```
Qover = Kweir × wWeir × (hp - Hover)^expWeir    if hp > Hover
Qover = 0                                       if hp ≤ Hover
```

Where:
- **hp** = Ponding depth [m]
- **Hover** = Overflow height [m]
- **Kweir** = Weir coefficient [-]
- **wWeir** = Weir width [m]
- **expWeir** = Weir exponent [-]

#### Infiltration from Ponding

```
Qinfp = Kf × A × (Cs + Pp × flagp) × hp^0.5
```

Where:
- **Kf** = Hydraulic conductivity of surrounding soil [m/s]
- **A** = Bottom area [m²]
- **Cs** = Side flow coefficient [-]
- **Pp** = Unlined perimeter [m]
- **flagp** = Lining flag (0 if lined, 1 if unlined) [-]

### 2. Unsaturated Zone

The unsaturated zone follows a simplified Richards' equation approach:

```
nusz × husz × ds/dt = Qinfp - Qet - Qhc - Qfs - Qpf
```

Where:
- **nusz** = Unsaturated zone porosity [-]
- **husz** = Unsaturated zone depth [m]
- **s** = Soil moisture content [-]
- **Qhc** = Moisture loss due to plant uptake [m³/s]
- **Qfs** = Flow to saturated zone [m³/s]
- **Qpf** = Lateral flow from unsaturated zone [m³/s]

#### Evapotranspiration

```
Qet = Kc × Emax × A × f(s)
```

Where f(s) is the soil moisture stress function:

```
f(s) = 0                           if s ≤ sh
f(s) = (s - sh)/(sw - sh)         if sh < s ≤ sw
f(s) = 1                           if sw < s ≤ sfc
f(s) = (ss - s)/(ss - sfc)        if sfc < s ≤ ss
f(s) = 0                           if s > ss
```

Where:
- **sh** = Hygroscopic point [-]
- **sw** = Wilting point [-]
- **sfc** = Field capacity [-]
- **ss** = Soil saturation point [-]

#### Unsaturated Flow

```
Qfs = A × Ks × (s/sfc)^gama × ((husz + hsz)/husz)
```

Where:
- **Ks** = Saturated hydraulic conductivity [m/s]
- **gama** = Pore-size distribution parameter [-]
- **hsz** = Saturated zone depth [m]

### 3. Saturated Zone

The saturated zone water balance:

```
nsz × A × dhsz/dt = Qfs - Qinfsz - Qpipe
```

Where:
- **nsz** = Saturated zone porosity [-]
- **Qinfsz** = Lateral infiltration from saturated zone [m³/s]
- **Qpipe** = Pipe drainage [m³/s]

#### Pipe Drainage

The pipe drainage follows an orifice/weir equation:

```
Qpipe = Cd × Apipe × (2g × heff)^0.5 × heff^eta
```

Where:
- **Cd** = Discharge coefficient [-]
- **Apipe** = Pipe cross-sectional area [m²]
- **g** = Gravitational acceleration [m/s²]
- **heff** = Effective head above pipe [m]
- **eta** = Drainage coefficient (0-1) [-]

The effective head is calculated as:

```
heff = max(0, hpipe + hsz - hpipe)
```

#### Lateral Infiltration

```
Qinfsz = Kf × A × (Cs + Psz × flagsz) × hsz^0.5
```

## Governing Equations

### State Variables

The model has three primary state variables:
1. **hp** - Ponding depth [m]
2. **s** - Soil moisture content [-]
3. **hsz** - Saturated zone depth [m]

### Derived Variables

Several variables are computed from the state variables:
- **husz** = L - hsz (Unsaturated zone depth)
- **nsz** = f(hsz) (Saturated zone porosity)
- **nusz** = f(husz, hsz) (Unsaturated zone porosity)

### Porosity Functions

```
nsz = cnsz(hsz, L, Dtl, Dg, nf, nt, ng)
nusz = cnusz(husz, hsz, nusz_ini, ng, Dg, Df)
```

These functions account for the layered structure of the pavement system.

## Numerical Implementation

### Time Stepping

The model uses explicit time stepping with adaptive or fixed time steps:

```
x(t+Δt) = x(t) + Δt × dx/dt
```

### Stability Conditions

For numerical stability, the time step must satisfy:

```
Δt ≤ min(Δt_diffusion, Δt_convection, Δt_drainage)
```

### Mass Conservation

At each time step, mass conservation is enforced:

```
Mass_in - Mass_out = ΔStorage
```

### Boundary Conditions

- **Upper boundary**: Specified rainfall flux
- **Lower boundary**: Free drainage or specified head
- **Lateral boundaries**: No-flow or specified infiltration

## Parameter Definitions

### Physical Parameters

| Parameter | Symbol | Units | Description |
|-----------|---------|-------|-------------|
| Filter depth | Df | m | Depth of filter media layer |
| Transition depth | Dtl | m | Depth of transition layer |
| Gravel depth | Dg | m | Depth of gravel layer |
| Total depth | L | m | Total depth (Df + Dtl + Dg) |
| Bottom area | A | m | Cross-sectional area |
| Ponding area | Ap | m | Surface ponding area |

### Hydraulic Parameters

| Parameter | Symbol | Units | Description |
|-----------|---------|-------|-------------|
| Saturated K | Ks | m/s | Saturated hydraulic conductivity |
| Filter porosity | nf | - | Porosity of filter media |
| Transition porosity | nt | - | Porosity of transition layer |
| Gravel porosity | ng | - | Porosity of gravel layer |
| Pore parameter | gama | - | Pore-size distribution parameter |

### Moisture Parameters

| Parameter | Symbol | Units | Description |
|-----------|---------|-------|-------------|
| Hygroscopic point | sh | - | Hygroscopic moisture content |
| Wilting point | sw | - | Wilting point moisture content |
| Field capacity | sfc | - | Field capacity moisture content |
| Saturation point | ss | - | Saturation moisture content |

### Drainage Parameters

| Parameter | Symbol | Units | Description |
|-----------|---------|-------|-------------|
| Pipe height | hpipe | m | Height of drainage pipe inlet |
| Pipe diameter | dpipe | mm | Diameter of drainage pipe |
| Discharge coefficient | Cd | - | Pipe discharge coefficient |
| Drainage coefficient | eta | - | Drainage efficiency parameter |

### Overflow Parameters

| Parameter | Symbol | Units | Description |
|-----------|---------|-------|-------------|
| Overflow height | Hover | m | Height at which overflow begins |
| Weir coefficient | Kweir | - | Weir discharge coefficient |
| Weir width | wWeir | m | Effective width of overflow weir |
| Weir exponent | expWeir | - | Weir equation exponent |

## Water Balance

### System Water Balance

The overall water balance for the system is:

```
Rainfall = Evapotranspiration + Overflow + Pipe_Drainage + Lateral_Infiltration + ΔStorage
```

### Storage Components

Total storage includes:
- **Ponding storage**: hp × Ap
- **Unsaturated storage**: s × nusz × husz × A
- **Saturated storage**: nsz × hsz × A

### Water Balance Error

The water balance error is calculated as:

```
Error = |Input - Output - ΔStorage| / Input × 100%
```

## Model Assumptions

### Physical Assumptions

1. **One-dimensional flow**: Vertical flow dominates horizontal flow
2. **Homogeneous layers**: Each layer has uniform properties
3. **Rigid porous media**: No deformation or consolidation
4. **Isothermal conditions**: Temperature effects neglected
5. **Single-phase flow**: Only liquid water considered

### Numerical Assumptions

1. **Explicit time stepping**: Simple forward Euler integration
2. **Lumped parameters**: Spatially-averaged properties
3. **Quasi-steady infiltration**: Rapid equilibration assumption
4. **Linear interpolation**: Between measured points

### Simplifications

1. **Simplified Richards equation**: Reduces computational complexity
2. **Empirical relationships**: For unsaturated hydraulic conductivity
3. **Lumped evapotranspiration**: Spatially-averaged ET rates
4. **Constant porosity**: Within each layer

## Model Validation

### Calibration Parameters

Typically calibrated parameters include:
- Saturated hydraulic conductivity (Ks)
- Moisture retention parameters (sw, sfc, ss)
- Drainage parameters (Cd, eta, hpipe)
- Pore-size parameter (gama)

### Performance Metrics

Model performance is evaluated using:
- Nash-Sutcliffe Efficiency (NSE)
- Root Mean Square Error (RMSE)
- Peak flow error
- Volume error
- Water balance error

## Mathematical Notation

### Units
- **[m]**: Meters (length)
- **[m²]**: Square meters (area)
- **[m³]**: Cubic meters (volume)
- **[m³/s]**: Cubic meters per second (flow rate)
- **[m/s]**: Meters per second (velocity)
- **[-]**: Dimensionless
- **[s]**: Seconds (time)

### Coordinate System
- **z**: Vertical coordinate (positive upward)
- **t**: Time coordinate
- 
This mathematical foundation provides the basis for all permeabledt simulations and enables the model to capture the essential physics of permeable pavement systems while maintaining computational efficiency for real-time applications.