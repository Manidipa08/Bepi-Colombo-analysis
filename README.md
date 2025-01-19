# Bepi-Colombo-analysis
## Brief Overview 
This project simulates real data that will come from the BepiColombo satellite that is of interest for testing General Relativity in the solar system, that is the range observable. The MPO orbiter of BepiColombo should orbit Mercury for minimum 2 years (2026-2027)(we considered in our analysis). Then by having the Spacecraft-Mercury vector (used SPICE Kernel to get this vector), geometrically the Earth-Mercury distance can be estimated.

In this analysis, the aim is to test General Relativity at astronomical scales, so we are interested in how planets, that are in constant free-fall, are affected by gravitational potentials in the solar system. Further, graviton is also introduced (refer to Bernus et al. 2019{https://inspirehep.net/literature/1713598}) to the PPN formalism to check for the effect on range observable compared to GR. In this notebook, Newtonian gravity, GR from reboundx, PPN formalism, additional acceleration to PPN due to graviton, sun's J2 effect were introduced. 

## Steps
To see the impact on Earth-Mercury Distance:

1. Newtonian Gravity using REBOUND, integrating the impact of all 8 planets, 3 big asteroids (Ceres, Pallas, Vesta), the MPO, the Sun
2. Integrator = WHfast
3. Start Date = 2026-11-01 00:00 UTC(when MPO orbit is stable around Mercury)
4. Simulation’s unit = ('days', 'km', 'kg')
5. Observer at Solar System Barycenter
6. Initial state vectors obtained from bc_plan_v430_20241120_001.tm meta-kernel from SPICE. 
7. REBOUNDx to add GR, Sun’s J2 effect
8. Shapiro delay defined
9. Later on, additional accelerations added to the simulation using REBOUNDx (PPN, Graviton)
10. This simulation will compute theoretical observable(Earth-Mercury distance) and Doppler measurement
11. Covariance matrix analysis to estimate expected accuracy on the constraint parameters
12. Partial derivative matrix, J = δO/δP (O is the observable and P are the parameters: cartesian coordinates, velocities, β, γ, λg.)
13. Covariance matrix, Cov(P) = (J<sup>T</sup>W J) <sup>(-1)</sup>
14. Diagonal of the Cov(P) represents the covariance vector of the fitted parameters, used to estimate σ(standard deviation)




## Disclaimer
This simulation is not a finalized version and is solely for research purposes. If you see any potential issues, feel free to reach out at <a href="mailto:manidipabanerjee38@gmail.com"> manidipabanerjee38@gmail.com</a>

