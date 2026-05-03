"""
    InfinitePE.EDMC

Exact diagonalization Monte Carlo (EDMC) scaffolding for `InfinitePE`.

This submodule intentionally contains only light-weight types, public function
signatures, and documentation. The physical Hamiltonian construction, diagonal
solver, Monte Carlo update rules, and measurements are left for later work.
"""
module EDMC

using LinearAlgebra
using Random

include("types.jl")
include("lattice_bridge.jl")
include("hamiltonian.jl")
include("diag.jl")
include("mc_update.jl")
include("measurements.jl")
include("driver.jl")

export AbstractEDMCModel, EDMCParameters, EDMCState, EDMCRunConfig
export KitaevBond, KitaevBondSet, Z2GaugeField, KitaevHamiltonianInput
export DiagonalizationResult, Z2BondFlipProposal, Z2BondFlipResult, EDMCDriverResult
export EDMCObservables, EDMCTemperatureScanResult
export lattice_to_edmc, extract_kitaev_bonds, initialize_z2_gauge
export build_hamiltonian, free_energy, diagonalize, majorana_energies
export flip_gauge, delta_free_energy, acceptance_probability
export propose_bond_flip, attempt_bond_flip, propose_update, accept_update
export acceptance_rate, warmup_acceptance_rate, sampling_acceptance_rate
export internal_energy, internal_energy_beta_derivative, measure, run_edmc, scan_temperatures
export comparison_row, comparison_table

end
