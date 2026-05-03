"""
    InfinitePE.FullED

Small-cluster full exact diagonalization baseline for Kitaev spin models.

This module works in the full spin-1/2 Hilbert space and is intended for
finite-size comparison tables against EDMC or Infinite PE workflows.
"""
module FullED

using LinearAlgebra

import ..EDMC: extract_kitaev_bonds, KitaevBondSet

include("types.jl")
include("hamiltonian.jl")
include("diag.jl")
include("measurements.jl")
include("driver.jl")

export FullEDHamiltonianInput, FullEDDiagonalizationResult, FullEDObservables
export FullEDTemperatureScanResult
export lattice_to_fulled, build_hamiltonian, diagonalize, spectrum
export partition_function, free_energy, thermal_observables, scan_temperatures
export comparison_row, comparison_table

end
