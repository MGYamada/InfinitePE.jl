"""
    scan_temperatures(input, temperatures; kwargs...)

Diagonalize once and compute exact thermal observables for all temperatures.
"""
function scan_temperatures(input::FullEDHamiltonianInput, temperatures::AbstractVector{<:Real}; kwargs...)
    isempty(temperatures) && throw(ArgumentError("temperature scan requires at least one temperature"))
    all(t -> isfinite(t) && t > 0, temperatures) ||
        throw(ArgumentError("all scan temperatures must be positive and finite"))

    result = spectrum(input; kwargs...)
    observables = FullEDObservables[
        thermal_observables(result, inv(Float64(temperature)), input.bondset.nsites) for temperature in temperatures
    ]
    return FullEDTemperatureScanResult(temperatures, observables, result)
end
