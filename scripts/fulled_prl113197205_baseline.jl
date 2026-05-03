#!/usr/bin/env julia

using InfinitePE

const USAGE = """
Full ED baseline scan for qualitative comparison with PRL 113, 197205
(arXiv:1406.5415), e.g. supplementary-material Fig. 5 style finite-size
energy / specific-heat curves.

Usage:
  julia --project=. scripts/fulled_prl113197205_baseline.jl [options]

Options:
  --lattice honeycomb|hyperhoneycomb   Lattice family [honeycomb]
  --boundary TypeI|TypeII              Boundary condition [TypeI]
  --Lx INT                             Linear size in x [1]
  --Ly INT                             Linear size in y [1]
  --Lz INT                             Linear size in z for hyperhoneycomb [1]
  --tmin FLOAT                         Minimum temperature [0.1]
  --tmax FLOAT                         Maximum temperature [3.0]
  --ntemps INT                         Number of temperatures [32]
  --temps CSV                          Explicit temperatures, e.g. 0.1,0.2,0.5
  --Jx FLOAT                           Kitaev x coupling [1.0]
  --Jy FLOAT                           Kitaev y coupling [1.0]
  --Jz FLOAT                           Kitaev z coupling [1.0]
  --sign FLOAT                         Overall sign in H=sign*sum(Jσσ) [-1.0]
  --max-states INT                     Dense Hilbert-space guard [65536]
  --output PATH                        CSV output path [fulled_prl113197205_baseline.csv]
  --help                               Show this message

Output columns:
  temperature,beta,energy_per_site,energy2_per_site2,
  specific_heat_per_site,free_energy_per_site,entropy_per_site,
  nstates,nsites

This script performs full Hilbert-space exact diagonalization and is therefore
limited to small clusters. It is a comparison baseline, not a thermodynamic
limit calculation.
"""

function parse_args(args)
    opts = Dict{String,String}()
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--help" || arg == "-h"
            println(USAGE)
            exit(0)
        elseif startswith(arg, "--")
            key = arg[3:end]
            i == length(args) && throw(ArgumentError("missing value for option $arg"))
            value = args[i + 1]
            startswith(value, "--") && throw(ArgumentError("missing value for option $arg"))
            opts[key] = value
            i += 2
        else
            throw(ArgumentError("unexpected positional argument: $arg"))
        end
    end
    return opts
end

get_string(opts, key, default) = get(opts, key, default)
get_int(opts, key, default) = parse(Int, get(opts, key, string(default)))
get_float(opts, key, default) = parse(Float64, get(opts, key, string(default)))

function temperatures_from_options(opts)
    if haskey(opts, "temps")
        temps = [parse(Float64, strip(part)) for part in split(opts["temps"], ",") if !isempty(strip(part))]
        isempty(temps) && throw(ArgumentError("--temps must contain at least one temperature"))
        any(t -> !isfinite(t) || t <= 0, temps) &&
            throw(ArgumentError("all temperatures must be positive and finite"))
        return temps
    end

    tmin = get_float(opts, "tmin", 0.1)
    tmax = get_float(opts, "tmax", 3.0)
    ntemps = get_int(opts, "ntemps", 32)
    isfinite(tmin) && tmin > 0 || throw(ArgumentError("--tmin must be positive and finite"))
    isfinite(tmax) && tmax > 0 || throw(ArgumentError("--tmax must be positive and finite"))
    ntemps > 0 || throw(ArgumentError("--ntemps must be positive"))
    ntemps == 1 && return [tmin]
    return collect(range(tmin, tmax; length=ntemps))
end

function boundary_from_options(opts)
    boundary = get_string(opts, "boundary", "TypeI")
    if boundary == "TypeI"
        return TypeI()
    elseif boundary == "TypeII"
        return TypeII()
    end
    throw(ArgumentError("--boundary must be TypeI or TypeII; got $boundary"))
end

function lattice_from_options(opts)
    lattice = get_string(opts, "lattice", "honeycomb")
    Lx = get_int(opts, "Lx", 1)
    Ly = get_int(opts, "Ly", 1)
    Lz = get_int(opts, "Lz", 1)
    bc = boundary_from_options(opts)

    if lattice == "honeycomb"
        return generate_honeycomb(Lx, Ly, bc)
    elseif lattice == "hyperhoneycomb"
        return generate_hyperhoneycomb(Lx, Ly, Lz, bc)
    end
    throw(ArgumentError("--lattice must be honeycomb or hyperhoneycomb; got $lattice"))
end

function csv_escape(value)
    text = string(value)
    if occursin(",", text) || occursin("\"", text) || occursin("\n", text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function write_baseline_csv(path, scan)
    rows = FullED.comparison_table(scan; metadata=(baseline="PRL113197205",))
    open(path, "w") do io
        println(io, join((
            "temperature",
            "beta",
            "energy_per_site",
            "energy2_per_site2",
            "specific_heat_per_site",
            "free_energy_per_site",
            "entropy_per_site",
            "nstates",
            "nsites",
        ), ","))

        for row_data in rows
            row = (
                row_data.temperature,
                row_data.beta,
                row_data.energy_per_site,
                row_data.energy2_per_site2,
                row_data.specific_heat_per_site,
                row_data.free_energy_per_site,
                row_data.entropy_per_site,
                row_data.nsamples,
                row_data.nsites,
            )
            println(io, join(csv_escape.(row), ","))
        end
    end
    return path
end

function main(args)
    opts = parse_args(args)
    lat = lattice_from_options(opts)
    input = FullED.lattice_to_fulled(lat)
    temperatures = temperatures_from_options(opts)
    output = get_string(opts, "output", "fulled_prl113197205_baseline.csv")
    couplings = (
        x=get_float(opts, "Jx", 1.0),
        y=get_float(opts, "Jy", 1.0),
        z=get_float(opts, "Jz", 1.0),
    )
    sign = get_float(opts, "sign", -1.0)
    max_states = get_int(opts, "max-states", 65536)

    scan = FullED.scan_temperatures(input, temperatures; couplings=couplings, sign=sign, max_states=max_states)
    write_baseline_csv(output, scan)

    println("wrote Full ED baseline CSV: $output")
    println("lattice=$(lat.kind), dims=$(lat.dims), nsites=$(lat.nsites), nstates=$(length(scan.diag_result.eigenvalues))")
    println("temperatures=$(length(temperatures)), sign=$sign, couplings=$couplings")
    return scan
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        main(ARGS)
    catch err
        println(stderr, "error: ", err)
        println(stderr)
        println(stderr, USAGE)
        exit(1)
    end
end
