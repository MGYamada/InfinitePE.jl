#!/usr/bin/env julia

using InfinitePE

const USAGE = """
EDMC baseline scan for qualitative comparison with PRL 113, 197205.

Usage:
  julia --project=. scripts/edmc_prl113197205_baseline.jl [options]

Options:
  --lattice honeycomb|hyperhoneycomb   Lattice family [honeycomb]
  --boundary TypeI|TypeII              Boundary condition [TypeI]
  --Lx INT                             Linear size in x [2]
  --Ly INT                             Linear size in y [2]
  --Lz INT                             Linear size in z for hyperhoneycomb [1]
  --warmup INT                         Warmup sweeps per temperature [2]
  --sampling INT                       Sampling sweeps per temperature [4]
  --seed INT                           RNG seed [113197205]
  --tmin FLOAT                         Minimum temperature [0.1]
  --tmax FLOAT                         Maximum temperature [3.0]
  --ntemps INT                         Number of temperatures [16]
  --temps CSV                          Explicit temperatures, e.g. 0.1,0.2,0.5
  --Jx FLOAT                           Kitaev x coupling [1.0]
  --Jy FLOAT                           Kitaev y coupling [1.0]
  --Jz FLOAT                           Kitaev z coupling [1.0]
  --output PATH                        CSV output path [edmc_prl113197205_baseline.csv]
  --help                               Show this message

Output columns:
  temperature,beta,energy_per_site,energy2_per_site2,
  energy_beta_derivative_per_site,specific_heat_per_site,
  acceptance_rate,warmup_acceptance_rate,sampling_acceptance_rate,
  nsamples,accepted,attempted

This script is a small-size EDMC baseline workflow. It is intended for
qualitative trend checks, such as peak position and shape, not for claiming
quantitative agreement with the thermodynamic-limit results of the paper.
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

function get_string(opts, key, default)
    return get(opts, key, default)
end

function get_int(opts, key, default)
    return parse(Int, get(opts, key, string(default)))
end

function get_float(opts, key, default)
    return parse(Float64, get(opts, key, string(default)))
end

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
    ntemps = get_int(opts, "ntemps", 16)
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
    else
        throw(ArgumentError("--boundary must be TypeI or TypeII; got $boundary"))
    end
end

function lattice_from_options(opts)
    lattice = get_string(opts, "lattice", "honeycomb")
    Lx = get_int(opts, "Lx", 2)
    Ly = get_int(opts, "Ly", 2)
    Lz = get_int(opts, "Lz", 1)
    bc = boundary_from_options(opts)

    if lattice == "honeycomb"
        return generate_honeycomb(Lx, Ly, bc)
    elseif lattice == "hyperhoneycomb"
        return generate_hyperhoneycomb(Lx, Ly, Lz, bc)
    else
        throw(ArgumentError("--lattice must be honeycomb or hyperhoneycomb; got $lattice"))
    end
end

function csv_escape(value)
    text = string(value)
    if occursin(",", text) || occursin("\"", text) || occursin("\n", text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function write_baseline_csv(path, scan)
    rows = EDMC.comparison_table(scan; metadata=(baseline="PRL113197205",))
    open(path, "w") do io
        println(io, join((
            "temperature",
            "beta",
            "energy_per_site",
            "energy2_per_site2",
            "energy_beta_derivative_per_site",
            "specific_heat_per_site",
            "acceptance_rate",
            "warmup_acceptance_rate",
            "sampling_acceptance_rate",
            "nsamples",
            "accepted",
            "attempted",
        ), ","))

        for (row_data, run) in zip(rows, scan.runs)
            row = (
                row_data.temperature,
                row_data.beta,
                row_data.energy_per_site,
                row_data.energy2_per_site2,
                row_data.energy_beta_derivative_per_site,
                row_data.specific_heat_per_site,
                EDMC.acceptance_rate(run),
                EDMC.warmup_acceptance_rate(run),
                EDMC.sampling_acceptance_rate(run),
                row_data.nsamples,
                run.accepted,
                run.attempted,
            )
            println(io, join(csv_escape.(row), ","))
        end
    end
    return path
end

function main(args)
    opts = parse_args(args)
    lat = lattice_from_options(opts)
    input = EDMC.lattice_to_edmc(lat)
    temperatures = temperatures_from_options(opts)
    warmup = get_int(opts, "warmup", 2)
    sampling = get_int(opts, "sampling", 4)
    seed = get_int(opts, "seed", 113197205)
    output = get_string(opts, "output", "edmc_prl113197205_baseline.csv")
    couplings = (
        x=get_float(opts, "Jx", 1.0),
        y=get_float(opts, "Jy", 1.0),
        z=get_float(opts, "Jz", 1.0),
    )

    scan = EDMC.scan_temperatures(
        input,
        temperatures;
        warmup_sweeps=warmup,
        sampling_sweeps=sampling,
        seed=seed,
        couplings=couplings,
    )
    write_baseline_csv(output, scan)

    println("wrote EDMC baseline CSV: $output")
    println("lattice=$(lat.kind), dims=$(lat.dims), nsites=$(lat.nsites), temperatures=$(length(temperatures))")
    println("warmup=$warmup, sampling=$sampling, seed=$seed")
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
