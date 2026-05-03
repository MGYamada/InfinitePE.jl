#!/usr/bin/env julia

using InfinitePE

const USAGE = """
Compare dense pseudofermion Infinite Product Expansion with EDMC.

Usage:
  julia --project=. scripts/pseudofermion_ipe_compare_edmc.jl [options]

Options:
  --Lx INT                             Honeycomb linear size in x [2]
  --Ly INT                             Honeycomb linear size in y [1]
  --warmup INT                         Warmup sweeps per temperature [10]
  --sampling INT                       Sampling sweeps per temperature [50]
  --cutoff INT                         Matsubara cutoff for pseudofermion IPE [8]
  --seed INT                           RNG seed [20260503]
  --temps CSV                          Temperatures, e.g. 0.5,1.0,2.0 [0.5,1.0,2.0]
  --Jx FLOAT                           Kitaev x coupling [1.0]
  --Jy FLOAT                           Kitaev y coupling [1.0]
  --Jz FLOAT                           Kitaev z coupling [1.0]
  --output PATH                        CSV output path [pseudofermion_ipe_vs_edmc.csv]
  --help                               Show this message
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
    text = get_string(opts, "temps", "0.5,1.0,2.0")
    temperatures = [parse(Float64, strip(part)) for part in split(text, ",") if !isempty(strip(part))]
    isempty(temperatures) && throw(ArgumentError("--temps must contain at least one temperature"))
    any(t -> !isfinite(t) || t <= 0, temperatures) &&
        throw(ArgumentError("all temperatures must be positive and finite"))
    return temperatures
end

function csv_escape(value)
    text = string(value)
    if occursin(",", text) || occursin("\"", text) || occursin("\n", text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function write_comparison_csv(path, temperatures, edmc_rows, edmc_runs, pf_rows, pf_runs)
    open(path, "w") do io
        println(io, join((
            "temperature",
            "method",
            "beta",
            "energy_per_site",
            "specific_heat_per_site",
            "acceptance_rate",
            "sampling_acceptance_rate",
            "nsamples",
            "cutoff",
        ), ","))

        for (row, run) in zip(edmc_rows, edmc_runs)
            values = (
                row.temperature,
                row.method,
                row.beta,
                row.energy_per_site,
                row.specific_heat_per_site,
                EDMC.acceptance_rate(run),
                EDMC.sampling_acceptance_rate(run),
                row.nsamples,
                "",
            )
            println(io, join(csv_escape.(values), ","))
        end

        for (row, run) in zip(pf_rows, pf_runs)
            values = (
                row.temperature,
                row.method,
                row.beta,
                row.energy_per_site,
                row.specific_heat_per_site,
                run.acceptance_rate,
                run.sampling_acceptance_rate,
                row.nsamples,
                row.metadata.cutoff,
            )
            println(io, join(csv_escape.(values), ","))
        end
    end
    return path
end

function print_summary(temperatures, edmc_rows, pf_rows)
    println("temperature,edmc_energy,pf_energy,energy_delta,edmc_C,pf_C,C_delta")
    for (temperature, edmc, pf) in zip(temperatures, edmc_rows, pf_rows)
        println(join((
            temperature,
            edmc.energy_per_site,
            pf.energy_per_site,
            pf.energy_per_site - edmc.energy_per_site,
            edmc.specific_heat_per_site,
            pf.specific_heat_per_site,
            pf.specific_heat_per_site - edmc.specific_heat_per_site,
        ), ","))
    end
end

function main(args)
    opts = parse_args(args)
    lat = generate_honeycomb(get_int(opts, "Lx", 2), get_int(opts, "Ly", 1), TypeI())
    input = EDMC.lattice_to_edmc(lat)
    temperatures = temperatures_from_options(opts)
    warmup = get_int(opts, "warmup", 10)
    sampling = get_int(opts, "sampling", 50)
    cutoff = get_int(opts, "cutoff", 8)
    seed = get_int(opts, "seed", 20260503)
    output = get_string(opts, "output", "pseudofermion_ipe_vs_edmc.csv")
    couplings = (
        x=get_float(opts, "Jx", 1.0),
        y=get_float(opts, "Jy", 1.0),
        z=get_float(opts, "Jz", 1.0),
    )

    edmc_scan = EDMC.scan_temperatures(
        input,
        temperatures;
        warmup_sweeps=warmup,
        sampling_sweeps=sampling,
        seed=seed,
        couplings=couplings,
    )
    pf_scan = scan_pseudofermion_temperatures(
        input,
        temperatures;
        cutoff=cutoff,
        warmup_sweeps=warmup,
        sampling_sweeps=sampling,
        seed=seed,
        couplings=couplings,
    )

    edmc_rows = EDMC.comparison_table(edmc_scan)
    pf_rows = pseudofermion_comparison_table(pf_scan; metadata=(observable="EDMC-compatible",))
    write_comparison_csv(output, temperatures, edmc_rows, edmc_scan.runs, pf_rows, pf_scan.runs)
    print_summary(temperatures, edmc_rows, pf_rows)
    println("wrote comparison CSV: $output")
    println("lattice=$(lat.kind), dims=$(lat.dims), nsites=$(lat.nsites), cutoff=$cutoff, warmup=$warmup, sampling=$sampling")
    return (edmc=edmc_scan, pseudofermion=pf_scan)
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
