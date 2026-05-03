#!/usr/bin/env julia

using InfinitePE
using Plots

const USAGE = """
Create a Plots.jl figure in the style of Supplemental Material Fig. 5 of
PRL 113, 197205 / arXiv:1406.5415.

Usage:
  julia --project=. scripts/plot_prl113197205_fig5.jl [options]

Options:
  --L INT                              Honeycomb linear size L [2]
  --tmin FLOAT                         Minimum temperature [0.01]
  --tmax FLOAT                         Maximum temperature [10.0]
  --ntemps INT                         Number of logarithmic temperatures [40]
  --temps CSV                          Explicit temperatures, e.g. 0.01,0.1,1.0
  --Jx FLOAT                           Kitaev x coupling [0.3333333333333333]
  --Jy FLOAT                           Kitaev y coupling [0.3333333333333333]
  --Jz FLOAT                           Kitaev z coupling [0.3333333333333333]
  --sign FLOAT                         Overall Full ED sign in H=sign*sum(Jσσ) [-1.0]
  --include-edmc true|false            Overlay EDMC comparison curve [true]
  --warmup INT                         EDMC warmup sweeps per temperature [20]
  --sampling INT                       EDMC sampling sweeps per temperature [80]
  --seed INT                           EDMC RNG seed [113197205]
  --output PATH                        Figure output path [prl113197205_fig5_fulled.png]
  --csv PATH                           Optional CSV output path for plotted data
  --help                               Show this message

Defaults follow the Fig. 8(b) Type-II honeycomb boundary used by the benchmark:
2 x L^2 sites with L=2, and Jx=Jy=Jz=1/3. The right panel is the log-log inset
content shown as a separate panel for readability.
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

function get_bool(opts, key, default)
    value = lowercase(get(opts, key, string(default)))
    value in ("true", "yes", "1") && return true
    value in ("false", "no", "0") && return false
    throw(ArgumentError("--$key must be true or false; got $value"))
end

function temperatures_from_options(opts)
    if haskey(opts, "temps")
        temps = [parse(Float64, strip(part)) for part in split(opts["temps"], ",") if !isempty(strip(part))]
        isempty(temps) && throw(ArgumentError("--temps must contain at least one temperature"))
        any(t -> !isfinite(t) || t <= 0, temps) &&
            throw(ArgumentError("all temperatures must be positive and finite"))
        return sort(temps)
    end

    tmin = get_float(opts, "tmin", 0.01)
    tmax = get_float(opts, "tmax", 10.0)
    ntemps = get_int(opts, "ntemps", 40)
    isfinite(tmin) && tmin > 0 || throw(ArgumentError("--tmin must be positive and finite"))
    isfinite(tmax) && tmax > 0 || throw(ArgumentError("--tmax must be positive and finite"))
    tmax >= tmin || throw(ArgumentError("--tmax must be greater than or equal to --tmin"))
    ntemps > 0 || throw(ArgumentError("--ntemps must be positive"))
    ntemps == 1 && return [tmin]
    return exp.(range(log(tmin), log(tmax); length=ntemps))
end

function csv_escape(value)
    text = string(value)
    if occursin(",", text) || occursin("\"", text) || occursin("\n", text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function write_plot_csv(path, temperatures, fulled_cv, edmc_cv, edmc_cv_error)
    open(path, "w") do io
        println(io, "temperature,fulled_specific_heat_per_site,edmc_specific_heat_per_site,edmc_specific_heat_jackknife_error")
        for i in eachindex(temperatures)
            edmc_value = edmc_cv === nothing ? "" : edmc_cv[i]
            edmc_error = edmc_cv_error === nothing ? "" : edmc_cv_error[i]
            println(io, join(csv_escape.((temperatures[i], fulled_cv[i], edmc_value, edmc_error)), ","))
        end
    end
    return path
end

function make_plot(temperatures, fulled_cv, edmc_cv, edmc_cv_error; title, output)
    default(
        framestyle=:box,
        grid=false,
        linewidth=2,
        markersize=4,
        legendfontsize=9,
        tickfontsize=9,
        guidefontsize=11,
        titlefontsize=11,
    )

    main = plot(
        temperatures,
        fulled_cv;
        label="Full ED",
        color=:black,
        marker=:circle,
        xlabel="T",
        ylabel="Cv",
        title=title,
        xscale=:log10,
        xlims=(minimum(temperatures), maximum(temperatures)),
    )
    if edmc_cv !== nothing
        plot!(
            main,
            temperatures,
            edmc_cv;
            yerror=edmc_cv_error,
            label="EDMC",
            color=:red,
            marker=:utriangle,
            linestyle=:dash,
        )
    end

    positive_fulled = max.(fulled_cv, eps(Float64))
    inset = plot(
        temperatures,
        positive_fulled;
        label="Full ED",
        color=:black,
        marker=:circle,
        xlabel="T",
        ylabel="Cv",
        xscale=:log10,
        yscale=:log10,
        legend=:bottomright,
    )
    if edmc_cv !== nothing
        plot!(
            inset,
            temperatures,
            max.(edmc_cv, eps(Float64));
            yerror=edmc_cv_error,
            label="EDMC",
            color=:red,
            marker=:utriangle,
            linestyle=:dash,
        )
    end

    fig = plot(main, inset; layout=(1, 2), size=(960, 420), margin=5Plots.mm)
    savefig(fig, output)
    return fig
end

function edmc_specific_heat_jackknife(input, beta, samples; couplings=(x=1.0, y=1.0, z=1.0), atol=1e-10)
    isempty(samples) && throw(ArgumentError("cannot compute Jackknife error without EDMC samples"))
    nsites = input.bondset.nsites
    energies = Float64[]
    derivatives = Float64[]
    sizehint!(energies, length(samples))
    sizehint!(derivatives, length(samples))

    for sample in samples
        sample_input = EDMC.KitaevHamiltonianInput(input.bondset, sample)
        push!(energies, EDMC.internal_energy(sample_input, beta; couplings=couplings, atol=atol))
        push!(derivatives, EDMC.internal_energy_beta_derivative(sample_input, beta; couplings=couplings, atol=atol))
    end

    estimate = _specific_heat_from_sums(sum(energies), sum(abs2, energies), sum(derivatives), length(samples), beta, nsites)
    length(samples) == 1 && return estimate, 0.0

    jackknife_values = Float64[]
    sizehint!(jackknife_values, length(samples))
    energy_sum = sum(energies)
    energy2_sum = sum(abs2, energies)
    derivative_sum = sum(derivatives)
    for i in eachindex(samples)
        push!(
            jackknife_values,
            _specific_heat_from_sums(
                energy_sum - energies[i],
                energy2_sum - abs2(energies[i]),
                derivative_sum - derivatives[i],
                length(samples) - 1,
                beta,
                nsites,
            ),
        )
    end

    jackknife_mean = sum(jackknife_values) / length(jackknife_values)
    variance = (length(jackknife_values) - 1) / length(jackknife_values) *
               sum(abs2(value - jackknife_mean) for value in jackknife_values)
    return estimate, sqrt(max(0.0, variance))
end

function _specific_heat_from_sums(energy_sum, energy2_sum, derivative_sum, nsamples, beta, nsites)
    mean_energy = energy_sum / nsamples
    mean_energy2 = energy2_sum / nsamples
    mean_derivative = derivative_sum / nsamples
    specific_heat = beta^2 * (mean_energy2 - mean_energy^2 - mean_derivative) / nsites
    return max(0.0, specific_heat)
end

function main(args)
    opts = parse_args(args)
    L = get_int(opts, "L", 2)
    L > 0 || throw(ArgumentError("--L must be positive"))
    temperatures = temperatures_from_options(opts)
    couplings = (
        x=get_float(opts, "Jx", 1 / 3),
        y=get_float(opts, "Jy", 1 / 3),
        z=get_float(opts, "Jz", 1 / 3),
    )
    sign = get_float(opts, "sign", -1.0)
    output = get_string(opts, "output", "prl113197205_fig5_fulled.png")
    csv = get(opts, "csv", nothing)
    include_edmc = get_bool(opts, "include-edmc", true)

    lat = generate_honeycomb(L, L, TypeII())
    fulled_input = FullED.lattice_to_fulled(lat)
    fulled_scan = FullED.scan_temperatures(fulled_input, temperatures; couplings=couplings, sign=sign)
    fulled_cv = [obs.specific_heat for obs in fulled_scan.observables]

    edmc_cv = nothing
    edmc_cv_error = nothing
    if include_edmc
        warmup = get_int(opts, "warmup", 20)
        sampling = get_int(opts, "sampling", 80)
        seed = get_int(opts, "seed", 113197205)
        edmc_input = EDMC.lattice_to_edmc(lat)
        edmc_scan = EDMC.scan_temperatures(
            edmc_input,
            temperatures;
            warmup_sweeps=warmup,
            sampling_sweeps=sampling,
            seed=seed,
            couplings=couplings,
        )
        estimates = [
            edmc_specific_heat_jackknife(edmc_input, inv(Float64(temperature)), run.samples; couplings=couplings)
            for (temperature, run) in zip(temperatures, edmc_scan.runs)
        ]
        edmc_cv = first.(estimates)
        edmc_cv_error = last.(estimates)
    end

    title = "Honeycomb Type-II, N=$(lat.nsites), Jx=Jy=Jz=$(couplings.x)"
    make_plot(temperatures, fulled_cv, edmc_cv, edmc_cv_error; title=title, output=output)
    csv !== nothing && write_plot_csv(csv, temperatures, fulled_cv, edmc_cv, edmc_cv_error)

    println("wrote figure: $output")
    csv !== nothing && println("wrote plotted data CSV: $csv")
    println("lattice=$(lat.kind), dims=$(lat.dims), nsites=$(lat.nsites), temperatures=$(length(temperatures))")
    println("include_edmc=$include_edmc")
    return output
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
