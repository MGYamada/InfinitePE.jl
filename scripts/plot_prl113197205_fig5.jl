#!/usr/bin/env julia

using InfinitePE
using Plots
using Random

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
  --include-fulled true|false          Include Full ED baseline [true]
  --include-edmc true|false            Overlay EDMC comparison curve [true]
  --include-pseudofermion true|false   Overlay pseudofermion IPE curve [true]
  --warmup INT                         EDMC warmup sweeps per temperature [20]
  --sampling INT                       EDMC sampling sweeps per temperature [80]
  --seed INT                           EDMC RNG seed [113197205]
  --pf-warmup INT                      Pseudofermion warmup sweeps per temperature [warmup]
  --pf-sampling INT                    Pseudofermion sampling sweeps per temperature [sampling]
  --pf-seed INT                        Pseudofermion RNG seed [seed]
  --pf-backend dense|sparse            Pseudofermion backend [sparse]
  --pf-observable edmc_compatible|pure|auto
                                        Pseudofermion observable estimator [pure]
  --pf-solver cg|direct                Sparse pseudofermion solver [cg]
  --pf-operator matrix_free|matrix      Sparse CG normal operator [matrix_free]
  --pf-tol FLOAT                       Sparse solver tolerance [1e-10]
  --pf-maxiter INT                     Sparse solver max iterations [1000]
  --pf-krylovdim INT                   Sparse solver Krylov dimension [30]
  --pf-measurement-seed INT            Pure observable RNG seed [pf-seed]
  --pf-measurement-replicas INT        Pure field refreshes per gauge sample [4]
  --large-lattice-threshold INT        Auto observable switches above this N [256]
  --cutoff INT                         Matsubara cutoff for pseudofermion IPE [8]
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

function get_symbol(opts, key, default, allowed)
    value = Symbol(get(opts, key, string(default)))
    value in allowed || throw(ArgumentError("--$key must be one of $(join(string.(allowed), ", ")); got :$value"))
    return value
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

function write_plot_csv(path, temperatures, fulled_cv, edmc_cv, edmc_cv_error, pf_cv, pf_cv_error)
    open(path, "w") do io
        println(
            io,
            "temperature,fulled_specific_heat_per_site,edmc_specific_heat_per_site,edmc_specific_heat_jackknife_error,pseudofermion_specific_heat_per_site,pseudofermion_specific_heat_jackknife_error",
        )
        for i in eachindex(temperatures)
            fulled_value = fulled_cv === nothing ? "" : fulled_cv[i]
            edmc_value = edmc_cv === nothing ? "" : edmc_cv[i]
            edmc_error = edmc_cv_error === nothing ? "" : edmc_cv_error[i]
            pf_value = pf_cv === nothing ? "" : pf_cv[i]
            pf_error = pf_cv_error === nothing ? "" : pf_cv_error[i]
            println(io, join(csv_escape.((temperatures[i], fulled_value, edmc_value, edmc_error, pf_value, pf_error)), ","))
        end
    end
    return path
end

function make_plot(temperatures, fulled_cv, edmc_cv, edmc_cv_error, pf_cv, pf_cv_error; title, output)
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

    main = plot(;
        xlabel="T",
        ylabel="Cv",
        title=title,
        xscale=:log10,
        xlims=(minimum(temperatures), maximum(temperatures)),
    )
    if fulled_cv !== nothing
        plot!(
            main,
            temperatures,
            fulled_cv;
            label="Full ED",
            color=:black,
            marker=:circle,
        )
    end
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
    if pf_cv !== nothing
        plot!(
            main,
            temperatures,
            pf_cv;
            yerror=pf_cv_error,
            label="Pseudofermion IPE",
            color=:blue,
            marker=:diamond,
            linestyle=:dot,
        )
    end

    inset = plot(;
        xlabel="T",
        ylabel="Cv",
        xscale=:log10,
        yscale=:log10,
        legend=:bottomright,
    )
    if fulled_cv !== nothing
        plot!(
            inset,
            temperatures,
            max.(fulled_cv, eps(Float64));
            label="Full ED",
            color=:black,
            marker=:circle,
        )
    end
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
    if pf_cv !== nothing
        plot!(
            inset,
            temperatures,
            max.(pf_cv, eps(Float64));
            yerror=pf_cv_error,
            label="Pseudofermion IPE",
            color=:blue,
            marker=:diamond,
            linestyle=:dot,
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

    estimate = _specific_heat_from_sums(sum(energies), sum(abs2, energies), sum(derivatives), length(energies), beta, nsites)
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

function pseudofermion_specific_heat_jackknife(input, beta, samples; couplings=(x=1.0, y=1.0, z=1.0), atol=1e-10)
    gauge_samples = [EDMC.Z2GaugeField(sample) for sample in samples]
    return edmc_specific_heat_jackknife(input, beta, gauge_samples; couplings=couplings, atol=atol)
end

function pure_pseudofermion_specific_heat_jackknife(
    input,
    beta,
    samples;
    cutoff::Integer,
    seed,
    measurement_replicas::Integer=4,
    couplings=(x=1.0, y=1.0, z=1.0),
    solver::Symbol=:cg,
    operator::Symbol=:matrix_free,
    tol::Real=1e-10,
    maxiter::Integer=1000,
    krylovdim::Integer=30,
)
    isempty(samples) && throw(ArgumentError("cannot compute Jackknife error without pseudofermion samples"))
    measurement_replicas > 0 || throw(ArgumentError("measurement_replicas must be positive; got $measurement_replicas"))
    rng = MersenneTwister(seed)
    gauge_samples = [sample isa EDMC.Z2GaugeField ? sample : EDMC.Z2GaugeField(sample) for sample in samples]
    nsites = input.bondset.nsites
    energies = Float64[]
    derivatives = Float64[]
    block_energy_sums = Float64[]
    block_energy2_sums = Float64[]
    block_derivative_sums = Float64[]
    sizehint!(energies, measurement_replicas * length(gauge_samples))
    sizehint!(derivatives, measurement_replicas * length(gauge_samples))
    sizehint!(block_energy_sums, length(gauge_samples))
    sizehint!(block_energy2_sums, length(gauge_samples))
    sizehint!(block_derivative_sums, length(gauge_samples))

    for sample in gauge_samples
        sample_input = EDMC.KitaevHamiltonianInput(input.bondset, sample)
        matrix = build_sparse_majorana_matrix(sample_input; couplings=couplings)
        block_energy_sum = 0.0
        block_energy2_sum = 0.0
        block_derivative_sum = 0.0
        for _ in 1:measurement_replicas
            fields = refresh_sparse_real_pseudofermions(matrix, beta; cutoff=cutoff, rng=rng)
            estimator = pure_pseudofermion_estimators(
                matrix,
                beta,
                fields;
                solver=solver,
                operator=operator,
                tol=tol,
                maxiter=maxiter,
                krylovdim=krylovdim,
            )
            push!(energies, estimator.energy)
            push!(derivatives, estimator.energy_beta_derivative)
            block_energy_sum += estimator.energy
            block_energy2_sum += abs2(estimator.energy)
            block_derivative_sum += estimator.energy_beta_derivative
        end
        push!(block_energy_sums, block_energy_sum)
        push!(block_energy2_sums, block_energy2_sum)
        push!(block_derivative_sums, block_derivative_sum)
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
                energy_sum - block_energy_sums[i],
                energy2_sum - block_energy2_sums[i],
                derivative_sum - block_derivative_sums[i],
                length(energies) - measurement_replicas,
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

function pseudofermion_specific_heat_estimate(
    input,
    beta,
    samples;
    observable::Symbol,
    cutoff::Integer,
    seed,
    measurement_replicas::Integer,
    couplings,
    solver::Symbol,
    operator::Symbol,
    tol::Real,
    maxiter::Integer,
    krylovdim::Integer,
    large_lattice_threshold::Integer,
    atol::Real=1e-10,
)
    mode = observable === :auto && input.bondset.nsites > large_lattice_threshold ? :pure :
           observable === :auto ? :edmc_compatible : observable
    if mode === :edmc_compatible
        return pseudofermion_specific_heat_jackknife(input, beta, samples; couplings=couplings, atol=atol)
    elseif mode === :pure
        return pure_pseudofermion_specific_heat_jackknife(
            input,
            beta,
            samples;
            cutoff=cutoff,
            seed=seed,
            measurement_replicas=measurement_replicas,
            couplings=couplings,
            solver=solver,
            operator=operator,
            tol=tol,
            maxiter=maxiter,
            krylovdim=krylovdim,
        )
    end
    throw(ArgumentError("unsupported pseudofermion observable :$observable"))
end

function pseudofermion_scan(
    input,
    temperatures;
    backend::Symbol,
    cutoff::Integer,
    warmup_sweeps::Integer,
    sampling_sweeps::Integer,
    seed,
    couplings,
    solver::Symbol,
    operator::Symbol,
    tol::Real,
    maxiter::Integer,
    krylovdim::Integer,
    observable::Symbol,
    measurement_seed,
    large_lattice_threshold::Integer,
)
    if backend === :dense
        return scan_pseudofermion_temperatures(
            input,
            temperatures;
            cutoff=cutoff,
            warmup_sweeps=warmup_sweeps,
            sampling_sweeps=sampling_sweeps,
            seed=seed,
            couplings=couplings,
        )
    elseif backend === :sparse
        return scan_sparse_pseudofermion_temperatures(
            input,
            temperatures;
            cutoff=cutoff,
            warmup_sweeps=warmup_sweeps,
            sampling_sweeps=sampling_sweeps,
            seed=seed,
            couplings=couplings,
            solver=solver,
            operator=operator,
            tol=tol,
            maxiter=maxiter,
            krylovdim=krylovdim,
            observable=observable,
            measurement_seed=measurement_seed,
            large_lattice_threshold=large_lattice_threshold,
        )
    end
    throw(ArgumentError("unsupported pseudofermion backend :$backend"))
end

function _specific_heat_from_sums(energy_sum, energy2_sum, derivative_sum, nsamples, beta, nsites)
    mean_energy = energy_sum / nsamples
    mean_energy2 = energy2_sum / nsamples
    mean_derivative = derivative_sum / nsamples
    specific_heat = beta^2 * (mean_energy2 - mean_energy^2 - mean_derivative) / nsites
    return max(0.0, specific_heat)
end

function coupling_label(couplings)
    if couplings.x == couplings.y == couplings.z
        if couplings.x ≈ 1 / 3
            return "J=1/3"
        end
        return "J=$(round(couplings.x; sigdigits=4))"
    end
    return "J=($(round(couplings.x; sigdigits=3)),$(round(couplings.y; sigdigits=3)),$(round(couplings.z; sigdigits=3)))"
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
    include_fulled = get_bool(opts, "include-fulled", true)
    include_edmc = get_bool(opts, "include-edmc", true)
    include_pseudofermion = get_bool(opts, "include-pseudofermion", true)

    lat = generate_honeycomb(L, L, TypeII())
    fulled_cv = nothing
    if include_fulled
        fulled_input = FullED.lattice_to_fulled(lat)
        fulled_scan = FullED.scan_temperatures(fulled_input, temperatures; couplings=couplings, sign=sign)
        fulled_cv = [obs.specific_heat for obs in fulled_scan.observables]
    end

    edmc_cv = nothing
    edmc_cv_error = nothing
    warmup = get_int(opts, "warmup", 20)
    sampling = get_int(opts, "sampling", 80)
    seed = get_int(opts, "seed", 113197205)
    edmc_input = EDMC.lattice_to_edmc(lat)
    if include_edmc
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

    pf_cv = nothing
    pf_cv_error = nothing
    cutoff = get_int(opts, "cutoff", 8)
    if include_pseudofermion
        pf_warmup = get_int(opts, "pf-warmup", warmup)
        pf_sampling = get_int(opts, "pf-sampling", sampling)
        pf_seed = get_int(opts, "pf-seed", seed)
        pf_backend = get_symbol(opts, "pf-backend", :sparse, (:dense, :sparse))
        pf_observable = get_symbol(opts, "pf-observable", :pure, (:edmc_compatible, :pure, :auto))
        pf_solver = get_symbol(opts, "pf-solver", :cg, (:cg, :direct))
        pf_operator = get_symbol(opts, "pf-operator", :matrix_free, (:matrix_free, :matrix))
        pf_tol = get_float(opts, "pf-tol", 1e-10)
        pf_maxiter = get_int(opts, "pf-maxiter", 1000)
        pf_krylovdim = get_int(opts, "pf-krylovdim", 30)
        pf_measurement_seed = get_int(opts, "pf-measurement-seed", pf_seed)
        pf_measurement_replicas = get_int(opts, "pf-measurement-replicas", 4)
        pf_measurement_replicas > 0 || throw(ArgumentError("--pf-measurement-replicas must be positive"))
        large_lattice_threshold = get_int(opts, "large-lattice-threshold", 256)
        large_lattice_threshold > 0 || throw(ArgumentError("--large-lattice-threshold must be positive"))
        pf_scan = pseudofermion_scan(
            edmc_input,
            temperatures;
            backend=pf_backend,
            cutoff=cutoff,
            warmup_sweeps=pf_warmup,
            sampling_sweeps=pf_sampling,
            seed=pf_seed,
            couplings=couplings,
            solver=pf_solver,
            operator=pf_operator,
            tol=pf_tol,
            maxiter=pf_maxiter,
            krylovdim=pf_krylovdim,
            observable=pf_observable,
            measurement_seed=pf_measurement_seed,
            large_lattice_threshold=large_lattice_threshold,
        )
        estimates = [
            pseudofermion_specific_heat_estimate(
                edmc_input,
                inv(Float64(temperature)),
                run.samples;
                observable=pf_observable,
                cutoff=cutoff,
                seed=pf_measurement_seed + i - 1,
                measurement_replicas=pf_measurement_replicas,
                couplings=couplings,
                solver=pf_solver,
                operator=pf_operator,
                tol=pf_tol,
                maxiter=pf_maxiter,
                krylovdim=pf_krylovdim,
                large_lattice_threshold=large_lattice_threshold,
            )
            for (i, (temperature, run)) in enumerate(zip(temperatures, pf_scan.runs))
        ]
        pf_cv = first.(estimates)
        pf_cv_error = last.(estimates)
    end

    title = "Type-II honeycomb, N=$(lat.nsites), $(coupling_label(couplings)), cutoff=$cutoff"
    make_plot(temperatures, fulled_cv, edmc_cv, edmc_cv_error, pf_cv, pf_cv_error; title=title, output=output)
    csv !== nothing && write_plot_csv(csv, temperatures, fulled_cv, edmc_cv, edmc_cv_error, pf_cv, pf_cv_error)

    println("wrote figure: $output")
    csv !== nothing && println("wrote plotted data CSV: $csv")
    println("lattice=$(lat.kind), dims=$(lat.dims), nsites=$(lat.nsites), temperatures=$(length(temperatures))")
    println("include_fulled=$include_fulled")
    println("include_edmc=$include_edmc")
    if include_pseudofermion
        println(
            "include_pseudofermion=$include_pseudofermion, backend=$(get(opts, "pf-backend", "sparse")), " *
            "observable=$(get(opts, "pf-observable", "pure")), cutoff=$cutoff, " *
            "measurement_replicas=$(get(opts, "pf-measurement-replicas", "4"))",
        )
    else
        println("include_pseudofermion=false")
    end
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
