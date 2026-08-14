use lie_cliffalg_analog_svd::lie_svd_block4;
use lie_cliffalg_analog_svd::lie_svd_bss;
use lie_cliffalg_analog_svd::lie_svd_compiler::{HardwareSchedule, HardwareTarget};
use lie_cliffalg_analog_svd::lie_svd_complex;
use lie_cliffalg_analog_svd::lie_svd_coreflow::LieSvdCoreFlowParams;
use lie_cliffalg_analog_svd::lie_svd_engine::PhaseEngine;
use lie_cliffalg_analog_svd::lie_svd_joint;
use lie_cliffalg_analog_svd::lie_svd_phaseflow;
use lie_cliffalg_analog_svd::lie_svd_phasehealth;
use lie_cliffalg_analog_svd::lie_svd_quadenergy;
use lie_cliffalg_analog_svd::lie_svd_tensor;
use lie_cliffalg_analog_svd::lie_svd_tensortrain;
use lie_cliffalg_analog_svd::lie_svd_topowarm::TopologicalWarmStartParams;
use lie_cliffalg_analog_svd::lie_svd_traceflow;
use lie_cliffalg_analog_svd::metrics;
use lie_cliffalg_analog_svd::profiles::{generate, Profile};
use lie_cliffalg_analog_svd::solvers;
use lie_cliffalg_analog_svd::SvdTriple;
use ndarray::{Array2, Array3};
use num_complex::Complex64;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

struct CountingAlloc;

static ALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
static CURRENT_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            record_alloc(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        sub_current(layout.size());
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = System.realloc(ptr, layout, new_size);
        if !new_ptr.is_null() {
            let old_size = layout.size();
            ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(new_size, Ordering::Relaxed);
            if new_size >= old_size {
                bump_current(new_size - old_size);
            } else {
                sub_current(old_size - new_size);
            }
        }
        new_ptr
    }
}

#[derive(Clone, Copy)]
struct AllocStats {
    calls: usize,
    mb: f64,
    peak_mb: f64,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let n = args
        .iter()
        .find_map(|s| {
            if s.starts_with("--") {
                None
            } else {
                s.parse::<usize>().ok()
            }
        })
        .unwrap_or(64);
    let full_suite = args.iter().any(|s| s == "--full-suite");
    let include_analog = args.iter().any(|s| s == "--analog") || full_suite;
    let include_coreflow = args.iter().any(|s| s == "--coreflow") || full_suite;
    let include_block4 = args.iter().any(|s| s == "--block4") || full_suite;
    let include_topowarm =
        args.iter().any(|s| s == "--topowarm" || s == "--topo-warm") || full_suite;
    let include_auto_trace = args.iter().any(|s| s == "--auto-trace") || full_suite;
    let include_kron_trace = args.iter().any(|s| s == "--kron-trace") || full_suite;
    let include_kron_chain = args.iter().any(|s| s == "--kron-chain") || full_suite;
    let include_traceflow = args.iter().any(|s| s == "--traceflow");
    let include_trace_nav = args.iter().any(|s| s == "--trace-nav") || full_suite;
    let include_quad_energy = args.iter().any(|s| s == "--quad-energy") || full_suite;
    let include_phase_health = args.iter().any(|s| s == "--phase-health") || full_suite;
    let include_phaseflow = args.iter().any(|s| s == "--phaseflow") || full_suite;
    let include_phaseflow_polish = args.iter().any(|s| s == "--phaseflow-polish") || full_suite;
    let include_golden_jumps = args.iter().any(|s| s == "--golden-jumps");
    let disable_golden_jumps = args.iter().any(|s| s == "--no-golden-jumps");
    let include_golden_prespin = args.iter().any(|s| s == "--golden-prespin");
    let disable_golden_prespin = args.iter().any(|s| s == "--no-golden-prespin");
    let include_causal_antispin = args.iter().any(|s| s == "--causal-antispin");
    let disable_causal_antispin = args.iter().any(|s| s == "--no-causal-antispin");
    let prespin_depth = parse_usize_flag(&args, "--prespin-depth");
    let yinyang_cycles = parse_usize_flag(&args, "--yinyang-cycles");
    let include_phase_conjugate = args.iter().any(|s| s == "--phase-conjugate");
    let include_bottleneck = args.iter().any(|s| s == "--bottleneck");
    let disable_incremental_bottleneck = args
        .iter()
        .any(|s| s == "--rescan-bottleneck" || s == "--no-incremental-bottleneck");
    let phase_viscosity = parse_f64_flag(&args, "--phase-viscosity");
    let phase_quantization_levels = parse_usize_flag(&args, "--phase-quantization-levels");
    let active_set_alpha = parse_f64_flag(&args, "--active-set-alpha");
    let adaptive_viscosity = args.iter().any(|s| s == "--adaptive-viscosity");
    let include_joint = args.iter().any(|s| s == "--joint") || full_suite;
    let include_joint_svd = args.iter().any(|s| s == "--joint-svd") || full_suite;
    let include_rect = args.iter().any(|s| s == "--rect") || full_suite;
    let include_bss_demo = args.iter().any(|s| s == "--bss-demo") || full_suite;
    let include_tensor_hosvd = args.iter().any(|s| s == "--tensor-hosvd") || full_suite;
    let include_complex_svd = args.iter().any(|s| s == "--complex-svd") || full_suite;
    let diagnostics_only = args.iter().any(|s| s == "--diagnostics-only");
    let rect_cols = parse_usize_flag(&args, "--rect-cols").unwrap_or(n + n / 2);
    let topowarm_rank = parse_usize_flag(&args, "--topowarm-rank");
    let topowarm_power_steps = parse_usize_flag(&args, "--topowarm-power-steps");
    let topowarm_graph_steps = parse_usize_flag(&args, "--topowarm-graph-steps");
    let topowarm_seed = parse_u64_flag(&args, "--topowarm-seed");
    let repel_lambda = parse_f64_flag(&args, "--repel-lambda").unwrap_or(0.0);
    let repel_eps = parse_f64_flag(&args, "--repel-eps").unwrap_or(1e-12);
    let profiles = [
        Profile::UniformRandom,
        Profile::DegenerateSpectrum,
        Profile::ExtremeIllConditioned,
        Profile::JordanDefective,
        Profile::SparseStructured,
        Profile::NearlyDiagonal,
        Profile::KronStructured,
    ];

    println!("lie_cliffalg_analog_svd CPU stress");
    println!(
        "n={n} full_suite={full_suite} analog_polished={include_analog} coreflow={include_coreflow} block4={include_block4} topowarm={include_topowarm} golden_jumps={include_golden_jumps} no_golden_jumps={disable_golden_jumps} golden_prespin={include_golden_prespin} no_golden_prespin={disable_golden_prespin} causal_antispin={include_causal_antispin} no_causal_antispin={disable_causal_antispin} prespin_depth={prespin_depth:?} yinyang_cycles={yinyang_cycles:?} phase_conjugate={include_phase_conjugate} bottleneck={include_bottleneck} rescan_bottleneck={disable_incremental_bottleneck} phase_viscosity={phase_viscosity:?} phase_quantization_levels={phase_quantization_levels:?} active_set_alpha={active_set_alpha:?} adaptive_viscosity={adaptive_viscosity} traceflow={include_traceflow} trace_nav={include_trace_nav} phaseflow={include_phaseflow} phaseflow_polish={include_phaseflow_polish} joint={include_joint} joint_svd={include_joint_svd} rect={include_rect} bss_demo={include_bss_demo} tensor_hosvd={include_tensor_hosvd} complex_svd={include_complex_svd} repel_lambda={repel_lambda:.3e} repel_eps={repel_eps:.3e}"
    );
    println!(
        "{:<24} {:<18} {:>8} {:>11} {:>10} {:>10} {:>11} {:>11} {:>9} {:>9} {:>9}",
        "profile",
        "solver",
        "time_s",
        "rel_recon",
        "orth_u",
        "orth_v",
        "sigma_max",
        "tail_rel",
        "allocs",
        "alloc_mb",
        "peak_mb"
    );

    if include_joint {
        print_joint_phase_jade_smoke(n);
    }
    if include_joint_svd {
        print_joint_svd_smoke(n);
    }
    if include_rect {
        print_rectangular_phaseflow_smoke(
            n,
            rect_cols,
            include_golden_prespin,
            disable_golden_prespin,
            include_causal_antispin,
            disable_causal_antispin,
            prespin_depth,
            yinyang_cycles,
            include_phase_conjugate,
            include_bottleneck,
            disable_incremental_bottleneck,
            phase_viscosity,
            phase_quantization_levels,
            active_set_alpha,
            adaptive_viscosity,
        );
    }
    if include_bss_demo {
        print_bss_demo(n);
    }
    if include_tensor_hosvd {
        print_tensor_hosvd_demo(n);
    }
    if include_complex_svd {
        print_complex_svd_demo(n);
    }
    if full_suite {
        print_phase_engine_and_compiler_smoke(n);
    }
    if diagnostics_only {
        return;
    }

    for profile in profiles {
        let case = generate(n, profile, 17);
        if include_auto_trace {
            let trace = lie_cliffalg_analog_svd::lie_svd_adaptive::choose_route(
                &case.a,
                lie_cliffalg_analog_svd::lie_svd_adaptive::LieSvdAdaptiveParams::default(),
            );
            println!(
                "auto_trace profile={} route={:?} score={:.3} offdiag={:.3} diag_dom={:.3} row_cv={:.3} col_cv={:.3} mismatch={:.3} sym={:.3} torsion={:.3} diag_h={:.3} row_h={:.3} col_h={:.3} phase_mean={:.3e} phase_twist={:.3e} phase_causal={:.3e} phase_entropy_gap={:.3e}",
                profile.name(),
                trace.route,
                trace.triage.suspicious_score,
                trace.triage.offdiag_ratio,
                trace.triage.diagonal_dominance,
                trace.triage.row_cv,
                trace.triage.col_cv,
                trace.triage.row_col_mismatch,
                trace.triage.symmetry_ratio,
                trace.triage.transpose_torsion_ratio,
                trace.triage.diagonal_entropy,
                trace.triage.row_mass_entropy,
                trace.triage.col_mass_entropy,
                trace.triage.phase_mean_stress,
                trace.triage.phase_max_twist,
                trace.triage.phase_causal_disbalance,
                trace.triage.phase_entropy_gap,
            );
        }
        if include_kron_trace {
            print_kron_trace(profile, &case.a);
        }
        if include_trace_nav {
            print_trace_nav(profile, &case.a);
        }
        if include_quad_energy {
            print_quad_energy(profile, &case.a);
        }
        if include_phase_health {
            print_phase_health(profile, &case.a);
        }
        if include_phaseflow {
            print_phaseflow_trace(
                profile,
                &case.a,
                include_golden_jumps,
                disable_golden_jumps,
                include_golden_prespin,
                disable_golden_prespin,
                include_causal_antispin,
                disable_causal_antispin,
                prespin_depth,
                yinyang_cycles,
                include_phase_conjugate,
                include_bottleneck,
                disable_incremental_bottleneck,
                phase_viscosity,
                phase_quantization_levels,
                active_set_alpha,
                adaptive_viscosity,
            );
        }
        if include_block4 {
            print_block4_trace(profile, &case.a);
        }
        run_one(profile, "Small", &case.a, case.sigma_ref.as_ref(), || {
            let (u, s, vt, _) = solvers::run_small(&case.a);
            (u, s, vt)
        });
        run_one(profile, "Hybrid", &case.a, case.sigma_ref.as_ref(), || {
            let (u, s, vt, _) = solvers::run_hybrid(&case.a);
            (u, s, vt)
        });
        run_one(profile, "Auto", &case.a, case.sigma_ref.as_ref(), || {
            let (u, s, vt, _) = solvers::run_auto(&case.a);
            (u, s, vt)
        });
        if include_analog {
            run_one(
                profile,
                "AnalogPolished",
                &case.a,
                case.sigma_ref.as_ref(),
                || {
                    let (u, s, vt, _) = solvers::run_analog_polished(&case.a);
                    (u, s, vt)
                },
            );
        }
        if include_coreflow {
            let mut coreflow_params = LieSvdCoreFlowParams::for_n(n);
            coreflow_params.repel_lambda = repel_lambda;
            coreflow_params.repel_eps = repel_eps;
            if include_topowarm {
                let mut warm = TopologicalWarmStartParams::for_n(n);
                if let Some(rank) = topowarm_rank {
                    warm.rank = rank;
                }
                if let Some(power_steps) = topowarm_power_steps {
                    warm.power_steps = power_steps;
                }
                if let Some(graph_steps) = topowarm_graph_steps {
                    warm.graph_relax_steps = graph_steps;
                }
                if let Some(seed) = topowarm_seed {
                    warm.random_probe_seed = seed;
                }
                coreflow_params.warm_start = Some(warm);
            }
            run_one(
                profile,
                "CoreFlow",
                &case.a,
                case.sigma_ref.as_ref(),
                || {
                    let (u, s, vt, _) = solvers::run_coreflow_with_params(&case.a, coreflow_params);
                    (u, s, vt)
                },
            );
        }
        if include_block4 {
            run_one(
                profile,
                "Block4Polished",
                &case.a,
                case.sigma_ref.as_ref(),
                || {
                    let (u, s, vt, _) = solvers::run_block4(&case.a);
                    (u, s, vt)
                },
            );
        }
        if include_traceflow {
            run_one(
                profile,
                "TraceFlow",
                &case.a,
                case.sigma_ref.as_ref(),
                || {
                    let (u, s, vt, _) = solvers::run_traceflow(&case.a);
                    (u, s, vt)
                },
            );
        }
        if include_phaseflow {
            run_one(
                profile,
                "PhaseFlow",
                &case.a,
                case.sigma_ref.as_ref(),
                || {
                    let mut params =
                        lie_svd_phaseflow::LieSvdPhaseFlowParams::for_n(case.a.nrows());
                    if include_golden_jumps {
                        params.use_golden_jumps = true;
                    }
                    if disable_golden_jumps {
                        params.use_golden_jumps = false;
                    }
                    if include_golden_prespin {
                        params.use_golden_prespin = true;
                    }
                    if disable_golden_prespin {
                        params.use_golden_prespin = false;
                    }
                    if include_causal_antispin {
                        params.use_causal_antispin = true;
                    }
                    if disable_causal_antispin {
                        params.use_causal_antispin = false;
                    }
                    if let Some(depth) = prespin_depth {
                        params.prespin_depth = depth;
                        params.adaptive_prespin_depth = false;
                    }
                    if let Some(cycles) = yinyang_cycles {
                        params.use_yinyang_prespin = cycles > 0;
                        params.yinyang_cycles = cycles;
                    }
                    apply_phaseflow_experiment_flags(
                        &mut params,
                        include_phase_conjugate,
                        include_bottleneck,
                        disable_incremental_bottleneck,
                        phase_viscosity,
                        phase_quantization_levels,
                        active_set_alpha,
                        adaptive_viscosity,
                    );
                    let (u, s, vt) =
                        lie_svd_phaseflow::LieSvdPhaseFlow::phase_lock_with_trace(&case.a, params)
                            .0;
                    (u, s, vt)
                },
            );
        }
        if include_phaseflow_polish {
            run_one(
                profile,
                "PhaseFlowPolished",
                &case.a,
                case.sigma_ref.as_ref(),
                || {
                    let mut params =
                        lie_svd_phaseflow::LieSvdPhaseFlowParams::for_n(case.a.nrows());
                    if include_golden_jumps {
                        params.use_golden_jumps = true;
                    }
                    if disable_golden_jumps {
                        params.use_golden_jumps = false;
                    }
                    if include_golden_prespin {
                        params.use_golden_prespin = true;
                    }
                    if disable_golden_prespin {
                        params.use_golden_prespin = false;
                    }
                    if include_causal_antispin {
                        params.use_causal_antispin = true;
                    }
                    if disable_causal_antispin {
                        params.use_causal_antispin = false;
                    }
                    if let Some(depth) = prespin_depth {
                        params.prespin_depth = depth;
                        params.adaptive_prespin_depth = false;
                    }
                    if let Some(cycles) = yinyang_cycles {
                        params.use_yinyang_prespin = cycles > 0;
                        params.yinyang_cycles = cycles;
                    }
                    apply_phaseflow_experiment_flags(
                        &mut params,
                        include_phase_conjugate,
                        include_bottleneck,
                        disable_incremental_bottleneck,
                        phase_viscosity,
                        phase_quantization_levels,
                        active_set_alpha,
                        adaptive_viscosity,
                    );
                    let ((u, s, vt), _) =
                        lie_svd_phaseflow::LieSvdPhaseFlow::solve_with_digital_polish(
                            &case.a, params,
                        );
                    (u, s, vt)
                },
            );
        }
        if include_kron_chain {
            if lie_svd_tensortrain::factor_kron2_chain(
                &case.a,
                lie_svd_tensortrain::TensorTrainSvdParams::default(),
            )
            .is_some()
            {
                run_one(
                    profile,
                    "KronChain",
                    &case.a,
                    case.sigma_ref.as_ref(),
                    || {
                        lie_svd_tensortrain::solve_if_kron_chain(
                            &case.a,
                            lie_svd_tensortrain::TensorTrainSvdParams::default(),
                        )
                        .expect("checked Kron chain")
                    },
                );
            }
        }
    }
}

fn print_joint_phase_jade_smoke(n: usize) {
    let n = n.clamp(4, 256);
    let family = 6usize;
    let mut seed = 0x9e3779b97f4a7c15_u64;
    let q = deterministic_orthogonal(n, &mut seed);
    let matrices = (0..family)
        .map(|k| {
            let diag = ndarray::Array1::from_shape_fn(n, |i| {
                1.0 + k as f64 * 0.2 + (i as f64 + 1.0).sin().abs()
            });
            let d = Array2::from_diag(&diag);
            q.dot(&d).dot(&q.t())
        })
        .collect::<Vec<_>>();
    let params = lie_svd_joint::JointDiagonalizationParams::for_n(n);
    let start = Instant::now();
    let (_v, _diagonals, trace) =
        lie_svd_joint::LieSvdJoint::diagonalize_symmetric_with_params(&matrices, params);
    println!(
        "joint_phase_jade n={} family={} time_s={:.6} offdiag={:.3e}->{:.3e} sweeps={} rotations={} rejected={}",
        n,
        family,
        start.elapsed().as_secs_f64(),
        trace.initial_offdiag,
        trace.final_offdiag,
        trace.sweeps,
        trace.rotations,
        trace.rejected_rotations,
    );
}

fn print_rectangular_phaseflow_smoke(
    rows: usize,
    cols: usize,
    golden_prespin: bool,
    disable_golden_prespin: bool,
    causal_antispin: bool,
    disable_causal_antispin: bool,
    prespin_depth: Option<usize>,
    yinyang_cycles: Option<usize>,
    phase_conjugate: bool,
    bottleneck: bool,
    disable_incremental_bottleneck: bool,
    phase_viscosity: Option<f64>,
    phase_quantization_levels: Option<usize>,
    active_set_alpha: Option<f64>,
    adaptive_viscosity: bool,
) {
    let rows = rows.clamp(4, 512);
    let cols = cols.clamp(4, 1024);
    let k = rows.min(cols);
    let a = Array2::from_shape_fn((rows, cols), |(i, j)| {
        if i == j {
            2.0 + i as f64 * 0.03
        } else {
            1e-3 * ((i * 17 + j * 13 + 7) as f64).sin()
        }
    });
    let sig = lie_svd_phasehealth::phase_signature(&a);
    let mut params = lie_svd_phaseflow::LieSvdPhaseFlowParams::for_n(rows.max(cols));
    params.max_passes = 16 + rows.max(cols) / 8;
    if golden_prespin {
        params.use_golden_prespin = true;
    }
    if disable_golden_prespin {
        params.use_golden_prespin = false;
    }
    if causal_antispin {
        params.use_causal_antispin = true;
    }
    if disable_causal_antispin {
        params.use_causal_antispin = false;
    }
    if let Some(depth) = prespin_depth {
        params.prespin_depth = depth;
        params.adaptive_prespin_depth = false;
    }
    if let Some(cycles) = yinyang_cycles {
        params.use_yinyang_prespin = cycles > 0;
        params.yinyang_cycles = cycles;
    }
    apply_phaseflow_experiment_flags(
        &mut params,
        phase_conjugate,
        bottleneck,
        disable_incremental_bottleneck,
        phase_viscosity,
        phase_quantization_levels,
        active_set_alpha,
        adaptive_viscosity,
    );
    let start = Instant::now();
    let ((_u, sigma, _vt), trace) =
        lie_svd_phaseflow::LieSvdPhaseFlow::phase_lock_rectangular_with_trace(&a, params);
    println!(
        "rect_phaseflow rows={} cols={} rank={} time_s={:.6} offdiag={:.3e}->{:.3e} stress={:.3e}->{:.3e} passes={} prespin={} causal={} yinyang={} conjugate={} cycles={} bottleneck={} cache_updates={} cache_refreshes={} jumps={} unwrap={} rejected={} sig_mean={:.3e} sig_twist={:.3e} sig_entropy_gap={:.3e} sigma0={:.3e}",
        rows,
        cols,
        k,
        start.elapsed().as_secs_f64(),
        trace.initial_offdiag,
        trace.final_offdiag,
        trace.initial_phase_stress,
        trace.final_phase_stress,
        trace.passes,
        trace.golden_prespins,
        trace.causal_antispins,
        trace.yinyang_prespins,
        trace.phase_conjugate_prespins,
        trace.yinyang_cycles,
        trace.bottleneck_rotations,
        trace.bottleneck_cache_updates,
        trace.bottleneck_cache_refreshes,
        trace.phase_jumps,
        trace.unwrap_rotations,
        trace.rejected_rotations,
        sig.mean_stress,
        sig.max_twist,
        sig.entropy_gap,
        sigma.get(0).copied().unwrap_or(0.0),
    );
}

fn print_joint_svd_smoke(n: usize) {
    let n = n.clamp(4, 128);
    let family = 4usize;
    let mut seed = 0xa24baed4963ee407_u64;
    let u0 = deterministic_orthogonal(n, &mut seed);
    let v0 = deterministic_orthogonal(n, &mut seed);
    let matrices = (0..family)
        .map(|k| {
            let diag =
                ndarray::Array1::from_shape_fn(n, |i| 1.0 + k as f64 * 0.15 + i as f64 * 0.05);
            let d = Array2::from_diag(&diag);
            u0.dot(&d).dot(&v0.t())
        })
        .collect::<Vec<_>>();
    let params = lie_svd_joint::JointDiagonalizationParams::for_n(n);
    let start = Instant::now();
    let (_u, _sigmas, _vt, trace) =
        lie_svd_joint::LieSvdJoint::joint_svd_with_params(&matrices, params);
    println!(
        "joint_svd n={} family={} time_s={:.6} offdiag={:.3e}->{:.3e} sweeps={} rotations={} rejected={}",
        n,
        family,
        start.elapsed().as_secs_f64(),
        trace.initial_offdiag,
        trace.final_offdiag,
        trace.sweeps,
        trace.rotations,
        trace.rejected_rotations,
    );
}

fn print_bss_demo(n: usize) {
    let channels = n.clamp(3, 8).min(4);
    let samples = 1024usize;
    let sources = Array2::from_shape_fn((channels, samples), |(i, t)| {
        let x = t as f64 / samples as f64;
        match i {
            0 => (2.0 * std::f64::consts::PI * 7.0 * x).sin(),
            1 => (2.0 * std::f64::consts::PI * 11.0 * x).cos().signum(),
            2 => {
                (2.0 * std::f64::consts::PI * 3.0 * x).sin()
                    + 0.4 * (2.0 * std::f64::consts::PI * 17.0 * x).sin()
            }
            _ => ((t * 37 + 11) as f64).sin() * 0.7,
        }
    });
    let mixing = Array2::from_shape_fn((channels, channels), |(i, j)| {
        if i == j {
            1.0
        } else {
            0.25 * ((i * 13 + j * 7 + 3) as f64).sin()
        }
    });
    let observations = mixing.dot(&sources);
    let before = lie_svd_bss::estimate_sir_db(&sources, &observations);
    let start = Instant::now();
    let result = lie_svd_bss::LieSvdBss::separate(
        &observations,
        lie_svd_bss::PhaseBssParams::for_channels(channels),
    );
    let after = lie_svd_bss::estimate_sir_db(&sources, &result.separated);
    println!(
        "bss_demo channels={} samples={} time_s={:.6} sir_db={:.3}->{:.3} coherence={:.3e} joint_offdiag={:.3e}->{:.3e} rotations={}",
        channels,
        samples,
        start.elapsed().as_secs_f64(),
        before,
        after,
        result.trace.mean_channel_coherence,
        result.trace.joint_initial_offdiag,
        result.trace.joint_final_offdiag,
        result.trace.joint_rotations,
    );
}

fn print_tensor_hosvd_demo(n: usize) {
    let dim = n.clamp(4, 32);
    let tensor = Array3::from_shape_fn((dim, dim, dim), |(i, j, k)| {
        if i == j && j == k {
            5.0 / (1.0 + i as f64)
        } else {
            1e-3 * ((i * 11 + j * 7 + k * 5) as f64).sin()
        }
    });
    let start = Instant::now();
    let fact = lie_svd_tensor::LieSvdTensor::hosvd3(&tensor);
    let recon = lie_svd_tensor::reconstruct_hosvd3(&fact);
    let rel = lie_svd_tensor::tensor_relative_error(&tensor, &recon);
    println!(
        "tensor_hosvd dim={} time_s={:.6} rel_recon={:.3e} offdiag={:.3e}->{:.3e} superdiag_mass={:.3e}",
        dim,
        start.elapsed().as_secs_f64(),
        rel,
        fact.trace.initial_offdiag,
        fact.trace.final_offdiag,
        fact.trace.superdiag_mass_ratio,
    );
}

fn print_complex_svd_demo(n: usize) {
    let n = n.clamp(4, 64);
    for profile in ["complex_iq", "complex_degenerate"] {
        let a = match profile {
            "complex_degenerate" => complex_degenerate_matrix(n),
            _ => complex_iq_matrix(n),
        };
        let mut params = lie_svd_complex::LieSvdComplexParams::for_n(n);
        params.use_golden_prespin = true;
        params.golden_prespin_layers = 1;
        let start = Instant::now();
        let ((u, sigma, vh), trace) = lie_svd_complex::LieSvdComplex::solve_with_trace(&a, params);
        let rel = lie_svd_complex::complex_relative_reconstruction_error(&a, &u, &sigma, &vh);
        let orth_u = lie_svd_complex::complex_unitarity_error(&u);
        let v = vh.t().mapv(|x| x.conj());
        let orth_v = lie_svd_complex::complex_unitarity_error(&v);
        println!(
            "complex_svd profile={} n={} time_s={:.6} rel_recon={:.3e} unitary_u={:.3e} unitary_v={:.3e} offdiag={:.3e}->{:.3e} stress={:.3e}->{:.3e} sweeps={} rotations={} prespin={} polar={} diag_phase={} sigma0={:.3e}",
            profile,
            n,
            start.elapsed().as_secs_f64(),
            rel,
            orth_u,
            orth_v,
            trace.initial_offdiag,
            trace.final_offdiag,
            trace.initial_phase_stress,
            trace.final_phase_stress,
            trace.sweeps,
            trace.rotations,
            trace.golden_prespins,
            trace.polar_polishes,
            trace.diagonal_phase_fixes,
            sigma.get(0).copied().unwrap_or(0.0),
        );
    }
}

fn print_phase_engine_and_compiler_smoke(n: usize) {
    let n = n.clamp(4, 64);
    let a = generate(n, Profile::JordanDefective, 23).a;
    let (_svd, report) = PhaseEngine::solve_real(&a);
    println!(
        "phase_engine real n={} route={:?} time_s={:.6} rel_recon={:.3e} orth_u={:.3e} orth_v={:.3e} stress={:.3e}->{:.3e} passport_mean={:.3e} twist={:.3e} causal={:.3e} chirality={:.3e} golden={:.3e} events={}",
        n,
        report.route,
        report.time_s,
        report.rel_recon.unwrap_or(f64::NAN),
        report.orth_u.unwrap_or(f64::NAN),
        report.orth_v.unwrap_or(f64::NAN),
        report.phase_stress.map(|x| x.0).unwrap_or(f64::NAN),
        report.phase_stress.map(|x| x.1).unwrap_or(f64::NAN),
        report.passport.mean_stress,
        report.passport.max_twist,
        report.passport.causal_disbalance,
        report.passport.chirality,
        report.passport.golden_resonance,
        report.schedule_events,
    );

    let mut params = lie_svd_phaseflow::LieSvdPhaseFlowParams::for_n(n);
    params.record_mzi_phases = true;
    params.max_passes = params.max_passes.min(16);
    let events = lie_svd_phaseflow::LieSvdPhaseFlow::to_mzi_phases(&a, params);
    let schedule = HardwareSchedule::from_real_phaseflow(&events, n, HardwareTarget::MziMesh);
    let json = schedule.to_json_string();
    println!(
        "phase_compiler target={} channels={} events={} layers={} json_bytes={}",
        schedule.target.as_str(),
        schedule.channels,
        schedule.total_events(),
        schedule.conflict_free_layer_count(),
        json.len(),
    );
}

fn complex_iq_matrix(n: usize) -> Array2<Complex64> {
    Array2::from_shape_fn((n, n), |(i, j)| {
        let re = ((i * 17 + j * 11 + 3) as f64).sin();
        let im = 0.5 * ((i * 7 + j * 19 + 5) as f64).cos();
        if i == j {
            Complex64::new(4.0 + 0.2 * i as f64 + 0.01 * re, 0.01 * im)
        } else {
            Complex64::new(0.005 * re, 0.005 * im)
        }
    })
}

fn complex_degenerate_matrix(n: usize) -> Array2<Complex64> {
    let mut seed = 0x9e3779b97f4a7c15_u64;
    let q_re = deterministic_orthogonal(n, &mut seed);
    let p_re = deterministic_orthogonal(n, &mut seed);
    let q = Array2::from_shape_fn((n, n), |(i, j)| {
        q_re[[i, j]] * Complex64::from_polar(1.0, 0.07 * (i + j) as f64)
    });
    let p = Array2::from_shape_fn((n, n), |(i, j)| {
        p_re[[i, j]] * Complex64::from_polar(1.0, -0.11 * (i + 2 * j) as f64)
    });
    let d = Array2::from_shape_fn((n, n), |(i, j)| {
        if i == j {
            let block = i / 4;
            Complex64::new(10.0 / (1.0 + block as f64), 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        }
    });
    q.dot(&d).dot(&p.t().mapv(|x| x.conj()))
}

fn deterministic_orthogonal(n: usize, seed: &mut u64) -> Array2<f64> {
    let mut q = Array2::from_shape_fn((n, n), |_| next_centered(seed));
    for j in 0..n {
        for k in 0..j {
            let mut dot = 0.0_f64;
            for r in 0..n {
                dot += q[[r, j]] * q[[r, k]];
            }
            for r in 0..n {
                q[[r, j]] -= dot * q[[r, k]];
            }
        }
        let mut norm = 0.0_f64;
        for r in 0..n {
            norm += q[[r, j]] * q[[r, j]];
        }
        let norm = norm.sqrt().max(1e-300);
        for r in 0..n {
            q[[r, j]] /= norm;
        }
    }
    q
}

fn next_centered(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let x = ((*seed >> 11) as f64) / ((1_u64 << 53) as f64);
    x - 0.5
}

fn print_phase_health(profile: Profile, a: &Array2<f64>) {
    let h = lie_svd_phasehealth::analyze_fractal_phase_health(a);
    let sig = lie_svd_phasehealth::phase_signature(a);
    println!(
        "phase_health profile={} row_twist={:.3e} col_twist={:.3e} row_max={:.3e} col_max={:.3e} row_entropy={:.3e} col_entropy={:.3e} twist_gap={:.3e} entropy_gap={:.3e} stress={:.3e} sig_mean={:.3e} sig_max_twist={:.3e} sig_causal={:.3e} sig_entropy_gap={:.3e}",
        profile.name(),
        h.rows.mean_twist_ratio,
        h.cols.mean_twist_ratio,
        h.rows.max_twist_ratio,
        h.cols.max_twist_ratio,
        h.rows.mean_entropy,
        h.cols.mean_entropy,
        h.row_col_twist_gap,
        h.row_col_entropy_gap,
        h.total_phase_stress,
        sig.mean_stress,
        sig.max_twist,
        sig.causal_disbalance,
        sig.entropy_gap,
    );
}

fn print_phaseflow_trace(
    profile: Profile,
    a: &Array2<f64>,
    golden_jumps: bool,
    disable_golden_jumps: bool,
    golden_prespin: bool,
    disable_golden_prespin: bool,
    causal_antispin: bool,
    disable_causal_antispin: bool,
    prespin_depth: Option<usize>,
    yinyang_cycles: Option<usize>,
    phase_conjugate: bool,
    bottleneck: bool,
    disable_incremental_bottleneck: bool,
    phase_viscosity: Option<f64>,
    phase_quantization_levels: Option<usize>,
    active_set_alpha: Option<f64>,
    adaptive_viscosity: bool,
) {
    let mut params = lie_svd_phaseflow::LieSvdPhaseFlowParams::for_n(a.nrows());
    if golden_jumps {
        params.use_golden_jumps = true;
    }
    if disable_golden_jumps {
        params.use_golden_jumps = false;
    }
    if golden_prespin {
        params.use_golden_prespin = true;
    }
    if disable_golden_prespin {
        params.use_golden_prespin = false;
    }
    if causal_antispin {
        params.use_causal_antispin = true;
    }
    if disable_causal_antispin {
        params.use_causal_antispin = false;
    }
    if let Some(depth) = prespin_depth {
        params.prespin_depth = depth;
        params.adaptive_prespin_depth = false;
    }
    if let Some(cycles) = yinyang_cycles {
        params.use_yinyang_prespin = cycles > 0;
        params.yinyang_cycles = cycles;
    }
    apply_phaseflow_experiment_flags(
        &mut params,
        phase_conjugate,
        bottleneck,
        disable_incremental_bottleneck,
        phase_viscosity,
        phase_quantization_levels,
        active_set_alpha,
        adaptive_viscosity,
    );
    let ((_u, _sigma, _vt), trace) =
        lie_svd_phaseflow::LieSvdPhaseFlow::phase_lock_with_trace(a, params);
    println!(
        "phaseflow_trace profile={} offdiag={:.3e}->{:.3e} stress={:.3e}->{:.3e} passes={} prespin={} causal={} yinyang={} conjugate={} cycles={} bottleneck={} cache_updates={} cache_refreshes={} jumps={} unwrap={} surgery={} rejected={}",
        profile.name(),
        trace.initial_offdiag,
        trace.final_offdiag,
        trace.initial_phase_stress,
        trace.final_phase_stress,
        trace.passes,
        trace.golden_prespins,
        trace.causal_antispins,
        trace.yinyang_prespins,
        trace.phase_conjugate_prespins,
        trace.yinyang_cycles,
        trace.bottleneck_rotations,
        trace.bottleneck_cache_updates,
        trace.bottleneck_cache_refreshes,
        trace.phase_jumps,
        trace.unwrap_rotations,
        trace.surgery_blocks,
        trace.rejected_rotations,
    );
}

fn print_block4_trace(profile: Profile, a: &Array2<f64>) {
    let params = lie_svd_block4::LieSvdBlock4Params::for_n(a.nrows());
    let sig = lie_svd_block4::analyze_block4_signature(a);
    let start = Instant::now();
    let ((_u, _sigma, _vt), trace) = lie_svd_block4::LieSvdBlock4::warm_start_with_trace(a, params);
    println!(
        "block4_trace profile={} time_s={:.6} offdiag={:.3e}->{:.3e} passes={} accepted={} rejected={} butterfly_layers={} so4_blocks={} self_dual={:.3e} anti_self_dual={:.3e} dual_balance={:.3e}",
        profile.name(),
        start.elapsed().as_secs_f64(),
        trace.initial_offdiag,
        trace.final_offdiag,
        trace.passes,
        trace.accepted_blocks,
        trace.rejected_blocks,
        trace.butterfly_layers,
        sig.blocks,
        sig.self_dual_norm,
        sig.anti_self_dual_norm,
        sig.dual_balance,
    );
}

fn print_quad_energy(profile: Profile, a: &Array2<f64>) {
    let q = lie_svd_quadenergy::analyze_quad_energy(a);
    println!(
        "quad_energy profile={} diag={:.3e} off={:.3e} sym={:.3e} skew={:.3e} upper={:.3e} lower={:.3e} row_metric={:.3e} col_metric={:.3e} dual_gap={:.3e} quad_spread={:.3e} tri_imb={:.3e} balance_err={:.3e}",
        profile.name(),
        q.diag_sq.sqrt(),
        q.offdiag_sq.sqrt(),
        q.sym_offdiag_sq.sqrt(),
        q.skew_sq.sqrt(),
        q.upper_sq.sqrt(),
        q.lower_sq.sqrt(),
        q.row_metric_offdiag_sq.sqrt(),
        q.col_metric_offdiag_sq.sqrt(),
        q.dual_mismatch_sq.sqrt(),
        q.quad_spread,
        q.triangular_imbalance,
        q.direct_balance_error().max(q.offdiag_split_error()),
    );
}

fn print_trace_nav(profile: Profile, a: &Array2<f64>) {
    let params = lie_svd_traceflow::LieSvdTraceFlowParams::for_n(a.nrows());
    let ((_u, _sigma, _vt), trace) =
        lie_svd_traceflow::LieSvdTraceFlow::precondition_with_trace(a, params, params.max_sweeps);
    println!(
        "trace_nav profile={} proj={:.3e}->{:.3e} offdiag={:.3e}->{:.3e} rotations={} rejected={} plateau={}",
        profile.name(),
        trace.initial_projection,
        trace.final_projection,
        trace.initial_offdiag,
        trace.final_offdiag,
        trace.rotations,
        trace.rejected_pairs,
        trace.plateau_pairs,
    );
}

fn print_kron_trace(profile: Profile, a: &Array2<f64>) {
    match lie_svd_tensortrain::kron2_diagnostic(a) {
        Some(first) => {
            let chain = lie_svd_tensortrain::factor_kron2_chain(
                a,
                lie_svd_tensortrain::TensorTrainSvdParams::default(),
            );
            match chain {
                Some(chain) => println!(
                    "kron_trace profile={} first_res={:.3e} ref_block=({},{}) levels={} chain_res={:.3e}",
                    profile.name(),
                    first.relative_residual,
                    first.reference_block.0,
                    first.reference_block.1,
                    chain.levels(),
                    chain.relative_residual,
                ),
                None => println!(
                    "kron_trace profile={} first_res={:.3e} ref_block=({},{}) levels=0 chain_res=n/a",
                    profile.name(),
                    first.relative_residual,
                    first.reference_block.0,
                    first.reference_block.1,
                ),
            }
        }
        None => println!(
            "kron_trace profile={} first_res=n/a ref_block=n/a levels=0 chain_res=n/a",
            profile.name()
        ),
    }
}

fn apply_phaseflow_experiment_flags(
    params: &mut lie_svd_phaseflow::LieSvdPhaseFlowParams,
    phase_conjugate: bool,
    bottleneck: bool,
    disable_incremental_bottleneck: bool,
    phase_viscosity: Option<f64>,
    phase_quantization_levels: Option<usize>,
    active_set_alpha: Option<f64>,
    adaptive_viscosity: bool,
) {
    if phase_conjugate {
        params.use_phase_conjugate_autospin = true;
    }
    if bottleneck {
        params.use_bottleneck_queue = true;
    }
    if disable_incremental_bottleneck {
        params.use_incremental_bottleneck_cache = false;
    }
    if let Some(value) = phase_viscosity {
        params.phase_viscosity = value.clamp(0.05, 1.0);
    }
    if let Some(levels) = phase_quantization_levels {
        params.phase_quantization_levels = levels;
    }
    if adaptive_viscosity {
        params.use_adaptive_viscosity = true;
    }
    if let Some(alpha) = active_set_alpha {
        params.active_set_alpha = alpha.max(0.0);
    }
}

fn parse_f64_flag(args: &[String], name: &str) -> Option<f64> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .and_then(|pair| pair[1].parse::<f64>().ok())
}

fn parse_usize_flag(args: &[String], name: &str) -> Option<usize> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .and_then(|pair| pair[1].parse::<usize>().ok())
}

fn parse_u64_flag(args: &[String], name: &str) -> Option<u64> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .and_then(|pair| pair[1].parse::<u64>().ok())
}

fn run_one<F>(
    profile: Profile,
    name: &str,
    a: &Array2<f64>,
    sigma_ref: Option<&ndarray::Array1<f64>>,
    f: F,
) where
    F: FnOnce() -> SvdTriple,
{
    reset_alloc_stats();
    let start = Instant::now();
    let (u, sigma, vt) = f();
    let elapsed = start.elapsed();
    let alloc = alloc_stats();
    let m = metrics::compute(a, &u, &sigma, &vt, sigma_ref);
    println!(
        "{:<24} {:<18} {:>8.3} {:>11.3e} {:>10.3e} {:>10.3e} {:>11} {:>11} {:>9} {:>9.2} {:>9.2}",
        profile.name(),
        name,
        elapsed.as_secs_f64(),
        m.rel_recon,
        m.orth_u,
        m.orth_v,
        fmt_opt(m.sigma_max_rel),
        fmt_opt(m.sigma_tail_rel),
        alloc.calls,
        alloc.mb,
        alloc.peak_mb,
    );
}

fn fmt_opt(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{x:.3e}"),
        None => "n/a".to_string(),
    }
}

fn record_alloc(bytes: usize) {
    ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
    ALLOC_BYTES.fetch_add(bytes, Ordering::Relaxed);
    bump_current(bytes);
}

fn bump_current(bytes: usize) {
    let current = CURRENT_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
    let mut peak = PEAK_BYTES.load(Ordering::Relaxed);
    while current > peak {
        match PEAK_BYTES.compare_exchange_weak(peak, current, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => break,
            Err(next) => peak = next,
        }
    }
}

fn sub_current(bytes: usize) {
    let mut current = CURRENT_BYTES.load(Ordering::Relaxed);
    loop {
        let next = current.saturating_sub(bytes);
        match CURRENT_BYTES.compare_exchange_weak(
            current,
            next,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
    }
}

fn reset_alloc_stats() {
    ALLOC_CALLS.store(0, Ordering::Relaxed);
    ALLOC_BYTES.store(0, Ordering::Relaxed);
    CURRENT_BYTES.store(0, Ordering::Relaxed);
    PEAK_BYTES.store(0, Ordering::Relaxed);
}

fn alloc_stats() -> AllocStats {
    AllocStats {
        calls: ALLOC_CALLS.load(Ordering::Relaxed),
        mb: ALLOC_BYTES.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0),
        peak_mb: PEAK_BYTES.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0),
    }
}

#[allow(dead_code)]
fn _elapsed_secs(duration: Duration) -> f64 {
    duration.as_secs_f64()
}
