use std::time::{Duration, Instant};

use crate::{
    arithmetic::{parallel_dot_series_matrix, sample_random_short_mat},
    helpers::println_with_timestamp,
    subroutines::{
        crs::CRS,
        decomp::{decomp_ell, verify_decomp},
        fast_norm_check::{
            norm_check_1, norm_check_2, norm_check_3, verify_norm_check_1, verify_norm_check_3,
            SumCheckProverState,
        },
        fold::{challenge_for_fold, fold, verifier_fold},
        project::{
            batch_projections, batching_challenge_for_project, challenge_for_project, project,
            send_cross_terms_before_join, verifier_join, verify_batching,
        },
        split::{split, verifier_split, VerifierState},
    },
};
pub const SNARK: bool = !FOLDING_SCHEME;

pub static INIT_WIT_REP: usize = if FOLDING_SCHEME { 4 } else { 1 };
pub static WIT_REP: usize = 1;
pub static DECOMP_PARTS: [usize; 12] = [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2];

cfg_if::cfg_if! {
    if #[cfg(feature = "A0")] {
        pub const N: usize = 128;
        pub const MOD_Q: u64 = 1125899904679937;
        pub static MODULE_SIZE: usize = 12;
        pub static WIT_LEN: usize = 1 << 19;
        pub static NOF_ROUNDS: i32 = 8;
        pub const FOLDING_SCHEME: bool = false;
    } else if #[cfg(feature = "A1")] {
        pub const N: usize = 256;
        pub const MOD_Q: u64 = 1125899906840833;
        pub static MODULE_SIZE: usize = 7;
        pub static WIT_LEN: usize = 1 << 18;
        pub static NOF_ROUNDS: i32 = 7;
        pub const FOLDING_SCHEME: bool = false;
    } else if #[cfg(feature = "A2")] {
        pub const N: usize = 512;
        pub const MOD_Q: u64 = 1125899906822657;
        pub static MODULE_SIZE: usize = 4;
        pub static WIT_LEN: usize = 1 << 17;
        pub static NOF_ROUNDS: i32 = 6;
        pub const FOLDING_SCHEME: bool = false;
    } else if #[cfg(feature = "B0")] {
        pub const N: usize = 128;
        pub const MOD_Q: u64 = 1125899904679937;
        pub static MODULE_SIZE: usize = 12;
        pub static WIT_LEN: usize = 1 << 21;
        pub static NOF_ROUNDS: i32 = 10;
        pub const FOLDING_SCHEME: bool = false;
    } else if #[cfg(feature = "B1")] {
        pub const N: usize = 256;
        pub const MOD_Q: u64 = 1125899906840833;
        pub static MODULE_SIZE: usize = 7;
        pub static WIT_LEN: usize = 1 << 20;
        pub static NOF_ROUNDS: i32 = 9;
        pub const FOLDING_SCHEME: bool = false;
    } else if #[cfg(feature = "B2")] {
        pub const N: usize = 512;
        pub const MOD_Q: u64 = 1125899906822657;
        pub static MODULE_SIZE: usize = 4;
        pub static WIT_LEN: usize = 1 << 19;
        pub static NOF_ROUNDS: i32 = 8;
        pub const FOLDING_SCHEME: bool = false;
    } else if #[cfg(feature = "C0")] {
        pub const N: usize = 128;
        pub const MOD_Q: u64 = 1125899904679937;
        pub static MODULE_SIZE: usize = 12;
        pub static WIT_LEN: usize = 1 << 23;
        pub static NOF_ROUNDS: i32 = 12;
    } else if #[cfg(feature = "C1")] {
        pub const N: usize = 256;
        pub const MOD_Q: u64 = 1125899906840833;
        pub static MODULE_SIZE: usize = 7;
        pub static WIT_LEN: usize = 1 << 22;
        pub static NOF_ROUNDS: i32 = 11;
    } else if #[cfg(feature = "C2")] {
        pub const N: usize = 512;
        pub const MOD_Q: u64 = 1125899906822657;
        pub static MODULE_SIZE: usize = 4;
        pub static WIT_LEN: usize = 1 << 21;
        pub static NOF_ROUNDS: i32 = 10;
        pub const FOLDING_SCHEME: bool = false;
    } else if #[cfg(feature = "F0")] {
        pub const N: usize = 128;
        pub const MOD_Q: u64 = 1125899904679937;
        pub static FOLDING_SCHEME: bool = true;
        pub static INIT_WIT_REP: usize = 4;
        pub static MODULE_SIZE: usize = 12;
        pub static WIT_LEN: usize = 1 << 17;

    } else if #[cfg(feature = "F1")] {
        pub const N: usize = 128;
        pub const MOD_Q: u64 = 1125899904679937;
        pub static FOLDING_SCHEME: bool = true;
        pub static MODULE_SIZE: usize = 13;
        pub static WIT_LEN: usize = 1 << 19;

    } else if #[cfg(feature = "F2")] {
        pub const N: usize = 128;
        pub static FOLDING_SCHEME: bool = true;
        pub const MOD_Q: u64 = 1125899904679937;
        pub static MODULE_SIZE: usize = 13;
        pub static WIT_LEN: usize = 1 << 21;
    } else {
        pub const N: usize = 128;
        pub const MOD_Q: u64 = 1125899904679937;
        pub static MODULE_SIZE: usize = 12;
        pub static WIT_LEN: usize = 1 << 19;
        pub static NOF_ROUNDS: i32 = 8;
        pub const FOLDING_SCHEME: bool = false;
    }
}

pub fn protocol<const MOD_Q: u64, const N: usize>() {
    println_with_timestamp!(
        "PARAMS: NEW: {:?}, MODULE: {:?}, WIT_LEN: {:?}, WIT_REP: {:?}, Q: {:?}, N: {:?}, DECOMP_PARTS: {:?}",
        true, MODULE_SIZE, WIT_LEN, WIT_REP, MOD_Q, N, DECOMP_PARTS);

    println_with_timestamp!("Start CRS");

    let crs = CRS::<MOD_Q, N>::gen_crs(WIT_LEN, MODULE_SIZE);
    println_with_timestamp!("end CRS");
    let mut witness = sample_random_short_mat(INIT_WIT_REP, WIT_LEN, 2);
    println_with_timestamp!("end sampling witness");

    let now = Instant::now();
    let commitment = parallel_dot_series_matrix::<MOD_Q, N>(&crs.ck, &witness);
    let elapsed = now.elapsed();

    let mut verifier_state = VerifierState::<MOD_Q, N> {
        wit_len: WIT_LEN,
        wit_rep: INIT_WIT_REP,
        rhs: commitment,
    };

    let mut verifier_runtime = Instant::now().elapsed();
    let mut prover_runtime = Instant::now().elapsed();

    println_with_timestamp!(
        "Time for parallel_dot_series_matrix (commitment): {:.2?}",
        elapsed
    );
    prover_runtime = prover_runtime + elapsed;
    let mut t_norm_check_1 = Duration::from_millis(0);
    let mut t_norm_check_2 = Duration::from_millis(0);
    let mut t_norm_check_3 = Duration::from_millis(0);
    let mut t_split = Duration::from_millis(0);
    let mut t_projection = Duration::from_millis(0);
    let mut t_fold = Duration::from_millis(0);
    let mut t_decomp = Duration::from_millis(0);

    let mut statement = crs.ck;
    for i in 0..NOF_ROUNDS {
        let parts = DECOMP_PARTS[i as usize];

        if SNARK && parts > 1 {
            let now = Instant::now();
            let (new_witness, bdecomp_output) = decomp_ell::<MOD_Q, N>(&statement, witness, parts);
            witness = new_witness;
            let elapsed = now.elapsed();
            println_with_timestamp!("Time for decomp: {:.2?}", elapsed);
            prover_runtime = prover_runtime + elapsed;

            t_decomp += elapsed;
            println_with_timestamp!(
                "Stmnt len {:?}, Witness  REPxLEN {:?}x{:?}  Verifier REPxLEN {:?}x{:?}",
                statement.len(),
                witness.len(),
                witness[0].len(),
                verifier_state.rhs.len(),
                verifier_state.rhs[0].len()
            );

            println_with_timestamp!(
                "bdedecomp: {:?} {:?}",
                bdecomp_output.rhs.len(),
                bdecomp_output.rhs[0].len()
            );
            let now = Instant::now();
            let new_verifier_state = verify_decomp::<MOD_Q, N>(bdecomp_output, &verifier_state);
            verifier_state = new_verifier_state;
            let elapsed = now.elapsed();

            println_with_timestamp!(
                "Verifier REPxLEN: {:?}x{:?}",
                verifier_state.rhs.len(),
                verifier_state.rhs[0].len()
            );

            println_with_timestamp!("Time for verify_decomp: {:.2?}", elapsed);
            verifier_runtime = verifier_runtime + elapsed;
        }

        let now_sc = Instant::now();
        let now = Instant::now();
        let (claims, flattened) = norm_check_1::<MOD_Q, N>(&witness);
        let sum_check_prover_state = SumCheckProverState {
            flattened_witness: flattened,
        };
        let elapsed = now.elapsed();
        println_with_timestamp!("Time for prover::norm_1: {:.2?}", elapsed);
        prover_runtime = prover_runtime + elapsed;
        t_norm_check_1 += elapsed;

        let now = Instant::now();
        let mut sumcheck_verifier_state = verify_norm_check_1::<MOD_Q, N>(claims);
        let elapsed = now.elapsed();
        println_with_timestamp!("Time for verifier::verify_norm_1: {:.2?}", elapsed);
        verifier_runtime = verifier_runtime + elapsed;

        // Amplification
        let mut sum_check_prover_state = sum_check_prover_state.clone();
        for _ in 0..2 {
            sumcheck_verifier_state = sumcheck_verifier_state.clone();
            let now = Instant::now();
            let inner_verifier_runtime =
                norm_check_2::<MOD_Q, N>(&mut sum_check_prover_state, &mut sumcheck_verifier_state);
            let elapsed = now.elapsed();
            println_with_timestamp!("Time for prover::norm_2: {:.2?}", elapsed);
            println_with_timestamp!(
                "Time for verifier::verify_norm_2: {:.2?}",
                inner_verifier_runtime
            );
            prover_runtime = prover_runtime + elapsed;
            t_norm_check_2 += elapsed;
            verifier_runtime = verifier_runtime + inner_verifier_runtime;

            let now = Instant::now();

            let rhs = norm_check_3::<MOD_Q, N>(
                &sumcheck_verifier_state.clone().cs,
                &mut statement,
                &witness,
            );
            let elapsed = now.elapsed();
            println_with_timestamp!("Time for prover::norm_3: {:.2?}", elapsed);
            prover_runtime = prover_runtime + elapsed;
            t_norm_check_3 += elapsed;

            let now = Instant::now();
            verify_norm_check_3::<MOD_Q, N>(rhs, &mut sumcheck_verifier_state, &mut verifier_state);
            let elapsed = now.elapsed();
            println_with_timestamp!("Time for verifier::verify_norm_3: {:.2?}", elapsed);
            verifier_runtime = verifier_runtime + elapsed;
        }

        let elapsed_sc = now_sc.elapsed();
        println_with_timestamp!("Time for sum check: {:.2?}", elapsed_sc);

        if SNARK {
            let now = Instant::now();

            let (new_witness, split_output) = split::<MOD_Q, N>(&mut statement, witness);
            witness = new_witness;
            let elapsed = now.elapsed();
            t_split += elapsed;

            println_with_timestamp!("Time for split: {:.2?}", elapsed);
            prover_runtime = prover_runtime + elapsed;

            let now = Instant::now();

            println_with_timestamp!(
                "Verifier REPxLEN: {:?}x{:?}",
                verifier_state.rhs.len(),
                verifier_state.rhs[0].len()
            );
            verifier_state = verifier_split(&statement, split_output, &verifier_state);
            println_with_timestamp!(
                "Verifier REPxLEN: {:?}x{:?}",
                verifier_state.rhs.len(),
                verifier_state.rhs[0].len()
            );
            let elapsed = now.elapsed();
            println_with_timestamp!("Time for split verifier: {:.2?}", elapsed);
        }

        let now = Instant::now();
        let challenge = challenge_for_project(&verifier_state);
        let elapsed = now.elapsed();

        t_projection += elapsed;
        println_with_timestamp!(
            "Time for challenge project: {:.2?} dims {:?} {:?}",
            elapsed,
            challenge.is_zero[0].len(),
            challenge.is_zero.len()
        );
        println_with_timestamp!("Dims of witness {:?} {:?}", witness.len(), witness[0].len());
        verifier_runtime = verifier_runtime + elapsed;

        let now = Instant::now();
        // we project the witness and compute the commitment
        let (mut projected_witness, projected_commitment) =
            project::<MOD_Q, N>(&statement, &witness, &challenge);
        println_with_timestamp!(
            "projected_witness {:?}x{:?}",
            projected_witness.len(),
            projected_witness[0].len()
        );

        let elapsed = now.elapsed();
        t_projection += elapsed;
        println_with_timestamp!("Time for project witness: {:.2?}", elapsed);
        prover_runtime = prover_runtime + elapsed;

        let now = Instant::now();
        let batching_projections_challenge = batching_challenge_for_project::<MOD_Q, N>();
        let elapsed = now.elapsed();
        t_projection += elapsed;
        println_with_timestamp!("Time for batching projections challenge: {:.2?}", elapsed);
        verifier_runtime = verifier_runtime + elapsed;

        let now = Instant::now();
        let (projected_lhs, projection_rhs, witness_rhs, inner_verifier_runtime) =
            batch_projections::<MOD_Q, N>(
                &projected_witness,
                &witness,
                &challenge,
                batching_projections_challenge,
                &mut statement,
            );
        verifier_runtime = verifier_runtime + inner_verifier_runtime;

        let elapsed = now.elapsed();
        println_with_timestamp!("Time for batch projections: {:.2?}", elapsed);
        prover_runtime = prover_runtime + elapsed;
        t_projection += elapsed;

        let now = Instant::now();
        verifier_state = verify_batching(witness_rhs, &projection_rhs, verifier_state);
        let elapsed = now.elapsed();
        println_with_timestamp!("Time for verify_batching: {:.2?}", elapsed);
        verifier_runtime = verifier_runtime + elapsed;

        let now = Instant::now();
        let challenge = challenge_for_fold::<MOD_Q, N>(&verifier_state, WIT_REP);
        let elapsed = now.elapsed();
        println_with_timestamp!(
            "Time for challenge fold: {:.2?} dims {:?} {:?}",
            elapsed,
            challenge[0].len(),
            challenge.len()
        );
        println_with_timestamp!(
            "Witness REP: {:?} LEN: {:?}",
            witness.len(),
            witness[0].len()
        );
        verifier_runtime = verifier_runtime + elapsed;

        let now = Instant::now();
        let new_witness = fold(witness, &challenge);
        witness = new_witness;
        println_with_timestamp!(
            "Folded witness REP: {:?} LEN: {:?}",
            witness.len(),
            witness[0].len()
        );
        let elapsed = now.elapsed();
        println_with_timestamp!("Time for fold: {:.2?}", elapsed);
        prover_runtime = prover_runtime + elapsed;
        t_fold += elapsed;

        // assert_eq!(projected_witness.len(), witness.len());

        let now = Instant::now();

        verifier_state = verifier_fold::<MOD_Q, N>(&verifier_state, &challenge, WIT_REP);
        let elapsed = now.elapsed();
        println_with_timestamp!("Time for fold verifier: {:.2?}", elapsed);
        verifier_runtime = verifier_runtime + elapsed;

        let now = Instant::now();
        let (l0, l1) = send_cross_terms_before_join(
            projected_lhs,
            &mut statement,
            &witness,
            &projected_witness,
        );

        let elapsed = now.elapsed();
        println_with_timestamp!("Time for send_cross_terms_before_join: {:.2?}", elapsed);
        prover_runtime = prover_runtime + elapsed;
        t_fold += elapsed;

        let now = Instant::now();
        verifier_state =
            verifier_join::<MOD_Q, N>(verifier_state, l0, l1, projection_rhs, projected_commitment);
        let elapsed = now.elapsed();
        println_with_timestamp!("Time for verifier_join: {:.2?}", elapsed);
        verifier_runtime = verifier_runtime + elapsed;

        let now = Instant::now();
        witness.append(&mut projected_witness);
        let elapsed = now.elapsed();
        prover_runtime = prover_runtime + elapsed;
        t_fold += elapsed;

        if FOLDING_SCHEME {
            let now = Instant::now();
            let (new_witness, bdecomp_output) = decomp_ell::<MOD_Q, N>(&statement, witness, 2);
            witness = new_witness;
            let elapsed = now.elapsed();
            println_with_timestamp!("Time for decomp: {:.2?}", elapsed);
            prover_runtime = prover_runtime + elapsed;

            t_decomp += elapsed;
            println_with_timestamp!(
                "Stmnt len {:?}, Witness  REPxLEN {:?}x{:?}  Verifier REPxLEN {:?}x{:?}",
                statement.len(),
                witness.len(),
                witness[0].len(),
                verifier_state.rhs.len(),
                verifier_state.rhs[0].len()
            );

            println_with_timestamp!(
                "bdedecomp: {:?} {:?}",
                bdecomp_output.rhs.len(),
                bdecomp_output.rhs[0].len()
            );
            let now = Instant::now();
            let new_verifier_state = verify_decomp::<MOD_Q, N>(bdecomp_output, &verifier_state);
            verifier_state = new_verifier_state;
            let elapsed = now.elapsed();

            println_with_timestamp!(
                "Verifier REPxLEN: {:?}x{:?}",
                verifier_state.rhs.len(),
                verifier_state.rhs[0].len()
            );

            println_with_timestamp!("Time for verify_decomp: {:.2?}", elapsed);
            verifier_runtime = verifier_runtime + elapsed;
            break;
        }
    }

    let now = Instant::now();

    println_with_timestamp!(
        "Final dim st {:?}, wrow {:} wcol {:?} vfrow {:?} vfcol {:?}",
        statement.len(),
        witness.len(),
        witness[0].len(),
        verifier_state.rhs.len(),
        verifier_state.rhs[0].len()
    );
    if !FOLDING_SCHEME {
        assert_eq!(
            parallel_dot_series_matrix(&statement, &witness),
            verifier_state.rhs
        );
        let elapsed = now.elapsed();
        println_with_timestamp!("Time for final assert_eq: {:.2?}", elapsed);
        verifier_runtime = verifier_runtime + elapsed;
    }

    println_with_timestamp!("PRV: {:.2?}", prover_runtime);
    println_with_timestamp!("VER: {:.2?}", verifier_runtime);
    println!("t_norm_check_1: {:.2?}", t_norm_check_1);
    println!("t_norm_check_2: {:.2?}", t_norm_check_2);
    println!("t_norm_check_3: {:.2?}", t_norm_check_3);
    println!("t_split: {:.2?}", t_split);
    println!("t_projection: {:.2?}", t_projection);
    println!("t_fold: {:.2?}", t_fold);
    println!("t_decomp: {:.2?}", t_decomp);
}

pub fn execute() {
    protocol::<MOD_Q, N>()
}
