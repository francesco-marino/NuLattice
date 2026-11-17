"""Benchmark CCSD residuals emitted by AutoGraph against NuLattice CCSD.

The function below is the standalone ``--format=einsum`` output generated from AutoGraph/Examples/CCSD.in.
(See https://github.com/francesco-marino/AutoGraph.git repository.)
"""

from copy import deepcopy
from time import perf_counter
import pathlib
import sys

import numpy as np
from opt_einsum import contract

sys.path.append(str(pathlib.Path(__file__).parent / ".."))
import NuLattice.CCM.coupled_cluster as ccm
import NuLattice.lattice as lat
import NuLattice.references as ref


def pAB(val):
    return val - contract('abij -> baij', val)


def pIJ(val):
    return val - contract('abij -> abji', val)


def generated_ccsd_residuals(f_hh, f_ph, f_pp, 
        t1, t2, 
        v_hhhh, v_phhh, v_phph, v_pphh, v_ppph, v_pppp
    ):
    O_ph = np.zeros_like(t1)
    O_pphh = np.zeros_like(t2)
    O_ph += contract('ai -> ai', f_ph)
    O_ph += contract('bj, abij -> ai', np.conj(f_ph), t2)
    O_ph += contract('ab, bi -> ai', f_pp, t1)
    O_ph += -contract('ji, aj -> ai', f_hh, t1)
    O_ph += -contract('bj, bi, aj -> ai', np.conj(f_ph), t1, t1)
    O_ph += 0.5 * contract('bcaj, bcij -> ai', np.conj(v_ppph), t2)
    O_ph += 0.5 * contract('bijk, abjk -> ai', np.conj(v_phhh), t2)
    O_ph += -contract('ajbi, bj -> ai', v_phph, t1)
    O_ph += -0.5 * contract('bcjk, bi, acjk -> ai', np.conj(v_pphh), t1, t2)
    O_ph += -0.5 * contract('bcjk, aj, bcik -> ai', np.conj(v_pphh), t1, t2)
    O_ph += contract('bcjk, bj, acik -> ai', np.conj(v_pphh), t1, t2)
    O_ph += -contract('bcaj, bj, ci -> ai', np.conj(v_ppph), t1, t1)
    O_ph += -contract('bijk, bj, ak -> ai', np.conj(v_phhh), t1, t1)
    O_ph += -contract('bcjk, bj, ci, ak -> ai', np.conj(v_pphh), t1, t1, t1)
    O_pphh += pAB(contract('bc, acij -> abij', f_pp, t2))
    O_pphh += pIJ(contract('ki, abjk -> abij', f_hh, t2))
    O_pphh += pIJ(contract('ck, ci, abjk -> abij', np.conj(f_ph), t1, t2))
    O_pphh += pAB(-contract('ck, bk, acij -> abij', np.conj(f_ph), t1, t2))
    O_pphh += contract('abij -> abij', v_pphh)
    O_pphh += 0.5 * contract('abcd, cdij -> abij', v_pppp, t2)
    O_pphh += pAB(pIJ(contract('bkci, acjk -> abij', v_phph, t2)))
    O_pphh += 0.5 * contract('klij, abkl -> abij', v_hhhh, t2)
    O_pphh += pIJ(-contract('abci, cj -> abij', v_ppph, t1))
    O_pphh += pAB(contract('bkij, ak -> abij', v_phhh, t1))
    O_pphh += 0.25 * contract('cdkl, cdij, abkl -> abij', np.conj(v_pphh), t2, t2)
    O_pphh += pAB(pIJ(-0.5 * contract('cdkl, bcik, adjl -> abij', np.conj(v_pphh), t2, t2)))
    O_pphh += pIJ(-0.5 * contract('cdkl, cdik, abjl -> abij', np.conj(v_pphh), t2, t2))
    O_pphh += pAB(0.5 * contract('cdkl, bckl, adij -> abij', np.conj(v_pphh), t2, t2))
    O_pphh += pAB(pIJ(-contract('cdbk, ci, adjk -> abij', np.conj(v_ppph), t1, t2)))
    O_pphh += pIJ(-0.5 * contract('cikl, cj, abkl -> abij', np.conj(v_phhh), t1, t2))
    O_pphh += pAB(0.5 * contract('cdbk, ak, cdij -> abij', np.conj(v_ppph), t1, t2))
    O_pphh += pAB(-contract('cdbk, ck, adij -> abij', np.conj(v_ppph), t1, t2))
    O_pphh += pAB(pIJ(-contract('cikl, bk, acjl -> abij', np.conj(v_phhh), t1, t2)))
    O_pphh += pIJ(contract('cikl, ck, abjl -> abij', np.conj(v_phhh), t1, t2))
    O_pphh += pIJ(0.5 * contract('abcd, ci, dj -> abij', v_pppp, t1, t1))
    O_pphh += pAB(pIJ(-contract('bkci, cj, ak -> abij', v_phph, t1, t1)))
    O_pphh += pAB(-0.5 * contract('klij, bk, al -> abij', v_hhhh, t1, t1))
    O_pphh += pIJ(0.25 * contract('cdkl, ci, dj, abkl -> abij', np.conj(v_pphh), t1, t1, t2))
    O_pphh += pAB(pIJ(contract('cdkl, ci, bk, adjl -> abij', np.conj(v_pphh), t1, t1, t2)))
    O_pphh += pIJ(contract('cdkl, ck, di, abjl -> abij', np.conj(v_pphh), t1, t1, t2))
    O_pphh += pAB(-0.25 * contract('cdkl, bk, al, cdij -> abij', np.conj(v_pphh), t1, t1, t2))
    O_pphh += pAB(-contract('cdkl, ck, bl, adij -> abij', np.conj(v_pphh), t1, t1, t2))
    O_pphh += pAB(pIJ(0.5 * contract('cdbk, ci, dj, ak -> abij', np.conj(v_ppph), t1, t1, t1)))
    O_pphh += pAB(pIJ(0.5 * contract('cikl, cj, bk, al -> abij', np.conj(v_phhh), t1, t1, t1)))
    O_pphh += pAB(pIJ(-0.25 * contract('cdkl, ci, dj, bk, al -> abij', np.conj(v_pphh), t1, t1, t1, t1)))
    return O_ph, O_pphh



def solve_generated(fock_mats, two_body_int, eps=1e-10, max_steps=500, mixing=0.4):
    """Solve CCSD with the standalone generated residuals and linear mixing."""
    f_pp, f_ph, f_hh = fock_mats
    v_pppp, v_ppph, v_pphh, v_phph, v_phhh, v_hhhh = two_body_int
    t1 = ccm.t1Init(f_ph, f_pp, f_hh, 0)
    t2 = ccm.t2Init(f_pp, f_hh, v_pphh, 0)
    denom_t1 = np.add.outer(-np.diag(f_pp), np.diag(f_hh))
    denom_t2 = np.add.outer(
        -np.add.outer(np.diag(f_pp), np.diag(f_pp)),
        np.add.outer(np.diag(f_hh), np.diag(f_hh)),
    )
    previous_energy = ccm.ccsd_energy(f_ph, v_pphh, t2, t1)

    for iteration in range(1, max_steps + 1):
        residual_t1, residual_t2 = generated_ccsd_residuals(
            f_hh, f_ph, f_pp, t1, t2,
            v_hhhh, v_phhh, v_phph, v_pphh, v_ppph, v_pppp,
        )
        
        t1 = mixing * t1 + (1.0 - mixing) * (t1 + residual_t1 / denom_t1)
        t2 = mixing * t2 + (1.0 - mixing) * (t2 + residual_t2 / denom_t2)
        energy = ccm.ccsd_energy(f_ph, v_pphh, t2, t1)

        if abs(energy - previous_energy) / abs(energy) < eps:
            return energy, t1, t2, iteration
        previous_energy = energy
    raise RuntimeError("Generated CCSD solver did not converge.")


def dense_two_body_blocks(fock_mats, two_body_int):
    """Prepares dense pppp and ppph blocks from corresponding sparse structures."""
    f_pp, _, f_hh = fock_mats
    pnum, hnum = f_pp.shape[0], f_hh.shape[0]
    v_pppp_sparse, v_ppph_sparse, *remaining = two_body_int
    dtype = np.result_type(f_pp, *remaining)
    v_pppp = np.zeros((pnum, pnum, pnum, pnum), dtype=dtype)
    v_ppph = np.zeros((pnum, pnum, pnum, hnum), dtype=dtype)
    for a, b, c, d, value in v_pppp_sparse:
        v_pppp[a, b, c, d] += value
    for a, b, c, d, value in v_ppph_sparse:
        v_ppph[a, b, c, d] += value
    return [v_pppp, v_ppph, *remaining]


def main():
    
    #
    # Same parameters as Example_CCSD.py
    #
    
    #Defining the lattice, here, 2x2x2 lattice with lattice spacing of 2.5 fm
    this_l, lattice_spacing = 2, 2.5
    
    #two-body contacts
    vT1 = -9.0; vS1 = -9.0
    #three-body contact
    w3  = 6.0
       
    reference_energy, fock_mats, two_body_int = ccm.get_norm_ord_int(
        this_l, ref.ref_16O_gs, vT1, vS1, w3, sparse=True
    )
    two_body_int = dense_two_body_blocks(fock_mats, two_body_int)

    start = perf_counter()
    reference = ccm.ccsd_solver(
        deepcopy(fock_mats), deepcopy(two_body_int), eps=1e-10,
        maxSteps=500, max_diis=10, mixing=0.5, sparse=False,
    )
    reference_time = perf_counter() - start
    
    start = perf_counter()
    generated = solve_generated(deepcopy(fock_mats), deepcopy(two_body_int))
    generated_time = perf_counter() - start
    corr_ref, t1_ref, t2_ref = reference
    corr_gen, t1_gen, t2_gen, generated_iterations = generated

    unit = lat.phys_unit(lattice_spacing)
    print(f"NuLattice: {reference_time:.3f} s, correlation = {corr_ref}, total = {(corr_ref + reference_energy) * unit}")
    print(f"AutoGraph: {generated_time:.3f} s, {generated_iterations} iterations, correlation = {corr_gen}, total = {(corr_gen + reference_energy) * unit}")
    print(f"max |delta t1| = {np.max(np.abs(t1_ref - t1_gen))}")
    print(f"max |delta t2| = {np.max(np.abs(t2_ref - t2_gen))}")
    np.testing.assert_allclose(corr_gen, corr_ref, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(t1_gen, t1_ref, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(t2_gen, t2_ref, rtol=1e-7, atol=1e-8)
    
if __name__ == "__main__":
    main()

