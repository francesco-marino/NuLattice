"""Benchmark CCSD residuals emitted by AutoGraph against NuLattice CCSD.

The function below is the standalone ``--format=einsum`` output ccsdt from AutoGraph/Examples/CCSD.in.
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

def pA_BC(val):
    return val - contract('abcijk -> bacijk', val) - contract('abcijk -> cbaijk', val)

def pI_JK(val):
    return val - contract('abcijk -> abcjik', val) - contract('abcijk ->  abckji', val)


def get_ccsdt_residuals(
    f_hh, f_ph, f_pp,
    v_hhhh, v_phhh, v_phph, v_pphh, v_ppph, v_pppp,
    t1, t2, t3,
    truncation="ccsdt",
    time_func=False
    ):

    if time_func: start = perf_counter()

    O_ph = np.zeros_like(t1)
    O_pphh = np.zeros_like(t2)
    pnum,hnum = O_ph.shape
    O_ppphhh = np.zeros( (pnum,pnum,pnum, hnum,hnum,hnum) )
    
    O_ph += contract('ai -> ai', f_ph)
    O_ph += contract('bj, abij -> ai', np.conj(f_ph), t2)
    O_ph += contract('ab, bi -> ai', f_pp, t1)
    O_ph += -contract('ji, aj -> ai', f_hh, t1)
    O_ph += -contract('bj, bi, aj -> ai', np.conj(f_ph), t1, t1)
    O_ph += 0.25 * contract('bcjk, abcijk -> ai', np.conj(v_pphh), t3)
    O_ph += 0.5 * contract('bcaj, bcij -> ai', np.conj(v_ppph), t2)
    O_ph += 0.5 * contract('bijk, abjk -> ai', np.conj(v_phhh), t2)
    O_ph += -contract('ajbi, bj -> ai', v_phph, t1)
    O_ph += -0.5 * contract('bcjk, bi, acjk -> ai', np.conj(v_pphh), t1, t2)
    O_ph += -0.5 * contract('bcjk, aj, bcik -> ai', np.conj(v_pphh), t1, t2)
    O_ph += contract('bcjk, bj, acik -> ai', np.conj(v_pphh), t1, t2)
    O_ph += -contract('bcaj, bj, ci -> ai', np.conj(v_ppph), t1, t1)
    O_ph += -contract('bijk, bj, ak -> ai', np.conj(v_phhh), t1, t1)
    O_ph += -contract('bcjk, bj, ci, ak -> ai', np.conj(v_pphh), t1, t1, t1)
    O_pphh += contract('ck, abcijk -> abij', np.conj(f_ph), t3)
    O_pphh += pAB(contract('bc, acij -> abij', f_pp, t2))
    O_pphh += pIJ(contract('ki, abjk -> abij', f_hh, t2))
    O_pphh += pIJ(contract('ck, ci, abjk -> abij', np.conj(f_ph), t1, t2))
    O_pphh += pAB(-contract('ck, bk, acij -> abij', np.conj(f_ph), t1, t2))
    O_pphh += contract('abij -> abij', v_pphh)
    O_pphh += pAB(0.5 * contract('cdbk, acdijk -> abij', np.conj(v_ppph), t3))
    O_pphh += pIJ(-0.5 * contract('cikl, abcjkl -> abij', np.conj(v_phhh), t3))
    O_pphh += 0.5 * contract('abcd, cdij -> abij', v_pppp, t2)
    O_pphh += pAB(pIJ(contract('bkci, acjk -> abij', v_phph, t2)))
    O_pphh += 0.5 * contract('klij, abkl -> abij', v_hhhh, t2)
    O_pphh += pIJ(-contract('abci, cj -> abij', v_ppph, t1))
    O_pphh += pAB(contract('bkij, ak -> abij', v_phhh, t1))
    O_pphh += 0.25 * contract('cdkl, cdij, abkl -> abij', np.conj(v_pphh), t2, t2)
    O_pphh += pAB(pIJ(-0.5 * contract('cdkl, bcik, adjl -> abij', np.conj(v_pphh), t2, t2)))
    O_pphh += pIJ(-0.5 * contract('cdkl, cdik, abjl -> abij', np.conj(v_pphh), t2, t2))
    O_pphh += pAB(0.5 * contract('cdkl, bckl, adij -> abij', np.conj(v_pphh), t2, t2))
    O_pphh += pIJ(0.5 * contract('cdkl, ci, abdjkl -> abij', np.conj(v_pphh), t1, t3))
    O_pphh += pAB(-0.5 * contract('cdkl, bk, acdijl -> abij', np.conj(v_pphh), t1, t3))
    O_pphh += contract('cdkl, ck, abdijl -> abij', np.conj(v_pphh), t1, t3)
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
    
    if truncation=="ccsdt":
        O_ppphhh += pA_BC(contract('cd, abdijk -> abcijk', f_pp, t3))
        O_ppphhh += pI_JK(-contract('li, abcjkl -> abcijk', f_hh, t3))
        O_ppphhh += pA_BC(pI_JK(contract('dl, cdij, abkl -> abcijk', np.conj(f_ph), t2, t2)))
        O_ppphhh += pI_JK(-contract('dl, di, abcjkl -> abcijk', np.conj(f_ph), t1, t3))
        O_ppphhh += pA_BC(-contract('dl, cl, abdijk -> abcijk', np.conj(f_ph), t1, t3))
        O_ppphhh += pA_BC(0.5 * contract('bcde, adeijk -> abcijk', v_pppp, t3))
        O_ppphhh += pA_BC(pI_JK(-contract('cldi, abdjkl -> abcijk', v_phph, t3)))
        O_ppphhh += pI_JK(0.5 * contract('lmij, abcklm -> abcijk', v_hhhh, t3))
        O_ppphhh += pA_BC(pI_JK(contract('bcdi, adjk -> abcijk', v_ppph, t2)))
        O_ppphhh += pA_BC(pI_JK(contract('clij, abkl -> abcijk', v_phhh, t2)))
        O_ppphhh += pA_BC(pI_JK(0.5 * contract('delm, cdij, abeklm -> abcijk', np.conj(v_pphh), t2, t3)))
        O_ppphhh += pI_JK(0.25 * contract('delm, deij, abcklm -> abcijk', np.conj(v_pphh), t2, t3))
        O_ppphhh += pA_BC(pI_JK(0.5 * contract('delm, bcil, adejkm -> abcijk', np.conj(v_pphh), t2, t3)))
        O_ppphhh += pA_BC(pI_JK(contract('delm, cdil, abejkm -> abcijk', np.conj(v_pphh), t2, t3)))
        O_ppphhh += pI_JK(0.5 * contract('delm, deil, abcjkm -> abcijk', np.conj(v_pphh), t2, t3))
        O_ppphhh += pA_BC(0.25 * contract('delm, bclm, adeijk -> abcijk', np.conj(v_pphh), t2, t3))
        O_ppphhh += pA_BC(0.5 * contract('delm, cdlm, abeijk -> abcijk', np.conj(v_pphh), t2, t3))
        O_ppphhh += pA_BC(pI_JK(0.5 * contract('decl, deij, abkl -> abcijk', np.conj(v_ppph), t2, t2)))
        O_ppphhh += pI_JK(contract('decl, bdil, aejk -> abcijk', np.conj(v_ppph), t2, t2))
        O_ppphhh += pA_BC(-contract('dilm, cdjl, abkm -> abcijk', np.conj(v_phhh), t2, t2))
        O_ppphhh += pA_BC(pI_JK(0.5 * contract('dilm, bclm, adjk -> abcijk', np.conj(v_phhh), t2, t2)))
        O_ppphhh += pA_BC(pI_JK(contract('decl, di, abejkl -> abcijk', np.conj(v_ppph), t1, t3)))
        O_ppphhh += -0.5 * contract('dilm, dj, abcklm -> abcijk', np.conj(v_phhh), t1, t3)
        O_ppphhh += 0.5 * contract('decl, bl, adeijk -> abcijk', np.conj(v_ppph), t1, t3)
        O_ppphhh += pA_BC(-contract('decl, dl, abeijk -> abcijk', np.conj(v_ppph), t1, t3))
        O_ppphhh += pA_BC(pI_JK(contract('dilm, cl, abdjkm -> abcijk', np.conj(v_phhh), t1, t3)))
        O_ppphhh += pI_JK(-contract('dilm, dl, abcjkm -> abcijk', np.conj(v_phhh), t1, t3))
        O_ppphhh += pA_BC(pI_JK(-contract('bcde, di, aejk -> abcijk', v_pppp, t1, t2)))
        O_ppphhh += pA_BC(-contract('cldi, dj, abkl -> abcijk', v_phph, t1, t2))
        O_ppphhh += pI_JK(contract('cldi, bl, adjk -> abcijk', v_phph, t1, t2))
        O_ppphhh += pA_BC(pI_JK(-contract('lmij, cl, abkm -> abcijk', v_hhhh, t1, t2)))
        O_ppphhh += pA_BC(contract('delm, di, cejl, abkm -> abcijk', np.conj(v_pphh), t1, t2, t2))
        O_ppphhh += pA_BC(pI_JK(-0.5 * contract('delm, di, bclm, aejk -> abcijk', np.conj(v_pphh), t1, t2, t2)))
        O_ppphhh += pA_BC(pI_JK(-0.5 * contract('delm, cl, deij, abkm -> abcijk', np.conj(v_pphh), t1, t2, t2)))
        O_ppphhh += pA_BC(pI_JK(contract('delm, dl, ceij, abkm -> abcijk', np.conj(v_pphh), t1, t2, t2)))
        O_ppphhh += pI_JK(-contract('delm, cl, bdim, aejk -> abcijk', np.conj(v_pphh), t1, t2, t2))
        O_ppphhh += 0.25 * contract('delm, di, ej, abcklm -> abcijk', np.conj(v_pphh), t1, t1, t3)
        O_ppphhh += pA_BC(pI_JK(-contract('delm, di, cl, abejkm -> abcijk', np.conj(v_pphh), t1, t1, t3)))
        O_ppphhh += pI_JK(-contract('delm, dl, ei, abcjkm -> abcijk', np.conj(v_pphh), t1, t1, t3))
        O_ppphhh += -0.25 * contract('delm, cl, bm, adeijk -> abcijk', np.conj(v_pphh), t1, t1, t3)
        O_ppphhh += pA_BC(-contract('delm, dl, cm, abeijk -> abcijk', np.conj(v_pphh), t1, t1, t3))
        O_ppphhh += pA_BC(0.5 * contract('decl, di, ej, abkl -> abcijk', np.conj(v_ppph), t1, t1, t2))
        O_ppphhh += pI_JK(-contract('decl, di, bl, aejk -> abcijk', np.conj(v_ppph), t1, t1, t2))
        O_ppphhh += pA_BC(contract('dilm, dj, cl, abkm -> abcijk', np.conj(v_phhh), t1, t1, t2))
        O_ppphhh += pI_JK(-0.5 * contract('dilm, cl, bm, adjk -> abcijk', np.conj(v_phhh), t1, t1, t2))
        O_ppphhh += pA_BC(-0.5 * contract('delm, di, ej, cl, abkm -> abcijk', np.conj(v_pphh), t1, t1, t1, t2))
        O_ppphhh += pI_JK(0.5 * contract('delm, di, cl, bm, aejk -> abcijk', np.conj(v_pphh), t1, t1, t1, t2))

    if time_func:
        tot_time = perf_counter() - start
        print (f"{tot_time} s")

    return O_ph, O_pphh, O_ppphhh


def solve_ccsdt(fock_mats, two_body_int, 
    eps=1e-10, max_steps=500, mixing=0.4,
    truncation="ccsdt",
    verbose=True
    ):
    """Solve CCSD with the standalone ccsdt residuals and linear mixing."""
    f_pp, f_ph, f_hh = fock_mats
    v_pppp, v_ppph, v_pphh, v_phph, v_phhh, v_hhhh = two_body_int
    t1 = ccm.t1Init(f_ph, f_pp, f_hh, 0)
    t2 = ccm.t2Init(f_pp, f_hh, v_pphh, 0)
    pnum,hnum = t1.shape
    t3 = np.zeros((pnum,pnum,pnum,hnum,hnum,hnum))

    pp_ens  = np.add.outer(np.diag(f_pp), np.diag(f_pp))
    ppp_ens = np.add.outer(pp_ens, np.diag(f_pp))
    hh_ens  = np.add.outer(np.diag(f_hh), np.diag(f_hh))
    hhh_ens = np.add.outer(hh_ens, np.diag(f_hh))

    denom_t1 = np.add.outer(-np.diag(f_pp), np.diag(f_hh))
    denom_t2 = np.add.outer(-pp_ens,  hh_ens, )
    denom_t3 = np.add.outer(-ppp_ens, hhh_ens,)
    previous_energy = ccm.ccsd_energy(f_ph, v_pphh, t2, t1)

    for iteration in range(1, max_steps + 1):
        residual_t1, residual_t2, residual_t3 = get_ccsdt_residuals(
            f_hh, f_ph, f_pp, 
            v_hhhh, v_phhh, v_phph, v_pphh, v_ppph, v_pppp,
            t1, t2, t3,
            truncation
        )
        
        t1 = mixing * t1 + (1.0 - mixing) * (t1 + residual_t1 / denom_t1)
        t2 = mixing * t2 + (1.0 - mixing) * (t2 + residual_t2 / denom_t2)
        if truncation=="ccsdt":
            t3 = mixing * t3 + (1.0 - mixing) * (t3 + residual_t3 / denom_t3)
        energy = ccm.ccsd_energy(f_ph, v_pphh, t2, t1)

        if verbose:
            tmp_str = f"iteration={iteration}   energy={energy:.8f}"
            print (tmp_str)

        if abs(energy - previous_energy) / abs(energy) < eps:
            return energy, t1, t2, t3, iteration
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

    this_l, lattice_spacing = 2, 2.5
    vT1 = -9.0
    vS1 = -9.0
    w3  = 6.0

    truncation = "ccsd"
       
    reference_energy, fock_mats, two_body_int = ccm.get_norm_ord_int(
        this_l, ref.ref_16O_gs, vT1, vS1, w3, sparse=True
    )
    two_body_int = dense_two_body_blocks(fock_mats, two_body_int)

    print (f"Coupled-cluster truncation: {truncation}")    
    start = perf_counter()

    ccsdt = solve_ccsdt(deepcopy(fock_mats), deepcopy(two_body_int), truncation=truncation, verbose=False)
    corr_gen, t1_out, t2_out, t3_out, iterations_count = ccsdt
    cc_time = perf_counter() - start

    unit = lat.phys_unit(lattice_spacing)
    #print(f"NuLattice: {reference_time:.3f} s, correlation = {corr_ref}, total = {(corr_ref + reference_energy) * unit}")
    print(f"AutoGraph: {cc_time:.3f} s, {iterations_count} iterations, correlation = {corr_gen}, total = {(corr_gen + reference_energy) * unit}")
    #print(f"max |delta t1| = {np.max(np.abs(t1_ref - t1_gen))}")
    #print(f"max |delta t2| = {np.max(np.abs(t2_ref - t2_gen))}")
    #np.testing.assert_allclose(corr_gen, corr_ref, rtol=1e-8, atol=1e-8)
    #np.testing.assert_allclose(t1_gen, t1_ref, rtol=1e-7, atol=1e-8)
    #np.testing.assert_allclose(t2_gen, t2_ref, rtol=1e-7, atol=1e-8)
    
if __name__ == "__main__":
    main()

