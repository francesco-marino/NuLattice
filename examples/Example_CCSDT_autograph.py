"""CCSDT calculation based on AutoGraph-generated coupled-cluster diagrams. 
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

# --- Permutation (antisymmetrizer) operators for the einsum equations ---
# Doubles axes: (a, b, i, j).  Triples axes: (a, b, c, i, j, k).
def pAB(x):
    return x - x.transpose(1, 0, 2, 3)
def pIJ(x):
    return x - x.transpose(0, 1, 3, 2)
def pA_BC(x):
    return x - x.transpose(1, 0, 2, 3, 4, 5) - x.transpose(2, 1, 0, 3, 4, 5)
def pB_AC(x):
    return x - x.transpose(1, 0, 2, 3, 4, 5) - x.transpose(0, 2, 1, 3, 4, 5)
def pC_AB(x):
    return x - x.transpose(2, 1, 0, 3, 4, 5) - x.transpose(0, 2, 1, 3, 4, 5)
def pI_JK(x):
    return x - x.transpose(0, 1, 2, 4, 3, 5) - x.transpose(0, 1, 2, 5, 4, 3)
def pJ_IK(x):
    return x - x.transpose(0, 1, 2, 4, 3, 5) - x.transpose(0, 1, 2, 3, 5, 4)
def pK_IJ(x):
    return x - x.transpose(0, 1, 2, 5, 4, 3) - x.transpose(0, 1, 2, 3, 5, 4)
def pABC(x):
    return (x
            - x.transpose(1, 0, 2, 3, 4, 5)
            - x.transpose(2, 1, 0, 3, 4, 5)
            - x.transpose(0, 2, 1, 3, 4, 5)
            + x.transpose(1, 2, 0, 3, 4, 5)
            + x.transpose(2, 0, 1, 3, 4, 5))
def pIJK(x):
    return (x
            - x.transpose(0, 1, 2, 4, 3, 5)
            - x.transpose(0, 1, 2, 5, 4, 3)
            - x.transpose(0, 1, 2, 3, 5, 4)
            + x.transpose(0, 1, 2, 4, 5, 3)
            + x.transpose(0, 1, 2, 5, 3, 4))

def get_ccsdt_residuals(
    f_hh, f_ph, f_pp,
    v_hhhh, v_phhh, v_phph, v_pphh, v_ppph, v_pppp,
    t1, t2, t3,
    truncation="ccsdt",
    time_func=False
    ):

    if time_func: start = perf_counter()
    need_t3 = (truncation in ["ccsdt",])

    O_ph = np.zeros_like(t1)
    O_pphh = np.zeros_like(t2)
    pnum,hnum = O_ph.shape
    O_ppphhh = np.zeros( (pnum,pnum,pnum, hnum,hnum,hnum) )

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
    if need_t3:
        O_ph += 0.25 * contract('bcjk, abcijk -> ai', np.conj(v_pphh), t3)

    O_pphh += pAB(contract('bc, acij -> abij', f_pp, t2))
    O_pphh += pIJ(contract('ki, abjk -> abij', f_hh, t2))
    O_pphh += pIJ(contract('ck, ci, abjk -> abij', np.conj(f_ph), t1, t2))
    O_pphh += -pAB(contract('ck, bk, acij -> abij', np.conj(f_ph), t1, t2))
    O_pphh += contract('abij -> abij', v_pphh)
    O_pphh += 0.5 * contract('abcd, cdij -> abij', v_pppp, t2)
    O_pphh += pAB(pIJ(contract('bkci, acjk -> abij', v_phph, t2)))
    O_pphh += 0.5 * contract('klij, abkl -> abij', v_hhhh, t2)
    O_pphh += -pIJ(contract('abci, cj -> abij', v_ppph, t1))
    O_pphh += pAB(contract('bkij, ak -> abij', v_phhh, t1))
    O_pphh += 0.25 * contract('cdkl, cdij, abkl -> abij', np.conj(v_pphh), t2, t2)
    O_pphh += -0.5 * pAB(pIJ(contract('cdkl, bcik, adjl -> abij', np.conj(v_pphh), t2, t2)))
    O_pphh += -0.5 * pIJ(contract('cdkl, cdik, abjl -> abij', np.conj(v_pphh), t2, t2))
    O_pphh += 0.5 * pAB(contract('cdkl, bckl, adij -> abij', np.conj(v_pphh), t2, t2))
    O_pphh += -pAB(pIJ(contract('cdbk, ci, adjk -> abij', np.conj(v_ppph), t1, t2)))
    O_pphh += -0.5 * pIJ(contract('cikl, cj, abkl -> abij', np.conj(v_phhh), t1, t2))
    O_pphh += 0.5 * pAB(contract('cdbk, ak, cdij -> abij', np.conj(v_ppph), t1, t2))
    O_pphh += -pAB(contract('cdbk, ck, adij -> abij', np.conj(v_ppph), t1, t2))
    O_pphh += -pAB(pIJ(contract('cikl, bk, acjl -> abij', np.conj(v_phhh), t1, t2)))
    O_pphh += pIJ(contract('cikl, ck, abjl -> abij', np.conj(v_phhh), t1, t2))
    O_pphh += 0.5 * pIJ(contract('abcd, ci, dj -> abij', v_pppp, t1, t1))
    O_pphh += -pAB(pIJ(contract('bkci, cj, ak -> abij', v_phph, t1, t1)))
    O_pphh += -0.5 * pAB(contract('klij, bk, al -> abij', v_hhhh, t1, t1))
    O_pphh += 0.25 * pIJ(contract('cdkl, ci, dj, abkl -> abij', np.conj(v_pphh), t1, t1, t2))
    O_pphh += pAB(pIJ(contract('cdkl, ci, bk, adjl -> abij', np.conj(v_pphh), t1, t1, t2)))
    O_pphh += pIJ(contract('cdkl, ck, di, abjl -> abij', np.conj(v_pphh), t1, t1, t2))
    O_pphh += -0.25 * pAB(contract('cdkl, bk, al, cdij -> abij', np.conj(v_pphh), t1, t1, t2))
    O_pphh += -pAB(contract('cdkl, ck, bl, adij -> abij', np.conj(v_pphh), t1, t1, t2))
    O_pphh += 0.5 * pAB(pIJ(contract('cdbk, ci, dj, ak -> abij', np.conj(v_ppph), t1, t1, t1)))
    O_pphh += 0.5 * pAB(pIJ(contract('cikl, cj, bk, al -> abij', np.conj(v_phhh), t1, t1, t1)))
    O_pphh += -0.25 * pAB(pIJ(contract('cdkl, ci, dj, bk, al -> abij', np.conj(v_pphh), t1, t1, t1, t1)))
    if need_t3:
        O_pphh += contract('ck, abcijk -> abij', np.conj(f_ph), t3)
        O_pphh += 0.5 * pIJ(contract('cdkl, ci, abdjkl -> abij', np.conj(v_pphh), t1, t3))
        O_pphh += -0.5 * pAB(contract('cdkl, bk, acdijl -> abij', np.conj(v_pphh), t1, t3))
        O_pphh += contract('cdkl, ck, abdijl -> abij', np.conj(v_pphh), t1, t3)
        O_pphh += 0.5 * pAB(contract('cdbk, acdijk -> abij', np.conj(v_ppph), t3))
        O_pphh += -0.5 * pIJ(contract('cikl, abcjkl -> abij', np.conj(v_phhh), t3))
    
    if need_t3:
        O_ppphhh += pC_AB(contract('cd, abdijk -> abcijk', f_pp, t3))
        O_ppphhh += -pI_JK(contract('li, abcjkl -> abcijk', f_hh, t3))
        O_ppphhh += pC_AB(pK_IJ(contract('dl, cdij, abkl -> abcijk', np.conj(f_ph), t2, t2)))
        O_ppphhh += -pI_JK(contract('dl, di, abcjkl -> abcijk', np.conj(f_ph), t1, t3))
        O_ppphhh += -pC_AB(contract('dl, cl, abdijk -> abcijk', np.conj(f_ph), t1, t3))
        O_ppphhh += 0.5 * pA_BC(contract('bcde, adeijk -> abcijk', v_pppp, t3))
        O_ppphhh += -pC_AB(pI_JK(contract('cldi, abdjkl -> abcijk', v_phph, t3)))
        O_ppphhh += 0.5 * pK_IJ(contract('lmij, abcklm -> abcijk', v_hhhh, t3))
        O_ppphhh += pA_BC(pI_JK(contract('bcdi, adjk -> abcijk', v_ppph, t2)))
        O_ppphhh += pC_AB(pK_IJ(contract('clij, abkl -> abcijk', v_phhh, t2)))
        O_ppphhh += 0.5 * pC_AB(pK_IJ(contract('delm, cdij, abeklm -> abcijk', np.conj(v_pphh), t2, t3)))
        O_ppphhh += 0.25 * pK_IJ(contract('delm, deij, abcklm -> abcijk', np.conj(v_pphh), t2, t3))
        O_ppphhh += 0.5 * pA_BC(pI_JK(contract('delm, bcil, adejkm -> abcijk', np.conj(v_pphh), t2, t3)))
        O_ppphhh += pC_AB(pI_JK(contract('delm, cdil, abejkm -> abcijk', np.conj(v_pphh), t2, t3)))
        O_ppphhh += 0.5 * pI_JK(contract('delm, deil, abcjkm -> abcijk', np.conj(v_pphh), t2, t3))
        O_ppphhh += 0.25 * pA_BC(contract('delm, bclm, adeijk -> abcijk', np.conj(v_pphh), t2, t3))
        O_ppphhh += 0.5 * pC_AB(contract('delm, cdlm, abeijk -> abcijk', np.conj(v_pphh), t2, t3))
        O_ppphhh += 0.5 * pC_AB(pK_IJ(contract('decl, deij, abkl -> abcijk', np.conj(v_ppph), t2, t2)))
        O_ppphhh += pABC(pI_JK(contract('decl, bdil, aejk -> abcijk', np.conj(v_ppph), t2, t2)))
        O_ppphhh += -pC_AB(pIJK(contract('dilm, cdjl, abkm -> abcijk', np.conj(v_phhh), t2, t2)))
        O_ppphhh += 0.5 * pA_BC(pI_JK(contract('dilm, bclm, adjk -> abcijk', np.conj(v_phhh), t2, t2)))
        O_ppphhh += pC_AB(pI_JK(contract('decl, di, abejkl -> abcijk', np.conj(v_ppph), t1, t3)))
        O_ppphhh += -0.5 * pIJK(contract('dilm, dj, abcklm -> abcijk', np.conj(v_phhh), t1, t3))
        O_ppphhh += 0.5 * pABC(contract('decl, bl, adeijk -> abcijk', np.conj(v_ppph), t1, t3))
        O_ppphhh += -pC_AB(contract('decl, dl, abeijk -> abcijk', np.conj(v_ppph), t1, t3))
        O_ppphhh += pC_AB(pI_JK(contract('dilm, cl, abdjkm -> abcijk', np.conj(v_phhh), t1, t3)))
        O_ppphhh += -pI_JK(contract('dilm, dl, abcjkm -> abcijk', np.conj(v_phhh), t1, t3))
        O_ppphhh += -pA_BC(pI_JK(contract('bcde, di, aejk -> abcijk', v_pppp, t1, t2)))
        O_ppphhh += -pC_AB(pIJK(contract('cldi, dj, abkl -> abcijk', v_phph, t1, t2)))
        O_ppphhh += pABC(pI_JK(contract('cldi, bl, adjk -> abcijk', v_phph, t1, t2)))
        O_ppphhh += -pC_AB(pK_IJ(contract('lmij, cl, abkm -> abcijk', v_hhhh, t1, t2)))
        O_ppphhh += pC_AB(pIJK(contract('delm, di, cejl, abkm -> abcijk', np.conj(v_pphh), t1, t2, t2)))
        O_ppphhh += -0.5 * pA_BC(pI_JK(contract('delm, di, bclm, aejk -> abcijk', np.conj(v_pphh), t1, t2, t2)))
        O_ppphhh += -0.5 * pC_AB(pK_IJ(contract('delm, cl, deij, abkm -> abcijk', np.conj(v_pphh), t1, t2, t2)))
        O_ppphhh += pC_AB(pK_IJ(contract('delm, dl, ceij, abkm -> abcijk', np.conj(v_pphh), t1, t2, t2)))
        O_ppphhh += -pABC(pI_JK(contract('delm, cl, bdim, aejk -> abcijk', np.conj(v_pphh), t1, t2, t2)))
        O_ppphhh += 0.25 * pIJK(contract('delm, di, ej, abcklm -> abcijk', np.conj(v_pphh), t1, t1, t3))
        O_ppphhh += -pC_AB(pI_JK(contract('delm, di, cl, abejkm -> abcijk', np.conj(v_pphh), t1, t1, t3)))
        O_ppphhh += -pI_JK(contract('delm, dl, ei, abcjkm -> abcijk', np.conj(v_pphh), t1, t1, t3))
        O_ppphhh += -0.25 * pABC(contract('delm, cl, bm, adeijk -> abcijk', np.conj(v_pphh), t1, t1, t3))
        O_ppphhh += -pC_AB(contract('delm, dl, cm, abeijk -> abcijk', np.conj(v_pphh), t1, t1, t3))
        O_ppphhh += 0.5 * pC_AB(pIJK(contract('decl, di, ej, abkl -> abcijk', np.conj(v_ppph), t1, t1, t2)))
        O_ppphhh += -pABC(pI_JK(contract('decl, di, bl, aejk -> abcijk', np.conj(v_ppph), t1, t1, t2)))
        O_ppphhh += pC_AB(pIJK(contract('dilm, dj, cl, abkm -> abcijk', np.conj(v_phhh), t1, t1, t2)))
        O_ppphhh += -0.5 * pABC(pI_JK(contract('dilm, cl, bm, adjk -> abcijk', np.conj(v_phhh), t1, t1, t2)))
        O_ppphhh += -0.5 * pC_AB(pIJK(contract('delm, di, ej, cl, abkm -> abcijk', np.conj(v_pphh), t1, t1, t1, t2)))
        O_ppphhh += 0.5 * pABC(pI_JK(contract('delm, di, cl, bm, aejk -> abcijk', np.conj(v_pphh), t1, t1, t1, t2)))
            
    if time_func:
        tot_time = perf_counter() - start
        print (f"{tot_time} s")

    return O_ph, O_pphh, O_ppphhh


def solve_ccsdt(fock_mats, two_body_int, 
    eps=1e-8,   # Relative error on energies 
    max_steps=500, 
    mixing=0.6, 
    mixing_t3=0.7,
    truncation="ccsdt",
    verbose=True
    ):
    """Solve CCSD with the standalone ccsdt residuals and linear mixing."""
    f_pp, f_ph, f_hh = fock_mats
    v_pppp, v_ppph, v_pphh, v_phph, v_phhh, v_hhhh = two_body_int
    t1 = ccm.t1Init(f_ph, f_pp, f_hh, 0)
    t2 = ccm.t2Init(f_pp, f_hh, v_pphh, 0)
    pnum, hnum = t1.shape
    t3 = np.zeros((pnum,pnum,pnum, hnum,hnum,hnum))

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
            t3 = mixing_t3 * t3 + (1.0 - mixing_t3) * (t3 + residual_t3 / denom_t3)
        energy = ccm.ccsd_energy(f_ph, v_pphh, t2, t1)

        if verbose:
            tmp_str = f"iteration={iteration}   correlation energy={energy:.8f}"
            print (tmp_str)

        rel_diff = abs(energy - previous_energy) / abs(energy)
        if rel_diff < eps:
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

    truncation = "ccsdt"

    eps = 1e-10 if truncation=="ccsd" else 1e-4
       
    reference_energy, fock_mats, two_body_int = ccm.get_norm_ord_int(
        this_l, ref.ref_16O_gs, vT1, vS1, w3, sparse=True
    )
    two_body_int = dense_two_body_blocks(fock_mats, two_body_int)

    print (f"Coupled-cluster truncation: {truncation}")    
    start = perf_counter()

    ccsdt = solve_ccsdt(deepcopy(fock_mats), deepcopy(two_body_int), truncation=truncation, verbose=True, mixing=0.4)
    corr_gen, t1_out, t2_out, t3_out, iterations_count = ccsdt
    cc_time = perf_counter() - start

    unit = lat.phys_unit(lattice_spacing)
    print(f"AutoGraph: {cc_time:.3f} s, {iterations_count} iterations, correlation = {corr_gen}, total = {(corr_gen + reference_energy) * unit}")
    
if __name__ == "__main__":
    main()

