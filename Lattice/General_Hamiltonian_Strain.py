from Lattice.General_Hamiltonian import *

def strain_conn(qax, r1vals, r2vals, r1signs, r2signs):
    return np.exp(qax * (r1signs * r1vals**2 + r2signs * r2vals**2))

def get_interval_rvals(rstart, rend, num_sites):
    rvals = np.zeros(num_sites)
    rvals[0] = rstart
    rvals[-1] = rend
    if num_sites % 2 == 1:
        rvals[:int((len(rvals) - 1)/2)] = np.arange(rstart, rstart + int((len(rvals) - 1)/2))
        rvals[int((len(rvals) - 1)/2 + 1):] = np.flip(np.arange(rend, rend + int((len(rvals) - 1)/2)))
        rvals[int((num_sites - 1) / 2)] = np.max(rvals) + 1
    else:
        rvals[:int(len(rvals)/2)] = np.arange(rstart, rstart + int(len(rvals)/2))
        rvals[int(len(rvals) / 2):] = np.flip(np.arange(rend, rend + int(len(rvals) / 2)))
    return rvals

# Copy of general_q3_hamiltonian_superoptimized from Fundamental.General_Hamiltonian
# But with modifications to accomodate strain
def general_q3_hamiltonian_with_strain(p, num_levels, qax): ##Extra argument qax for strength of uniform axial field
    sites_per_level, tot_num_sites = number_points_q3_general_from_repeating_pattern(p, num_levels)
    first_sites_of_each_level = np.array([0])
    for n in range(num_levels):
        first_sites_of_each_level = np.append(first_sites_of_each_level, sites_per_level[n] + np.sum(sites_per_level[:n]))

    ham = dok_matrix((tot_num_sites, tot_num_sites), dtype=np.longdouble)
    t = 1

    if num_levels >= 0: #Have to hard code first gen
        sion = np.arange(first_sites_of_each_level[0], first_sites_of_each_level[1])

        #More optimized NN intra layer hopping
        sion_one_above = sion + 1
        sion_one_above[-1] = sion[0]
        ham[sion, sion_one_above] = t
        ham[sion_one_above, sion] = t

        #Hard code inter generation connection

        first_gen_inter_connect_hardcode = np.arange(first_sites_of_each_level[1], first_sites_of_each_level[2], (p-3))

        ## Modified to accomodate strain
        r1vals = np.repeat(0, len(sion))
        r2vals = np.repeat(1, len(first_gen_inter_connect_hardcode))
        if p % 2 == 0:
            r1signs = np.tile([1, -1], int(p/2))
        else:
            raise ValueError
        r2signs = -1 * r1signs
        ham[sion, first_gen_inter_connect_hardcode] = t * strain_conn(qax, r1vals, r2vals, r1signs, r2signs)
        ham[first_gen_inter_connect_hardcode, sion] = t * strain_conn(qax, r1vals, r2vals, r1signs, r2signs)

        ## Define variables to carry over needed information for next iteration
        if 2 * len(first_gen_inter_connect_hardcode) / p == round(2 * len(first_gen_inter_connect_hardcode) / p, 0):
            anchor_sites = first_gen_inter_connect_hardcode[:int(2 * len(first_gen_inter_connect_hardcode) / p + 1)]
        else:
            raise ValueError
        if 2 * len(r2vals) / p == round(2 * len(r2vals) / p, 0):
            anchor_rvals = r2vals[:int(2 * len(r2vals) / p + 1)]
        else:
            raise ValueError
        if 2 * len(r2signs) / p == round(2 * len(r2signs) / p, 0):
            anchor_signs = r2signs[:int(2 * len(r2signs) / p + 1)]
        else:
            raise ValueError


    for n in range(1, num_levels-1): #iterate over all levels except first and last level
        print('Working on generation: {}'.format(n+1))
        #sion stands for: site_indices_on_level
        sion = np.arange(first_sites_of_each_level[n], first_sites_of_each_level[n+1])

        # More optimized NN intra layer hopping
        ## Leverage p/2 fold rotational symmetry of strain texture for optimization
        if 2 * len(sion) / p == round(2 * len(sion) / p, 0):
            rel_sector_sites = sion[:int(2 * len(sion) / p)]
        else:
            raise ValueError

        ## Modified to include strain
        total_interval_intrahops = np.array([])
        total_rvals = np.array([])
        total_rsigns = np.array([])
        for ind in range(len(anchor_sites) - 1):
            num_interval_sites = anchor_sites[ind + 1] - anchor_sites[ind] + 1
            interval_sites = np.arange(anchor_sites[ind], anchor_sites[ind + 1] + 1)
            rel_interval_rvals = get_interval_rvals(anchor_rvals[ind], anchor_rvals[ind + 1], num_interval_sites)

            if ind == 0: # if check to avoid overlap
                total_rvals = np.append(total_rvals, rel_interval_rvals)
            else:
                total_rvals = np.append(total_rvals, rel_interval_rvals[1:])

            if num_interval_sites % 2 == 0:
                rel_interval_rsigns = np.tile([anchor_signs[ind], -1 * anchor_signs[ind]],
                                              int(num_interval_sites/2))
            else:
                rel_interval_rsigns = np.tile([anchor_signs[ind], -1 * anchor_signs[ind]],
                                              int((num_interval_sites - 1)/2))
                rel_interval_rsigns = np.append(rel_interval_rsigns, anchor_signs[ind + 1])

            if ind == 0: # if check to avoid overlap
                total_rsigns = np.append(total_rsigns, rel_interval_rsigns)
            else:
                total_rsigns = np.append(total_rsigns, rel_interval_rsigns[1:])

            interval_intrahops = strain_conn(qax, rel_interval_rvals[:-1], rel_interval_rvals[1:],
                                             rel_interval_rsigns[:-1], rel_interval_rsigns[1:])

            ham[interval_sites[:-1], interval_sites[1:]] = interval_intrahops
            ham[interval_sites[1:], interval_sites[:-1]] = interval_intrahops

            total_interval_intrahops = np.append(total_interval_intrahops, interval_intrahops)

        if (p - 2) % 2 == 0:
            leftover_sites = np.arange(rel_sector_sites[-1] + 1, first_sites_of_each_level[n+1])
            leftover_sites = np.append(leftover_sites, sion[0])
            ham[leftover_sites[:-1], leftover_sites[1:]] = np.tile(total_interval_intrahops, int((p - 2)/2))
            ham[leftover_sites[1:], leftover_sites[:-1]] = np.tile(total_interval_intrahops, int((p - 2)/2))
        else:
            raise ValueError

        #Make inter generation connections
        #points_this_inter stands for: points_onthislayer_that_interconnect
        #points_next_inter stands for: points_onnextlayer_that_interconnect
        st = time.time()
        points_this_inter_mask = get_if_point_is_connected_with_upper_layer_q3_general_optimized(p, n)
        points_this_inter = sion[points_this_inter_mask]
        if (p - 2) % 2 == 0:
            whole_gen_rvals = np.append(total_rvals[:-1], np.tile(total_rvals[:-1], int((p - 2) / 2)))
            whole_gen_rsigns = np.append(total_rsigns[:-1], np.tile(total_rsigns[:-1], int((p - 2) / 2)))
        else:
            raise ValueError
        points_this_inter_rvals = whole_gen_rvals[points_this_inter_mask]
        points_this_inter_rsigns = whole_gen_rsigns[points_this_inter_mask]
        print(time.time() - st)
        #Not using get_points_that_connect_with_prev_layer_q3_general_optimized since its actually not optimized
        st = time.time()
        points_next_inter = get_points_that_connect_with_prev_layer_q3_general_optimized(p, n+1, first_sites_of_each_level[n+1])
        points_next_inter_rvals = points_this_inter_rvals + 1
        points_next_inter_rsigns = -1 * points_this_inter_rsigns
        print(time.time()-st)

        inter_hoppings = strain_conn(qax, points_this_inter_rvals, points_next_inter_rvals,
                                     points_this_inter_rsigns, points_next_inter_rsigns)

        ham[points_this_inter, points_next_inter] = inter_hoppings
        ham[points_next_inter, points_this_inter] = inter_hoppings

        ## Define variables to carry over needed information for next iteration
        if 2 * len(points_next_inter) / p == round(2 * len(points_next_inter) / p, 0):
            anchor_sites = points_next_inter[:int(2 * len(points_next_inter) / p + 1)]
        else:
            raise ValueError
        if 2 * len(points_next_inter_rvals) / p == round(2 * len(points_next_inter_rvals) / p, 0):
            anchor_rvals = points_next_inter_rvals[:int(2 * len(points_next_inter_rvals) / p + 1)]
        else:
            raise ValueError
        if 2 * len(points_next_inter_rsigns) / p == round(2 * len(points_next_inter_rsigns) / p, 0):
            anchor_signs = points_next_inter_rsigns[:int(2 * len(points_next_inter_rsigns) / p + 1)]
        else:
            raise ValueError

    # And finally for last layer
    ## Modified to take into account strain
    # sion stands for: site_indices_on_level
    sion = np.arange(first_sites_of_each_level[num_levels-1], first_sites_of_each_level[num_levels])

    # More optimized NN intra layer hopping
    ## Leverage p/2 fold rotational symmetry of strain texture for optimization
    if 2 * len(sion) / p == round(2 * len(sion) / p, 0):
        rel_sector_sites = sion[:int(2 * len(sion) / p)]
    else:
        raise ValueError

    ## Modified to include strain
    total_interval_intrahops = np.array([])
    total_rvals = np.array([])
    total_rsigns = np.array([])
    for ind in range(len(anchor_sites) - 1):
        num_interval_sites = anchor_sites[ind + 1] - anchor_sites[ind] + 1
        interval_sites = np.arange(anchor_sites[ind], anchor_sites[ind + 1] + 1)
        rel_interval_rvals = get_interval_rvals(anchor_rvals[ind], anchor_rvals[ind + 1], num_interval_sites)

        if ind == 0:  # if check to avoid overlap
            total_rvals = np.append(total_rvals, rel_interval_rvals)
        else:
            total_rvals = np.append(total_rvals, rel_interval_rvals[1:])

        if num_interval_sites % 2 == 0:
            rel_interval_rsigns = np.tile([anchor_signs[ind], -1 * anchor_signs[ind]],
                                          int(num_interval_sites / 2))
        else:
            rel_interval_rsigns = np.tile([anchor_signs[ind], -1 * anchor_signs[ind]],
                                          int((num_interval_sites - 1) / 2))
            rel_interval_rsigns = np.append(rel_interval_rsigns, anchor_signs[ind + 1])

        if ind == 0:  # if check to avoid overlap
            total_rsigns = np.append(total_rsigns, rel_interval_rsigns)
        else:
            total_rsigns = np.append(total_rsigns, rel_interval_rsigns[1:])

        interval_intrahops = strain_conn(qax, rel_interval_rvals[:-1], rel_interval_rvals[1:],
                                         rel_interval_rsigns[:-1], rel_interval_rsigns[1:])

        ham[interval_sites[:-1], interval_sites[1:]] = interval_intrahops
        ham[interval_sites[1:], interval_sites[:-1]] = interval_intrahops

        total_interval_intrahops = np.append(total_interval_intrahops, interval_intrahops)

    if (p - 2) % 2 == 0:
        leftover_sites = np.arange(rel_sector_sites[-1] + 1, first_sites_of_each_level[num_levels])
        leftover_sites = np.append(leftover_sites, sion[0])
        ham[leftover_sites[:-1], leftover_sites[1:]] = np.tile(total_interval_intrahops, int((p - 2) / 2))
        ham[leftover_sites[1:], leftover_sites[:-1]] = np.tile(total_interval_intrahops, int((p - 2) / 2))
    else:
        raise ValueError

    return ham


def sublattice_label_q3(pval, nval):
    points_per_level, total_num_sites = number_points_q3_general_from_repeating_pattern(pval, nval)
    first_site_on_gen = np.cumsum(np.append(np.array([0]), points_per_level))
    asites = np.array([])
    bsites = np.array([])

    if nval >= 1:
        asites = np.append(asites, np.arange(first_site_on_gen[0], first_site_on_gen[1], 2))
        bsites = np.append(bsites, np.arange(first_site_on_gen[0] + 1, first_site_on_gen[1], 2))
    if nval > 1:
        for nl in range(2, nval + 1):
            asites = np.append(asites, np.arange(first_site_on_gen[nl-1] + 1, first_site_on_gen[nl], 2))
            bsites = np.append(bsites, np.arange(first_site_on_gen[nl-1], first_site_on_gen[nl], 2))

    return asites.astype(int), bsites.astype(int)


def general_q3_hamiltonian_with_strain_nonHermitian(pval, nval, qax, alpha_NH):
    ham = general_q3_hamiltonian_with_strain(pval, nval, qax)
    asites, bsites = sublattice_label_q3(pval, nval)
    sublat_basis = np.concatenate((asites, bsites))

    ham = ham.tocsr()
    ham = ham[:, sublat_basis][sublat_basis, :]
    ham = ham.multiply(np.concatenate((np.repeat(1 + alpha_NH, int(ham.shape[0] / 2)),
                                       np.repeat(1 - alpha_NH, int(ham.shape[0] / 2)))).reshape(-1, 1))
    # ham[:int(ham.shape[0] / 2), int(ham.shape[0] / 2):] = (1 + alpha_NH) * ham[:int(ham.shape[0] / 2), int(ham.shape[0] / 2):]
    # ham[int(ham.shape[0] / 2):, :int(ham.shape[0] / 2)] = (1 - alpha_NH) * ham[int(ham.shape[0] / 2):, :int(ham.shape[0] / 2)]

    return ham




