from Lattice.General_Hamiltonian import *
from Lattice.General_Hamiltonian_Strain import sublattice_label_q3

# Same as Fundamental.Hamiltonian_PeierlsSubstitution Number_Plaquets but cleaned up
def number_plaquets_q3(p, num_levels):
    plaquets_per_level = np.array([])
    number_plaquets = 0

    plaquets_per_threeside = (p-4)
    plaquets_per_fourside = (p-5)

    if num_levels >= 1:
        number_plaquets = number_plaquets + 1
        plaquets_per_level = np.append(plaquets_per_level, 1)

    if num_levels >= 2:
        number_plaquets = number_plaquets + p
        plaquets_per_level = np.append(plaquets_per_level, p)

    if num_levels >= 3:
        num_prev_threesides = p
        num_prev_foursides = 0
        for n in range(3, num_levels + 1):
            addterm = (num_prev_threesides * plaquets_per_threeside) + (num_prev_foursides * plaquets_per_fourside)
            number_plaquets = number_plaquets + addterm
            plaquets_per_level = np.append(plaquets_per_level, addterm)
            num_prev_threesides = (p-5)*num_prev_threesides + (p-6)*num_prev_foursides
            num_prev_foursides = plaquets_per_level[n-2]

    return plaquets_per_level, number_plaquets

# This function is a copy of general_q3_hamiltonian_superoptimized from General_Hamiltonian.py
# but with alterations to account for Peierls phase
def general_q3_hamiltonian_with_real_B_field(p, num_levels, beta): ## extra argument for beta
    if num_levels == 2: #Warning because for some reason this code (but not the below superoptimized one has problems when total num generations = 2)
        raise ValueError('This code is unstable for two total number of generations.'
                         'Larger nnumber of total generations is stable however')
    sites_per_level, tot_num_sites = number_points_q3_general_from_repeating_pattern(p, num_levels)
    first_sites_of_each_level = np.array([0])
    for n in range(num_levels):
        first_sites_of_each_level = np.append(first_sites_of_each_level,
                                              sites_per_level[n] + np.sum(sites_per_level[:n]))

    ham = dok_matrix((tot_num_sites, tot_num_sites), dtype=np.complex128)
    t = 1

    if num_levels >= 0:  # Have to hard code first gen
        sion = np.arange(first_sites_of_each_level[0], first_sites_of_each_level[1])

        # More optimized NN intra layer hopping
        sion_one_above = sion + 1
        sion_one_above[-1] = sion[0]

        ## Hard code new zeroth gen intralayer hoppings with Peierls phase
        ham[sion, sion_one_above] = t * np.exp(-1j * beta / p)
        ham[sion_one_above, sion] = t * np.exp(1j * beta / p)

        # Hard code inter generation connection
        first_gen_inter_connect_hardcode = np.arange(first_sites_of_each_level[1], first_sites_of_each_level[2],
                                                     (p - 3))
        ham[sion, first_gen_inter_connect_hardcode] = t
        ham[first_gen_inter_connect_hardcode, sion] = t

        ## Extra dataholder to keep track interlayer sites and previous Peierls phases
        current_layer_interlayer_sites = first_gen_inter_connect_hardcode
        prev_layer_ps_phases = np.repeat(beta / p, len(sion))

    for n in range(1, num_levels - 1):  # iterate over all levels except first and last level
        # print('Working on generation: {}'.format(n + 1))
        # sion stands for: site_indices_on_level
        sion = np.arange(first_sites_of_each_level[n], first_sites_of_each_level[n + 1])

        # More optimized NN intra layer hopping
        sion_one_above = sion + 1
        sion_one_above[-1] = sion[0]

        ## Code to account for Peierls phases on previous layer and adapt current intralayer hoppings accordingly
        current_layer_interval_sites = [np.arange(current_layer_interlayer_sites[ind],
                                                  current_layer_interlayer_sites[ind + 1] + 1)
                                        for ind in range(len(current_layer_interlayer_sites) - 1)]
        current_layer_ps_phases = np.array([])
        for ind, interval_sites in enumerate(current_layer_interval_sites):
            sites_1 = interval_sites[:-1]
            sites_2 = interval_sites[1:]
            rel_ps_phases = (beta + prev_layer_ps_phases[ind]) / (len(interval_sites) - 1)
            current_layer_ps_phases = np.append(current_layer_ps_phases,
                                                np.repeat(rel_ps_phases, len(interval_sites) - 1))

            ham[sites_1, sites_2] = t * np.exp(-1j * rel_ps_phases)
            ham[sites_2, sites_1] = t * np.exp(1j * rel_ps_phases)

        ## Extra last bit to code hopping between last few and first sites on layer
        rel_sites = np.append(np.arange(current_layer_interlayer_sites[-1], first_sites_of_each_level[n + 1]),
                              first_sites_of_each_level[n])
        if n != 1:
            rel_ps_phases = (beta + prev_layer_ps_phases[-1]) / (p - 4)
            current_layer_ps_phases = np.append(current_layer_ps_phases, np.repeat(rel_ps_phases, (p - 4)))
        else:
            rel_ps_phases = (beta + prev_layer_ps_phases[-1]) / (p - 3)
            current_layer_ps_phases = np.append(current_layer_ps_phases, np.repeat(rel_ps_phases, (p - 3)))
        ham[rel_sites[:-1], rel_sites[1:]] = t * np.exp(-1j * rel_ps_phases)
        ham[rel_sites[1:], rel_sites[:-1]] = t * np.exp(1j * rel_ps_phases)

        # Make inter generation connections
        # points_this_inter stands for: points_onthislayer_that_interconnect
        # points_next_inter stands for: points_onnextlayer_that_interconnect
        # st = time.time()
        points_this_inter = sion[get_if_point_is_connected_with_upper_layer_q3_general_optimized(p, n)]
        # print(time.time() - st)
        # Not using get_points_that_connect_with_prev_layer_q3_general_optimized since its actually not optimized
        # st = time.time()
        points_next_inter = get_points_that_connect_with_prev_layer_q3_general_optimized(p, n + 1,
                                                                                         first_sites_of_each_level[
                                                                                             n + 1])
        # print(time.time() - st)
        ham[points_this_inter, points_next_inter] = t
        ham[points_next_inter, points_this_inter] = t

        ## Update relevant interlayer hopping sites and Peierls phases for next iteration
        current_layer_interlayer_sites = points_next_inter
        prev_layer_ps_phases = np.array([])
        if n >= 2:
            ind_tracker = 1 #need this to deal with one later type of bond dependence from layer to layer in n>=2 layers
        else:
            ind_tracker = 0
        for ind in range(len(points_this_inter) - 1):
            num_rel_bonds = points_this_inter[ind + 1] - points_this_inter[ind]
            prev_layer_ps_phases = np.append(prev_layer_ps_phases,
                                             np.sum(current_layer_ps_phases[ind_tracker:ind_tracker+num_rel_bonds]))
            # if num_rel_bonds == 2 and n==2:
            #     print(np.sum(current_layer_ps_phases[ind_tracker:ind_tracker+num_rel_bonds]))

            ind_tracker = ind_tracker + num_rel_bonds

            # if n == 2:
            #     print(current_layer_ps_phases[ind_tracker:ind_tracker+num_rel_bonds], ind_tracker)
                # print(points_this_inter[ind], points_this_inter[ind + 1], np.sum(current_layer_ps_phases[ind_tracker:ind_tracker+num_rel_bonds]))

        ## Extra bit of code to add peierls phase associated with last few bond on layer between last and first sites
        prev_layer_ps_phases = np.append(prev_layer_ps_phases, current_layer_ps_phases[0] + current_layer_ps_phases[-1])
        # if n==2:
        #     print(prev_layer_ps_phases)


    ## Modified code for last layer intralayer hopping with Peierls phases included

    current_layer_interval_sites = [np.arange(current_layer_interlayer_sites[ind],
                                              current_layer_interlayer_sites[ind + 1] + 1)
                                    for ind in range(len(current_layer_interlayer_sites) - 1)]
    current_layer_ps_phases = np.array([])
    for ind, interval_sites in enumerate(current_layer_interval_sites):
        sites_1 = interval_sites[:-1]
        sites_2 = interval_sites[1:]
        rel_ps_phases = (beta + prev_layer_ps_phases[ind]) / (len(interval_sites) - 1)
        current_layer_ps_phases = np.append(current_layer_ps_phases,
                                            np.repeat(rel_ps_phases, len(interval_sites) - 1))

        ham[sites_1, sites_2] = t * np.exp(-1j * rel_ps_phases)
        ham[sites_2, sites_1] = t * np.exp(1j * rel_ps_phases)

    ## Extra last bit to code hopping between last few and first sites on layer
    rel_sites = np.append(np.arange(current_layer_interlayer_sites[-1], first_sites_of_each_level[num_levels]),
                          first_sites_of_each_level[num_levels - 1])
    if num_levels != 1:
        rel_ps_phases = (beta + prev_layer_ps_phases[-1]) / (p - 4)
        current_layer_ps_phases = np.append(current_layer_ps_phases, np.repeat(rel_ps_phases, (p - 4)))
    else:
        rel_ps_phases = (beta + prev_layer_ps_phases[-1]) / (p - 3)
        current_layer_ps_phases = np.append(current_layer_ps_phases, np.repeat(rel_ps_phases, (p - 3)))
    ham[rel_sites[:-1], rel_sites[1:]] = t * np.exp(-1j * rel_ps_phases)
    ham[rel_sites[1:], rel_sites[:-1]] = t * np.exp(1j * rel_ps_phases)

    return (ham)

# This is just like the above function but more optimized
def general_q3_hamiltonian_with_real_B_field_superoptimized(p, num_levels, beta): ## extra argument for beta
    sites_per_level, tot_num_sites = number_points_q3_general_from_repeating_pattern(p, num_levels)
    first_sites_of_each_level = np.array([0])
    for n in range(num_levels):
        first_sites_of_each_level = np.append(first_sites_of_each_level,
                                              sites_per_level[n] + np.sum(sites_per_level[:n]))

    ham = dok_matrix((tot_num_sites, tot_num_sites), dtype=np.complex128)
    t = 1

    if num_levels >= 0:  # Have to hard code first gen
        sion = np.arange(first_sites_of_each_level[0], first_sites_of_each_level[1])

        # More optimized NN intra layer hopping
        sion_one_above = sion + 1
        sion_one_above[-1] = sion[0]

        ## Hard code new zeroth gen intralayer hoppings with Peierls phase
        ham[sion, sion_one_above] = t * np.exp(-1j * beta / p)
        ham[sion_one_above, sion] = t * np.exp(1j * beta / p)

        # Hard code inter generation connection
        first_gen_inter_connect_hardcode = np.arange(first_sites_of_each_level[1], first_sites_of_each_level[2],
                                                     (p - 3))
        ham[sion, first_gen_inter_connect_hardcode] = t
        ham[first_gen_inter_connect_hardcode, sion] = t

        ## Extra dataholder to keep track interlayer sites and previous Peierls phases
        current_layer_interlayer_sites = first_gen_inter_connect_hardcode
        prev_layer_ps_phases = np.repeat(beta / p, len(sion))

    for n in range(1, num_levels - 1):  # iterate over all levels except first and last level

        # sion stands for: site_indices_on_level
        sion = np.arange(first_sites_of_each_level[n], first_sites_of_each_level[n + 1])

        # More optimized NN intra layer hopping
        sion_one_above = sion + 1
        sion_one_above[-1] = sion[0]

        ## Code to account for Peierls phases on previous layer and adapt current intralayer hoppings accordingly
        if len(current_layer_interlayer_sites) % p == 0:
            current_layer_interval_sites = [np.arange(current_layer_interlayer_sites[ind],
                                                      current_layer_interlayer_sites[ind + 1] + 1)
                                            for ind in range(int(len(current_layer_interlayer_sites) / p))]
        else:
            raise ValueError
        current_layer_ps_phases = np.array([])
        for ind, interval_sites in enumerate(current_layer_interval_sites):
            sites_1 = interval_sites[:-1]
            sites_2 = interval_sites[1:]
            rel_ps_phases = (beta + prev_layer_ps_phases[ind]) / (len(interval_sites) - 1)
            current_layer_ps_phases = np.append(current_layer_ps_phases,
                                                np.repeat(rel_ps_phases, len(interval_sites) - 1))

            ham[sites_1, sites_2] = t * np.exp(-1j * rel_ps_phases)
            ham[sites_2, sites_1] = t * np.exp(1j * rel_ps_phases)

        remaining_sites = np.append(np.arange(current_layer_interval_sites[-1][-1], first_sites_of_each_level[n + 1]),
                                    first_sites_of_each_level[n])
        ham[remaining_sites[:-1], remaining_sites[1:]] = t * np.exp(-1j * np.tile(current_layer_ps_phases, p - 1))
        ham[remaining_sites[1:], remaining_sites[:-1]] = t * np.exp(1j * np.tile(current_layer_ps_phases, p - 1))

        # Make inter generation connections
        # points_this_inter stands for: points_onthislayer_that_interconnect
        # points_next_inter stands for: points_onnextlayer_that_interconnect
        # st = time.time()
        points_this_inter = sion[get_if_point_is_connected_with_upper_layer_q3_general_optimized(p, n)]
        # print(time.time() - st)
        # Not using get_points_that_connect_with_prev_layer_q3_general_optimized since its actually not optimized
        # st = time.time()
        points_next_inter = get_points_that_connect_with_prev_layer_q3_general_optimized(p, n + 1,
                                                                                         first_sites_of_each_level[n + 1])
        # print(time.time() - st)
        ham[points_this_inter, points_next_inter] = t
        ham[points_next_inter, points_this_inter] = t

        ## Update relevant interlayer hopping sites and Peierls phases for next iteration
        current_layer_interlayer_sites = points_next_inter
        prev_layer_ps_phases = np.array([])
        if n >= 2:
            ind_tracker = 1 #need this to deal with one later type of bond dependence from layer to layer in n>=2 layers
        else:
            ind_tracker = 0

        if len(points_this_inter) % p == 0:
            for ind in range(int(len(points_this_inter) / p)):
                num_rel_bonds = points_this_inter[ind + 1] - points_this_inter[ind]
                prev_layer_ps_phases = np.append(prev_layer_ps_phases,
                                                 np.sum(current_layer_ps_phases[ind_tracker:ind_tracker+num_rel_bonds]))

                ind_tracker = ind_tracker + num_rel_bonds

        else:
            raise ValueError

        # Need to manually fix last entry since it requires wrapping which is not captured in optimized current_layer_ps_phases
        if n >= 2:
            prev_layer_ps_phases[-1] = np.sum(current_layer_ps_phases[[0, -1]])


    ## Modified code for last layer intralayer hopping with Peierls phases included

    if len(current_layer_interlayer_sites) % p == 0:
        current_layer_interval_sites = [np.arange(current_layer_interlayer_sites[ind],
                                                  current_layer_interlayer_sites[ind + 1] + 1)
                                        for ind in range(int(len(current_layer_interlayer_sites) / p))]
    else:
        raise ValueError

    current_layer_ps_phases = np.array([])
    for ind, interval_sites in enumerate(current_layer_interval_sites):
        sites_1 = interval_sites[:-1]
        sites_2 = interval_sites[1:]
        rel_ps_phases = (beta + prev_layer_ps_phases[ind]) / (len(interval_sites) - 1)
        current_layer_ps_phases = np.append(current_layer_ps_phases,
                                            np.repeat(rel_ps_phases, len(interval_sites) - 1))

        ham[sites_1, sites_2] = t * np.exp(-1j * rel_ps_phases)
        ham[sites_2, sites_1] = t * np.exp(1j * rel_ps_phases)

    remaining_sites = np.append(np.arange(current_layer_interval_sites[-1][-1], tot_num_sites),
                                first_sites_of_each_level[-2])
    ham[remaining_sites[:-1], remaining_sites[1:]] = t * np.exp(-1j * np.tile(current_layer_ps_phases, p - 1))
    ham[remaining_sites[1:], remaining_sites[:-1]] = t * np.exp(1j * np.tile(current_layer_ps_phases, p - 1))

    return ham


def general_q3_hamiltonian_with_real_B_field_NH(p, num_levels, beta, alpha_NH):
    ham = general_q3_hamiltonian_with_real_B_field_superoptimized(p, num_levels, beta)
    asites, bsites = sublattice_label_q3(p, num_levels)
    sublat_basis = np.concatenate((asites, bsites))

    ham = ham.tocsr()
    ham = ham[:, sublat_basis][sublat_basis, :]
    ham = ham.multiply(np.concatenate((np.repeat(1 + alpha_NH, int(ham.shape[0] / 2)),
                                       np.repeat(1 - alpha_NH, int(ham.shape[0] / 2)))).reshape(-1, 1))

    return ham
