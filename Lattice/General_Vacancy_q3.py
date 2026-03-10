import numpy as np
from Lattice.General_Hamiltonian_Strain import sublattice_label_q3
from Lattice.General_Hamiltonian import general_q3_hamiltonian_superoptimized
from Lattice.General_Hamiltonian import number_points_q3_general_from_repeating_pattern
from Operations.Neighbors import identify_nearest_neighbors_hyperbolic_q3
from Operations.Neighbors import identify_next_nearest_neighbors_hyperbolic_q3
from Operations.Neighbors import get_neighbors_of_site


def check_for_isolated_sites(vacancy_ham, num_vacancies):
    num_nonisolated_points = len(np.unique(vacancy_ham.indices))
    num_total_original_sites = vacancy_ham.shape[0]
    if num_nonisolated_points == (num_total_original_sites - num_vacancies):
        return False
    else:
        return True


def hyperbolic_q3_equal_sublattice_vacancy_density(pval, nval, vacancy_density,
                                                   seed=None, isocheck=False):
    if (pval % 2) != 0:
        raise ValueError
    sparse_ham = general_q3_hamiltonian_superoptimized(pval, nval)

    total_num_sites = number_points_q3_general_from_repeating_pattern(pval, nval)[1]
    num_sublat_sites = int(total_num_sites / 2)
    num_vacancies_per_sublat = int(round(vacancy_density * num_sublat_sites, 0))

    asites, bsites = sublattice_label_q3(pval, nval)
    if seed is None:
        asite_vacancies = np.random.choice(asites, size=num_vacancies_per_sublat, replace=False)
        bsite_vacancies = np.random.choice(bsites, size=num_vacancies_per_sublat, replace=False)
    else:
        rng = np.random.RandomState(seed=seed)
        asite_vacancies = rng.choice(asites, size=num_vacancies_per_sublat, replace=False)
        bsite_vacancies = rng.choice(bsites, size=num_vacancies_per_sublat, replace=False)

    all_site_vacancies = np.concatenate((asite_vacancies, bsite_vacancies))

    sparse_ham = sparse_ham.tocsc()
    sparse_ham.data[np.isin(sparse_ham.indices, all_site_vacancies)] = 0
    sparse_ham.eliminate_zeros()

    sparse_ham = sparse_ham.tocsr()
    sparse_ham.data[np.isin(sparse_ham.indices, all_site_vacancies)] = 0
    sparse_ham.eliminate_zeros()

    if isocheck:
        check_res = check_for_isolated_sites(sparse_ham, len(all_site_vacancies))
        print('Are there any sites that have had all neighbors removed: {}'.format(check_res))

    return sparse_ham


def hyperbolic_q3_equal_sublattice_vacancy_density_bulk_only(pval, nval, bulk_vacancy_density,
                                                             seed=None, isocheck=False):
    if (pval % 2) != 0:
        raise ValueError
    sparse_ham = general_q3_hamiltonian_superoptimized(pval, nval)

    points_per_level = number_points_q3_general_from_repeating_pattern(pval, nval)[0]
    num_sublat_bulk_sites = int(np.sum(points_per_level[:-1]) / 2)
    num_bulk_vacancies_per_sublat = int(round(bulk_vacancy_density * num_sublat_bulk_sites, 0))

    asites, bsites = sublattice_label_q3(pval, nval)
    bulk_asites = asites[asites < np.sum(points_per_level[:-1])]
    bulk_bsites = bsites[bsites < np.sum(points_per_level[:-1])]

    if seed is None:
        bulk_asite_vacancies = np.random.choice(bulk_asites, size=num_bulk_vacancies_per_sublat, replace=False)
        bulk_bsite_vacancies = np.random.choice(bulk_bsites, size=num_bulk_vacancies_per_sublat, replace=False)
    else:
        rng = np.random.RandomState(seed=seed)
        bulk_asite_vacancies = rng.choice(bulk_asites, size=num_bulk_vacancies_per_sublat, replace=False)
        bulk_bsite_vacancies = rng.choice(bulk_bsites, size=num_bulk_vacancies_per_sublat, replace=False)

    all_site_vacancies = np.concatenate((bulk_asite_vacancies, bulk_bsite_vacancies))

    sparse_ham = sparse_ham.tocsc()
    sparse_ham.data[np.isin(sparse_ham.indices, all_site_vacancies)] = 0
    sparse_ham.eliminate_zeros()

    sparse_ham = sparse_ham.tocsr()
    sparse_ham.data[np.isin(sparse_ham.indices, all_site_vacancies)] = 0
    sparse_ham.eliminate_zeros()

    if isocheck:
        check_res = check_for_isolated_sites(sparse_ham, len(all_site_vacancies))
        print('Are there any sites that have had all neighbors removed: {}'.format(check_res))

    return sparse_ham


def hyperbolic_q3_equal_sublattice_vacancy_density_distance_restriction(pval, nval, vacancy_density,
                                                                        seed, isocheck=False):
    if (pval % 2) != 0:
        raise ValueError
    sparse_ham = general_q3_hamiltonian_superoptimized(pval, nval)

    total_num_sites = number_points_q3_general_from_repeating_pattern(pval, nval)[1]
    num_sublat_sites = int(total_num_sites / 2)
    num_vacancies_per_sublat = int(round(vacancy_density * num_sublat_sites, 0))

    asites, bsites = sublattice_label_q3(pval, nval)

    all_site_vacancies = np.array([])
    rng = np.random.RandomState(seed=seed)
    random_seeds = rng.randint(low=1, high=10000000, size=100000)
    while_loop_counter = 0
    nn_rows, nn_cols = identify_nearest_neighbors_hyperbolic_q3(sparse_ham)
    nnn_rows, nnn_cols = identify_next_nearest_neighbors_hyperbolic_q3(sparse_ham)
    exclusion_mask_asites = np.repeat(True, len(asites))
    exclusion_mask_bsites = np.repeat(True, len(bsites))
    new_excluded_asites = np.zeros(10, dtype=np.int64)
    new_excluded_bsites = np.zeros(10, dtype=np.int64)
    while len(all_site_vacancies) < (2 * num_vacancies_per_sublat):

        curr_rng = np.random.RandomState(seed=random_seeds[while_loop_counter])
        trial_vac_asite = curr_rng.choice(asites[exclusion_mask_asites], size=1)

        new_excluded_asites[:] = -1
        new_excluded_bsites[:] = -1
        nn_neighbors = get_neighbors_of_site(trial_vac_asite, nn_rows, nn_cols)
        nnn_neighbors = get_neighbors_of_site(trial_vac_asite, nnn_rows, nnn_cols)
        new_excluded_asites[0] = trial_vac_asite[0]
        new_excluded_bsites[:len(nn_neighbors)] = nn_neighbors
        new_excluded_asites[1:len(nnn_neighbors) + 1] = nnn_neighbors

        exclusion_mask_bsites[np.isin(bsites, new_excluded_bsites)] = False

        while_loop_counter += 1

        curr_rng = np.random.RandomState(seed=random_seeds[while_loop_counter])
        trial_vac_bsite = curr_rng.choice(bsites[exclusion_mask_bsites], size=1)

        more_nn_neighbors = get_neighbors_of_site(trial_vac_bsite, nn_rows, nn_cols)
        more_nnn_neighbors = get_neighbors_of_site(trial_vac_bsite, nnn_rows, nnn_cols)
        new_excluded_bsites[len(nn_neighbors)] = trial_vac_bsite[0]
        new_excluded_asites[len(nnn_neighbors) + 1:len(nnn_neighbors) + 1 + len(more_nn_neighbors)] = more_nn_neighbors
        new_excluded_bsites[len(nn_neighbors) + 1:len(nn_neighbors) + 1 + len(more_nnn_neighbors)] = more_nnn_neighbors

        exclusion_mask_asites[np.isin(asites, new_excluded_asites)] = False
        exclusion_mask_bsites[np.isin(bsites, new_excluded_bsites)] = False

        while_loop_counter += 1

        all_site_vacancies = np.append(all_site_vacancies, trial_vac_asite)
        all_site_vacancies = np.append(all_site_vacancies, trial_vac_bsite)

    sparse_ham = sparse_ham.tocsc()
    sparse_ham.data[np.isin(sparse_ham.indices, all_site_vacancies)] = 0
    sparse_ham.eliminate_zeros()

    sparse_ham = sparse_ham.tocsr()
    sparse_ham.data[np.isin(sparse_ham.indices, all_site_vacancies)] = 0
    sparse_ham.eliminate_zeros()

    if isocheck:
        check_res = check_for_isolated_sites(sparse_ham, len(all_site_vacancies))
        print('Are there any sites that have had all neighbors removed: {}'.format(check_res))

    return sparse_ham
