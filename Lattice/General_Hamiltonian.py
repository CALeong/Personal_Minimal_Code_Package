import numpy as np
from Fundamental.Number_Points import points
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix
from Fundamental.Hamiltonian_PeierlsSubstitution import Number_Plaquets
import time

def number_points_q3_general_from_number_plaquets(p, num_levels):
    q = 3

    if num_levels == 1:
        sites_per_level = np.array([p])
    elif num_levels == 2:
        sites_per_level = np.array([p, p * (p - 3)])
    elif num_levels >= 3:
        sites_per_level = np.array([p, p*(p-3), (p-4)*p + p*(p-5)*(p-3)])
    number_plaqs_per_level = Number_Plaquets(p, q, num_levels)[0]
    for n in range(3, num_levels):
        sites_per_level = np.append(sites_per_level, (p-5)*(number_plaqs_per_level[n-1] - number_plaqs_per_level[n-2])*(p-3)
                                    + (p-6)*number_plaqs_per_level[n-2]*(p-3)
                                    + number_plaqs_per_level[n-1]*(p-4))

    sites_per_level = [int(i) for i in sites_per_level]
    return(sites_per_level, np.sum(sites_per_level))

def number_points_q3_general_from_repeating_pattern(p, num_levels):
    q= 3
    sites_per_level = np.array([])
    for n in range(num_levels):
        if n == 0:
            sites_per_level = np.append(sites_per_level, p)
        elif n == 1:
            sites_per_level = np.append(sites_per_level, p*(p-3))
            number_prev_3sides = p
            number_prev_4sides = 0
        else:
            number_3sides = (p-5)*number_prev_3sides + (p-6)*number_prev_4sides
            number_4sides = number_prev_3sides + number_prev_4sides
            sites_per_level = np.append(sites_per_level, number_3sides*(p-3) + number_4sides*(p-4))
            number_prev_3sides = number_3sides
            number_prev_4sides = number_4sides
    sites_per_level = np.array([int(i) for i in sites_per_level])
    return(sites_per_level, np.sum(sites_per_level))

#Define: sector as the 1/p [art of a level which repeats p times to give you that level
def get_types_of_polygons_in_sector_q3_general(p, specific_level):
    q = 3
    for n in range(specific_level):
        if n == 0:
            number_3sides = 1
            number_4sides = 0
        elif n == 1:
            number_3sides = p
            number_4sides = 0
            number_prev_3sides = p
            number_prev_4sides = 0
        else:
            number_3sides = (p - 5) * number_prev_3sides + (p - 6) * number_prev_4sides
            number_4sides = number_prev_3sides + number_prev_4sides
            number_prev_3sides = number_3sides
            number_prev_4sides = number_4sides
    return(int(number_3sides/p), int(number_4sides/p))

#Define: motif is the pattern of how a layer is bonded
#Define: Symbolic means give sequence in terms of 3sides and 4sides
def get_motif_of_sector_symbolic_q3_general(p, specific_level):
    q = 3
    ontop_3side_motif = np.concatenate((np.repeat('3side', (p-5)), np.array(['4side'])))
    ontop_4side_motif = np.concatenate((np.repeat('3side', (p-6)), np.array(['4side'])))
    for n in range(1, specific_level+1):
        # Code will not work for num_levels=1 since this is just trivial case anyway
        if n == 1:
            new_motif = np.array(['3side'])
            prev_motif = np.array(['3side'])
        else:
            new_motif = np.array([])
            for pm in prev_motif:
                if pm == '3side':
                    new_motif = np.append(new_motif, ontop_3side_motif)
                elif pm == '4side':
                    new_motif = np.append(new_motif, ontop_4side_motif)
            prev_motif = np.copy(new_motif) #Originally had no copy and it works but just to be careful add copy here
    return(new_motif)

def get_points_that_connect_with_prev_layer_q3_general(p, specific_level, first_site_in_layer_index):
    q = 3
    motif_string = np.tile(get_motif_of_sector_symbolic_q3_general(p, specific_level), p)
    prev_layer_connected = np.array([0])
    for ms in motif_string:
        if ms == '3side':
            prev_layer_connected = np.append(prev_layer_connected, prev_layer_connected[-1] + (p-3))
        elif ms == '4side':
            prev_layer_connected = np.append(prev_layer_connected, prev_layer_connected[-1] + (p-4))
    prev_layer_connected = prev_layer_connected[:-1] #To avoid double counting first point
    return(prev_layer_connected + first_site_in_layer_index)

def get_if_point_is_connected_with_upper_layer_q3_general(p, specific_level):
    q = 3
    #Since get_motif_of_sector_symbolic_q3_general labels generations starting with 0
    specific_level = specific_level - 1
    entire_motif = np.tile(get_motif_of_sector_symbolic_q3_general(p, specific_level+2), p)
    isconnected = np.array([False], dtype=bool)
    for em in entire_motif:
        if em == '3side':
            isconnected = np.append(isconnected, True)
        elif em == '4side': #Two appending due to nature of plaquets and corresponding points
            isconnected = np.append(isconnected, True)
            isconnected = np.append(isconnected, False)
    return(isconnected[:-1]) #To avoid over count of last value

def general_q3_hamiltonian(p, num_levels):
    q = 3
    sites_per_level, tot_num_sites = number_points_q3_general_from_repeating_pattern(p, num_levels)
    first_sites_of_each_level = np.array([0])
    for n in range(num_levels):
        first_sites_of_each_level = np.append(first_sites_of_each_level, sites_per_level[n] + np.sum(sites_per_level[:n]))

    ham = dok_matrix((tot_num_sites, tot_num_sites), dtype=int)
    t = 1

    if num_levels >= 0: #Have to hard code first gen
        sion = np.arange(first_sites_of_each_level[0], first_sites_of_each_level[1])

        for i in range(len(sion)):
            # Handles nearest-neighbor on same generation hopping
            ham[sion[i], np.take(sion, i + 1, mode='wrap')] = t
            ham[np.take(sion, i + 1, mode='wrap'), sion[i]] = t
            ham[sion[i], sion[i - 1]] = t
            ham[sion[i - 1], sion[i]] = t

        #Hard code inter generation connection
        first_gen_inter_connect_hardcode = np.arange(first_sites_of_each_level[1], first_sites_of_each_level[2], (p-3))
        ham[sion, first_gen_inter_connect_hardcode] = t
        ham[first_gen_inter_connect_hardcode, sion] = t

    for n in range(1, num_levels-1): #iterate over all levels except first and last level
        print('Working on generation: {}'.format(n+1))
        #sion stands for: site_indices_on_level
        sion = np.arange(first_sites_of_each_level[n], first_sites_of_each_level[n+1])

        for i in range(len(sion)):

            # Handles nearest-neighbor on same generation hopping
            ham[sion[i], np.take(sion, i + 1, mode='wrap')] = t
            ham[np.take(sion, i + 1, mode='wrap'), sion[i]] = t
            ham[sion[i], sion[i - 1]] = t
            ham[sion[i - 1], sion[i]] = t

        #Make inter generation connections
        #points_this_inter stands for: points_onthislayer_that_interconnect
        #points_next_inter stands for: points_onnextlayer_that_interconnect
        st = time.time()
        points_this_inter = sion[get_if_point_is_connected_with_upper_layer_q3_general(p, n)]
        points_next_inter = get_points_that_connect_with_prev_layer_q3_general(p, n+1, first_sites_of_each_level[n+1])
        print(time.time()-st)
        ham[points_this_inter, points_next_inter] = t
        ham[points_next_inter, points_this_inter] = t

    #And finally for last layer
    sion = np.arange(first_sites_of_each_level[num_levels-1], first_sites_of_each_level[num_levels])

    for i in range(len(sion)):
        # Handles nearest-neighbor on same generation hopping
        ham[sion[i], np.take(sion, i + 1, mode='wrap')] = t
        ham[np.take(sion, i + 1, mode='wrap'), sion[i]] = t
        ham[sion[i], sion[i - 1]] = t
        ham[sion[i - 1], sion[i]] = t

    return(ham)

###########################################################################################################
def replace_3side_4side(input, ontop_3side_motif, ontop_4side_motif):
    output = np.empty((len(input), len(ontop_3side_motif)), dtype=object)
    index_3side = np.where(input=='3side')[0]
    index_4side = np.where(input=='4side')[0]
    output[index_3side] = ontop_3side_motif
    output[index_4side] = np.concatenate((ontop_4side_motif,np.array(['delete_me'])))
    return(output)

# def get_motif_of_sector_symbolic_q3_general_optimized(p, specific_level):
#     q = 3
#     ontop_3side_motif = np.concatenate((np.repeat('3side', (p-5)), np.array(['4side'])))
#     ontop_4side_motif = np.concatenate((np.repeat('3side', (p-6)), np.array(['4side'])))
#     for n in range(1, specific_level+1):
#         # Code will not work for num_levels=1 since this is just trivial case anyway
#         if n == 1:
#             new_motif = np.array(['3side'])
#             prev_motif = np.array(['3side'])
#         else:
#             new_motif_mat = replace_3side_4side(prev_motif, ontop_3side_motif, ontop_4side_motif)
#             new_motif = new_motif_mat.reshape(-1)
#             new_motif = np.delete(new_motif, np.where(new_motif == 'delete_me')[0])
#             prev_motif = new_motif
#     return(new_motif)

def increment_3side_4side(p, input):
    index_3side = np.where(input == '3side')[0]
    index_4side = np.where(input == '4side')[0]
    increment_array = np.zeros(len(input))
    increment_array[index_3side] = (p-3)
    increment_array[index_4side] = (p-4)
    return(increment_array)

# def get_points_that_connect_with_prev_layer_q3_general_optimized(p, specific_level, first_site_in_layer_index):
#     q = 3
#     motif_string = np.tile(get_motif_of_sector_symbolic_q3_general(p, specific_level), p)
#     prev_layer_connected = np.array([0])
#     #Increment array -> matrix -> triangular matrix -> sum each row is trick to get around original for loop
#     #Need to go through the sparse matrix multiplication due to memory allocation problems
#     increment_array = increment_3side_4side(p, motif_string)
#     lower_triangular_sparse = lil_matrix((len(increment_array), len(increment_array)))
#     lower_triangular_sparse[np.tril_indices(len(increment_array))] = 1
#     increment_mat_triangle = lower_triangular_sparse.multiply(increment_array)
#     prev_layer_connected = np.concatenate((prev_layer_connected, np.ravel(np.sum(increment_mat_triangle, axis=1))))
#     prev_layer_connected = prev_layer_connected[:-1] #To avoid double counting first point
#     return(prev_layer_connected + first_site_in_layer_index)

def get_isconnected_status(input):
    index_3side = np.where(input == '3side')[0]
    index_4side = np.where(input == '4side')[0]
    output = np.empty((len(input), 2), dtype=object)
    output[index_3side] = np.array([True, 'delete_me'])
    output[index_4side] = np.array([True, False])
    return(output)

# def get_if_point_is_connected_with_upper_layer_q3_general_optimized(p, specific_level):
#     q = 3
#     #Since get_motif_of_sector_symbolic_q3_general labels generations starting with 0
#     specific_level = specific_level - 1
#     entire_motif = np.tile(get_motif_of_sector_symbolic_q3_general(p, specific_level+2), p)
#     isconnected = np.array([False], dtype=bool)
#     connection_array = get_isconnected_status(entire_motif).reshape(-1)
#     connection_array = np.delete(connection_array, np.where(connection_array == 'delete_me')[0])
#     isconnected = np.append(isconnected, connection_array.astype(bool))
#     return(isconnected[:-1]) #To avoid over count of last value

def general_q3_hamiltonian_optimized(p, num_levels):
    q = 3
    sites_per_level, tot_num_sites = number_points_q3_general_from_repeating_pattern(p, num_levels)
    first_sites_of_each_level = np.array([0])
    for n in range(num_levels):
        first_sites_of_each_level = np.append(first_sites_of_each_level, sites_per_level[n] + np.sum(sites_per_level[:n]))

    ham = dok_matrix((tot_num_sites, tot_num_sites), dtype=int)
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
        ham[sion, first_gen_inter_connect_hardcode] = t
        ham[first_gen_inter_connect_hardcode, sion] = t

    for n in range(1, num_levels-1): #iterate over all levels except first and last level
        print('Working on generation: {}'.format(n+1))
        #sion stands for: site_indices_on_level
        sion = np.arange(first_sites_of_each_level[n], first_sites_of_each_level[n+1])

        # More optimized NN intra layer hopping
        sion_one_above = sion + 1
        sion_one_above[-1] = sion[0]
        ham[sion, sion_one_above] = t
        ham[sion_one_above, sion] = t

        #Make inter generation connections
        #points_this_inter stands for: points_onthislayer_that_interconnect
        #points_next_inter stands for: points_onnextlayer_that_interconnect
        st = time.time()
        points_this_inter = sion[get_if_point_is_connected_with_upper_layer_q3_general(p, n)]
        print(time.time() - st)
        #Not using get_points_that_connect_with_prev_layer_q3_general_optimized since its actually not optimized
        st = time.time()
        points_next_inter = get_points_that_connect_with_prev_layer_q3_general(p, n+1, first_sites_of_each_level[n+1])
        print(time.time()-st)
        ham[points_this_inter, points_next_inter] = t
        ham[points_next_inter, points_this_inter] = t

    #And finally for last layer
    sion = np.arange(first_sites_of_each_level[num_levels-1], first_sites_of_each_level[num_levels])

    # More optimized NN intra layer hopping
    sion_one_above = sion + 1
    sion_one_above[-1] = sion[0]
    ham[sion, sion_one_above] = t
    ham[sion_one_above, sion] = t

    return(ham)

#########################################################################################################

def get_points_that_connect_with_prev_layer_q3_general_optimized(p, specific_level, first_site_in_layer_index):
    q = 3
    motif_string = np.tile(get_motif_of_sector_symbolic_q3_general(p, specific_level), p)
    motif_string_innumbers = np.zeros(np.shape(motif_string), dtype=int)
    motif_string_innumbers[np.where(motif_string == '4side')] = 1
    motif_translator = np.array([p-3, p-4])
    prev_layer_connected = np.cumsum(motif_translator[motif_string_innumbers])
    prev_layer_connected = np.concatenate((np.array([0]), prev_layer_connected))
    prev_layer_connected = prev_layer_connected[:-1] #To avoid double counting first point
    return(prev_layer_connected + first_site_in_layer_index)

def get_if_point_is_connected_with_upper_layer_q3_general_optimized(p, specific_level):
    #Since get_motif_of_sector_symbolic_q3_general labels generations starting with 0
    specific_level = specific_level - 1
    entire_motif = np.tile(get_motif_of_sector_symbolic_q3_general(p, specific_level+2), p)

    entire_motif_numversion = np.zeros(np.shape(entire_motif), dtype=int)
    entire_motif_numversion[np.where(entire_motif == '4side')] = 1

    motif_translator = np.array([np.array([True]), np.array([True, False])], dtype=object)
    isconnected = np.concatenate(motif_translator[entire_motif_numversion])

    isconnected = np.concatenate((np.array([False]), isconnected))

    return(isconnected[:-1]) #To avoid over count of last value

#Just general_q3_hamiltonian_optimized but with above two optimized methods used here instead
def general_q3_hamiltonian_superoptimized(p, num_levels):
    sites_per_level, tot_num_sites = number_points_q3_general_from_repeating_pattern(p, num_levels)
    first_sites_of_each_level = np.array([0])
    for n in range(num_levels):
        first_sites_of_each_level = np.append(first_sites_of_each_level, sites_per_level[n] + np.sum(sites_per_level[:n]))

    ham = dok_matrix((tot_num_sites, tot_num_sites), dtype=int)
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
        ham[sion, first_gen_inter_connect_hardcode] = t
        ham[first_gen_inter_connect_hardcode, sion] = t

    for n in range(1, num_levels-1): #iterate over all levels except first and last level
        print('Working on generation: {}'.format(n+1))
        #sion stands for: site_indices_on_level
        sion = np.arange(first_sites_of_each_level[n], first_sites_of_each_level[n+1])

        # More optimized NN intra layer hopping
        sion_one_above = sion + 1
        sion_one_above[-1] = sion[0]
        ham[sion, sion_one_above] = t
        ham[sion_one_above, sion] = t

        #Make inter generation connections
        #points_this_inter stands for: points_onthislayer_that_interconnect
        #points_next_inter stands for: points_onnextlayer_that_interconnect
        st = time.time()
        points_this_inter = sion[get_if_point_is_connected_with_upper_layer_q3_general_optimized(p, n)]
        print(time.time() - st)
        #Not using get_points_that_connect_with_prev_layer_q3_general_optimized since its actually not optimized
        st = time.time()
        points_next_inter = get_points_that_connect_with_prev_layer_q3_general_optimized(p, n+1, first_sites_of_each_level[n+1])
        print(time.time()-st)
        ham[points_this_inter, points_next_inter] = t
        ham[points_next_inter, points_this_inter] = t

    #And finally for last layer
    sion = np.arange(first_sites_of_each_level[num_levels-1], first_sites_of_each_level[num_levels])

    # More optimized NN intra layer hopping
    sion_one_above = sion + 1
    sion_one_above[-1] = sion[0]
    ham[sion, sion_one_above] = t
    ham[sion_one_above, sion] = t

    return(ham)

#########################################################################################################

def number_points_q4_general(p, num_levels):
    num_sites_per_gen = np.array([])
    threeside_counter = 0
    twoside_counter = 0
    twoside_counter_old = 0
    threeside_counter_old = 0
    for n in range(1, num_levels+1):
        # print(n, threeside_counter, twoside_counter, threeside_counter_old, twoside_counter_old)
        if n == 1:
            num_sites_per_gen = np.append(num_sites_per_gen, p)
            threeside_counter = threeside_counter + 1
        else:
            # num_sites_per_gen = np.append(num_sites_per_gen, ((p-2)*threeside_counter + (p-3)*twoside_counter - 2*twoside_counter_old)*p)
            # twoside_counter_old = twoside_counter
            # threeside_counter_old = threeside_counter
            # threeside_counter = (p-3)*threeside_counter_old + (p-2)*twoside_counter_old
            # twoside_counter = threeside_counter_old - twoside_counter_old
            num_sites_per_gen = np.append(num_sites_per_gen, ((p - 2) * threeside_counter + (p - 3) * twoside_counter) * p)
            twoside_counter_old = twoside_counter
            threeside_counter_old = threeside_counter
            threeside_counter = (p - 3) * threeside_counter_old + (p - 4) * twoside_counter_old
            twoside_counter = threeside_counter_old - twoside_counter_old + 2*twoside_counter_old
    return(num_sites_per_gen, np.sum(num_sites_per_gen))
