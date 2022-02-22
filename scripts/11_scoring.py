# -*- coding: utf-8 -*-
"""
Created on Jul 1 2021

Finally, it will merge epitopes, score them and
make a selection.

@author: Albert Ros-Lucas
"""

import pandas as pd
from pathlib import Path
import ast
import numpy as np
from collections import defaultdict, Counter
import csv
import argparse
from tqdm import tqdm


def init_argparser():
    input_parser = argparse.ArgumentParser()
    input_parser.add_argument('-e',
                              '--excel',
                              required=True
                              )
    input_parser.add_argument('-b',
                              '--iedb_bcell',
                              required=True
                              )
    input_parser.add_argument('-t',
                              '--iedb_tcell',
                              required=True
                              )
    input_parser.add_argument('-v',
                              '--snake_venom',
                              required=True
                              )
    input_parser.add_argument('-cd',
                              '--clusters_dir',
                              required=True
                              )
    input_parser.add_argument('-spp',
                              '--snake_species',
                              required=True
                              )
    return input_parser


def mkdir(path_obj):
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def parse_iedb_b_csv(csv_file):
    iedb_epitopes = {}
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        next(reader)
        next(reader)
        for line in reader:
            iedb_epitopes[line[0]] = (line[11],  # Sequence
                                      # line[15],  # Antigen
                                      # line[17],  # Antigen ID
                                      # line[20],  # Species
                                      # Assay
                                      f"{line[71]}|{line[72]}|{line[74]}"
                                      )
    return iedb_epitopes


def parse_iedb_t_csv(csv_file):
    iedb_epitopes = {}
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        next(reader)
        next(reader)
        for line in reader:
            iedb_epitopes[line[0]] = (line[11],  # Sequence
                                      # line[15],  # Antigen
                                      # line[17],  # Antigen ID
                                      # line[20],  # Species
                                      # Assay
                                      f"{line[84]}|{line[85]}|{line[87]}"
                                      )
    return iedb_epitopes


def parse_list_tuples(list_of_tuples):
    result = []
    tuples = list_of_tuples.strip()
    tuples = tuples.strip('][')
    number_of_tuples = tuples.count("(")
    items = [x.replace("(", "") for x in tuples.split(")") if x]
    for i in items:
        val = ast.literal_eval(i)
        if type(val) == tuple:
            result.append(val)
        else:
            raise Exception("Could not parse results")
    if len(result) == number_of_tuples:
        return result
    else:
        raise Exception("Could not parse results")


def parse_epitope_proteins(string):
    proteins_dict = {}
    proteins = string.split("\n")
    for protein in proteins:
        pos_key, rest = protein.split(":", 1)
        rest = rest.strip()
        list_tuples = parse_list_tuples(rest)
        proteins_dict[pos_key] = list_tuples
    return proteins_dict


def parse_epitope_species(string):
    species_dict = {}
    species_regions = string.split("\n")
    for species_list in species_regions:
        pos_key, rest = species_list.split(":", 1)
        rest = rest.strip()
        taxid_list = ast.literal_eval(rest)
        species_dict[pos_key] = taxid_list
    return species_dict


def parse_epitopes_clusters(string):
    clusters_dict = {}
    cluster_regions = string.split("\n")
    for cluster_list in cluster_regions:
        pos_key, rest = cluster_list.split(":", 1)
        rest = rest.strip()
        clusters_list = ast.literal_eval(rest)
        clusters_dict[pos_key] = clusters_list
    return clusters_dict


def parse_against_iedb(df, iedb_epitopes):
    df["IEDB"] = "N"
    for ix in df.index:
        epi = df["seq"].iloc[ix]
        for iedb_epi in iedb_epitopes.values():
            if epi in iedb_epi:
                df["IEDB"].iloc[ix] = "Y"
            elif iedb_epi in epi:
                df["IEDB"].iloc[ix] = "Y"
    return df


def epitope_score(venom_dict,
                  seq_tier,
                  species_list,
                  protein_family,
                  max_depth,
                  iedb_flag
                  ):
    """
    First, we assign a base score:

    · For the epitope tier, we have tiers 1-4 (PDB), 5 and 6 (models) and 7
    (Bepipred). We assign values 3, 2 and 1 accordingly.

    · For each extra snake species, we give one extra point (up to 2)

    · If cluster has more than 10 sequences, we give 2 extra points. If it
    has more than 5, 1 point.

    · Finally, if it has an IEDB epitope, we give it 3 extra points.

    The maximum total is 10, while the minimum will be 1.

    For the final score, we multiply the base score by the average venom
    proportion of the species. The theoretical maximum score would be 100
    (unrealistic), and the minimum 0 (will happen as there are protein families
    not present in species)
    """
    # Tier score
    seq_tier_vals = [int(x) for x in seq_tier]
    best_value = min(seq_tier_vals)
    if best_value <= 4:
        base_score = 3
    elif best_value <= 6:
        base_score = 2
    else:
        base_score = 1

    # Snake species
    if len(species_list) == 1:
        base_score += 0
    elif len(species_list) == 2:
        base_score += 1
    else:
        base_score += 2

    # Cluster depth
    if max_depth >= 10:
        base_score += 2
    if max_depth >= 5:
        base_score += 1
    else:
        base_score += 0

    # IEDB
    if iedb_flag:
        base_score += 3
    else:
        base_score += 0

    # Multiply by venom proportion for each species
    proportions = []
    for species in species_list:
        venom_proportions = venom_dict[species]
        proportion = venom_proportions[protein_family] / 10
        proportions.append(proportion)
    mean_proportion = np.array(proportions).mean()

    final_score = base_score * mean_proportion

    return final_score


def normalize_score(score, max_score, min_score):
    norm_score = (score - min_score) / (max_score - min_score)
    return int(norm_score * 100)


def merge_entire_epitopes(big_epi, small_epi):
    final_epi = [x for x in big_epi]

    # First, let's find the alignment start and end. Since the small epitope is
    # entirely inside the big one, the alignment is simple and we can brute
    # force it by iterating through the sequence
    big_epi_seq = final_epi[5]
    small_epi_seq = small_epi[5]
    for pos, aa in enumerate(big_epi_seq):
        if not small_epi_seq[0] == aa:
            continue
        else:
            equal_seq = big_epi_seq[pos:pos+len(small_epi_seq)]
            if small_epi_seq == equal_seq:
                merge_start = pos
                break

    # Merge clusters
    final_epi[2] = final_epi[2].union(small_epi[2])

    # Merge species
    final_epi[3] = final_epi[3].union(small_epi[3])

    # Merge proteins
    final_epi[4] = final_epi[4].union(small_epi[4])

    # Merge depth
    big_epi_min = final_epi[6]
    big_epi_max = final_epi[7]
    small_epi_min = small_epi[6]
    small_epi_max = small_epi[7]
    new_min = max(big_epi_min, small_epi_min)
    new_max = max(big_epi_max, small_epi_max)
    final_epi[6] = new_min
    final_epi[7] = new_max

    # Merge the tiers of both epitopes, keeping the best for each position
    big_epi_qual = list(final_epi[8])
    small_epi_qual = list(small_epi[8])
    for count, small_qual in enumerate(small_epi_qual):
        big_qual = big_epi_qual[merge_start + count]
        best_qual = min(int(small_qual), int(big_qual))
        big_epi_qual[merge_start + count] = str(best_qual)
    new_best_tier = min(big_epi_qual)
    final_epi[8] = "".join(big_epi_qual)
    final_epi[9] = new_best_tier

    # Merge IEDB
    small_epi_iedb = small_epi[10]
    if small_epi_iedb:
        final_epi[10] += small_epi_iedb

    # Score
    big_epi_score = final_epi[11]
    small_epi_score = small_epi[11]
    new_score = max(big_epi_score, small_epi_score)
    final_epi[11] = new_score

    # We want to count how many times an epitope is merged,
    abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    big_epi_ocur = list(final_epi[-1])
    for i in range(len(small_epi_seq)):
        big_ocur = big_epi_ocur[merge_start + i]

        if big_ocur in abc:
            ix = abc.find(big_ocur)
            new_ocur = abc[ix + 1]
            big_epi_ocur[merge_start + i] = new_ocur
        else:
            new_ocur = int(big_ocur) + 1
            if new_ocur < 10:
                big_epi_ocur[merge_start + i] = str(new_ocur)
            else:
                diff = 10 - new_ocur
                diff_dict = {i: x for i, x in enumerate(list(abc))}
                big_epi_ocur[merge_start + i] = diff_dict[diff]
    final_epi[-1] = "".join(big_epi_ocur)

    return final_epi


def transform_epi(epi):
    new_epi = []
    new_epi.extend(epi[:2])

    new_epi_cluster = set([epi[2]])
    new_epi.append(new_epi_cluster)

    new_epi_spp = set([x for x in epi[3].split("; ")])
    new_epi.append(new_epi_spp)

    prot = epi[4]
    prot_name = epi[5]
    prot_family = epi[6]
    new_epi_protein = set([f"{prot} || {prot_name} || {prot_family}"])
    new_epi.append(new_epi_protein)

    new_epi.extend(epi[7:])

    new_epi.append("1"*len(epi[7]))

    return new_epi


def merge_epitopes(epitope_list):
    records = sorted(epitope_list, key=lambda x: len(x[7]), reverse=True)

    if not records:
        return []

    parsed_records = {records[0][7]: transform_epi(records[0])}
    for ix, epi in enumerate(records[1:]):
        merged = False
        epi_seq = epi[7]
        parsed_epi = transform_epi(epi)
        for other_epi in sorted(parsed_records.keys(),
                                key=lambda x: len(x),
                                reverse=True
                                ):
            if epi_seq in other_epi:
                merged_epi = merge_entire_epitopes(parsed_records[other_epi],
                                                   parsed_epi
                                                   )
                parsed_records[other_epi] = merged_epi
                merged = True
                break
        if not merged:
            parsed_records[epi_seq] = parsed_epi
    return_records = list(parsed_records.values())
    return return_records


# %%
if __name__ == '__main__':
    parser = init_argparser()
    args = parser.parse_args()
    excel_path = Path(args.excel)
    iedb_bcell_path = Path(args.iedb_bcell)
    iedb_tcell_path = Path(args.iedb_tcell)
    snake_venom_csv = Path(args.snake_venom)
    snake_spp_csv = Path(args.snake_species)
    clusters_dir = Path(args.clusters_dir)

    root_output = mkdir(Path("./RESULTS/10_SCORING/"))

    iedb_b_epis = parse_iedb_b_csv(iedb_bcell_path)
    iedb_b_epis = set(iedb_b_epis.values())

    tiered_dataframes = pd.read_excel(excel_path, sheet_name=None, index_col=0)

    snake_spp_list = []
    snakes_df = pd.read_csv(snake_spp_csv)
    for ix in snakes_df.index:
        taxid = str(snakes_df["taxid"].iloc[ix])
        fam = snakes_df["family"].iloc[ix]
        snake_spp_list.append((taxid, fam))

    venom_df = pd.read_csv(snake_venom_csv, decimal=",")
    venom_families = [x.upper() for x in venom_df.columns[3:]]
    snake_families = set([x for x in snakes_df["family"]])

    default_venom_dict = defaultdict(dict)
    for x in snake_families:
        for y in venom_families:
            default_venom_dict[x][y] = 100/len(venom_families)

    for family, empty_dict in default_venom_dict.items():
        for protein_family in venom_families:
            values = []
            for ix in venom_df.index:
                if not pd.isna(venom_df["OTHER"].iloc[ix]):  # no info
                    if venom_df['family'].iloc[ix] == family:
                        percent = venom_df[protein_family].iloc[ix]
                        if pd.isna(percent):
                            percent = 0
                        values.append(percent)
            if values:
                mean = np.array(values).mean()
                default_venom_dict[family][protein_family] = mean
            else:
                continue

    main_venom_dict = {}
    for snake in snake_spp_list:
        main_venom_dict[snake[0]] = default_venom_dict[snake[1]]

    parsed_venom_dict = {}
    for ix in venom_df.index:
        taxid = str(venom_df["taxid"].iloc[ix])
        if not pd.isna(venom_df["OTHER"].iloc[ix]):  # no info on this sp
            family_dict = {}
            for family in venom_families:
                venom_proportion = venom_df[family].iloc[ix]
                if pd.isna(venom_proportion):
                    venom_proportion = 0
                family_dict[family] = venom_proportion
            parsed_venom_dict[taxid] = family_dict

    for taxid in main_venom_dict:
        if taxid in parsed_venom_dict:
            main_venom_dict[taxid] = parsed_venom_dict[taxid]

    # Now we score the epitopes
    print("Scoring epitopes...")
    parsed_dfs = {}
    for tier, df in tiered_dataframes.items():
        scores = defaultdict(lambda: defaultdict(list))
        normalized_scores = defaultdict(lambda: defaultdict(list))
        df["IEDB"] = ""
        df["score"] = 0
        for ix in tqdm(df.index):
            group = df["group"].iloc[ix]
            cutoff = "0" + str(df["cutoff"].iloc[ix])

            iedb_flag = False
            # Get the sequence tier info
            seq_tier = str(df["seq_tier"].iloc[ix])

            # Get species list
            species_list = str(df["species"].iloc[ix]).split("; ")

            # Get venom protein family
            protein_family = df["protein_family"].iloc[ix].upper()

            # Get max depth
            max_depth = df["max_depth"].iloc[ix]

            # Get if IEDB
            epi_seq = df["epitope"].iloc[ix]
            for b in iedb_b_epis:
                b_seq = b[0]
                if epi_seq in b_seq:
                    df["IEDB"].iloc[ix] += f"{b[1]}\n"
                    iedb_flag = True
                elif b_seq in epi_seq:
                    df["IEDB"].iloc[ix] += f"{b[1]}\n"
                    iedb_flag = True

            score = epitope_score(main_venom_dict,
                                  seq_tier,
                                  species_list,
                                  protein_family,
                                  max_depth,
                                  iedb_flag
                                  )
            df["score"].iloc[ix] = score
            scores[group][cutoff].append(score)

        parsed_dfs[tier] = df

    # We want to check whether all the clusters are covered, so
    # we will create a dictionary that will hold which are those.
    main_cluster_dict = defaultdict(lambda: defaultdict(dict))
    cluster_files = list(clusters_dir.glob("**/*.cluster"))
    for cluster in cluster_files:
        group = cluster.parent.parent.stem
        cutoff = cluster.parent.stem
        number = cluster.stem.split("_")[1]
        main_cluster_dict[group][cutoff][number] = [set(), []]

    # We will use Tier 7 dataframe to see if we cover all clusters and spp
    df = parsed_dfs["tier_7"]

    for ix in df.index:
        group = df["group"].iloc[ix]
        cutoff = "0" + str(df["cutoff"].iloc[ix])
        cluster = df["cluster"].iloc[ix]
        cluster_name = df["protein_name"].iloc[ix]
        main_cluster_dict[group][cutoff][cluster][0].add(cluster_name)
        main_cluster_dict[group][cutoff][cluster][1].append(ix)

    all_epitopes = set()
    merged_records = []
    tier_stats_dict = defaultdict(lambda: defaultdict(dict))
    print("Merging epitopes...")
    for group, cutoff_dict in main_cluster_dict.items():
        for cutoff, cluster_dict in tqdm(cutoff_dict.items()):
            tier_stats_dict[group][cutoff] = ""
            ix_to_merge = [x for k, v in cluster_dict.items()
                           for x in v[1]
                           ]
            epis_to_merge = [list(df.iloc[ix]) for ix in ix_to_merge]
            merged_epis = merge_epitopes(epis_to_merge)
            formated_records = []
            for record in merged_epis:
                formated_record = [x for x in record]
                formated_record[2] = "; ".join(formated_record[2])
                formated_record[3] = "; ".join(formated_record[3])
                formated_record[4] = "\n".join(formated_record[4])
                formated_records.append(formated_record)
                all_epitopes.add(formated_record[5])
                tier_stats_dict[group][cutoff] += formated_record[8]

            scores = [x[11] for x in formated_records]
            min_score = min(scores)
            max_score = max(scores)
            for record in formated_records:
                record[11] = normalize_score(record[11],
                                             max_score, min_score
                                             )
            norm_scores = [x[11] for x in formated_records]

            best_records = []
            cluster_combs = set([x[2] for x in formated_records])
            for c_comb in cluster_combs:
                curr_comb = [x for x in formated_records if x[2] == c_comb]
                high_score = max([x[11] for x in curr_comb])
                high_comb = [x for x in curr_comb if x[11] >= high_score]
                sorted_comb = sorted(high_comb, key=lambda x: 1/len(x[5]))
                best_records.append(sorted_comb[0])

            selected_epis = []
            covered_clusters = []
            for cluster in cluster_dict:
                if cluster in covered_clusters:
                    continue

                cluster_epis = []
                for f in best_records:
                    f_clustrs = f[2].split("; ")
                    if any(f_clust == cluster for f_clust in f_clustrs):
                        cluster_epis.append(f)

                cluster_epis = [x for x in cluster_epis if x[11] > 0]
                sorted_epis = sorted(cluster_epis,
                                     key=lambda x: (x[11], len(x[5])),
                                     reverse=True
                                     )
                unique_epis = [x
                               for x in sorted_epis
                               if x not in selected_epis
                               ]
                if unique_epis:
                    best_epi = unique_epis[0]
                    best_epi_clusters = best_epi[2].split("; ")
                    covered_clusters.extend(best_epi_clusters)
                    selected_epis.append(best_epi)
            merged_records.extend(selected_epis)

    selected_epitopes_df = pd.DataFrame(merged_records,
                                        columns=["group",
                                                 "cutoff",
                                                 "clusters",
                                                 "species",
                                                 "proteins",
                                                 "epitope",
                                                 "min_depth",
                                                 "max_depth",
                                                 "seq_tier",
                                                 "best_tier",
                                                 "iedb",
                                                 "score",
                                                 "occurrences"
                                                 ]
                                        )

    covered_clusters_dict = {x: y for x, y in main_cluster_dict.items()}
    for group, cutoffs in covered_clusters_dict.items():
        for cutoff, clusters in cutoffs.items():
            for cluster, data in clusters.items():
                covered_clusters_dict[group][cutoff][cluster] = [data[0], 0]
    snakes_covered = {}
    for group, cutoffs in covered_clusters_dict.items():
        snakes_covered[group] = {x: {y[0]: 0 for y in snake_spp_list}
                                 for x in cutoffs
                                 }

    for ix in selected_epitopes_df.index:
        group = selected_epitopes_df["group"].iloc[ix]
        cutoff = "0" + str(selected_epitopes_df["cutoff"].iloc[ix])
        clusters = selected_epitopes_df["clusters"].iloc[ix].split("; ")
        sp_list = str(selected_epitopes_df["species"].iloc[ix]).split("; ")
        for sp in sp_list:
            snakes_covered[group][cutoff][sp] += 1
        for cluster in clusters:
            covered_clusters_dict[group][cutoff][cluster][1] += 1

    rows = []
    for group, cutoffs in covered_clusters_dict.items():
        for cutoff, clusters in cutoffs.items():
            for cluster, data in clusters.items():
                protein_name = [x for x in data[0]]
                if protein_name:
                    protein_name = protein_name[0]
                else:
                    protein_name = ""
                row = (group, cutoff, cluster, protein_name, data[1])
                rows.append(row)
    covered_clusters_df = pd.DataFrame(rows, columns=["group",
                                                      "cutoff",
                                                      "cluster",
                                                      "protein_name",
                                                      "coverage"
                                                      ]
                                       )

    rows = []
    for group, cutoffs in snakes_covered.items():
        for cutoff, spp in cutoffs.items():
            for sp, cov in spp.items():
                row = (group, cutoff, sp, cov)
                rows.append(row)
    covered_snakes_df = pd.DataFrame(rows, columns=["group",
                                                    "cutoff",
                                                    "sp",
                                                    "coverage"
                                                    ]
                                     )

    rows = []
    for group, cutoff_dict in tier_stats_dict.items():
        for cutoff, seqtier in cutoff_dict.items():
            counter = Counter(seqtier)
            percents = {x: 0 for x in range(1, 8)}
            for tier, counts in counter.items():
                percents[int(tier)] = counts/len(seqtier)
            row = [group, cutoff]
            row.extend([y for x, y in sorted(percents.items())])
            rows.append(tuple(row))
    tier_stats_df = pd.DataFrame(rows, columns=["group",
                                                "cutoff",
                                                1,
                                                2,
                                                3,
                                                4,
                                                5,
                                                6,
                                                7
                                                ]
                                 )

    excel_output_path = root_output / "scored_epitopes.xlsx"
    writer = pd.ExcelWriter(excel_output_path, engine='openpyxl')
    for tier, df in sorted(parsed_dfs.items()):
        df.to_excel(writer, sheet_name=tier)
    selected_epitopes_df.to_excel(writer, sheet_name="selected_epitopes")
    covered_clusters_df.to_excel(writer, sheet_name="clusters_summary")
    covered_snakes_df.to_excel(writer, sheet_name="species_summary")
    tier_stats_df.to_excel(writer, sheet_name="tier_summary")
    writer.save()
    writer.close()

    with open(root_output / "all_epitopes.fasta", "w+") as f:
        for counter, epi in enumerate(all_epitopes, 1):
            if len(epi) >= 15:
                f.write(f">{counter}\n{epi}\n")
