# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 20:41:09 2021

@author: arosl
"""

import pandas as pd
import numpy as np
from pathlib import Path
from io import StringIO
from tqdm import tqdm
from Bio import SeqIO, AlignIO, Entrez
from Bio.Blast import NCBIXML
from collections import defaultdict
import argparse
import ast


def init_argparser():
    input_parser = argparse.ArgumentParser()
    input_parser.add_argument('-e',
                              '--email',
                              required=True
                              )
    input_parser.add_argument('-cd',
                              '--cluster_dir',
                              required=True
                              )
    input_parser.add_argument('-bd',
                              '--blast_dir',
                              required=True
                              )
    return input_parser


def mkdir(path_obj):
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def blast_parser(xml_file):
    xml_file = Path(xml_file)
    print(f"Parsing BLAST xml output of {xml_file.stem}...", flush=True)
    best_hits = {}
    handle = open(xml_file)
    records = NCBIXML.parse(handle)
    for record in tqdm(records):
        epitope_length = record.query_length
        hits = []
        if record.alignments:
            for alignment in record.alignments:
                for hsp in alignment.hsps:
                    true_identity = hsp.identities - hsp.gaps
                    new_hit = (alignment.title.split("|")[1],
                               round(true_identity / epitope_length * 100, 2),
                               hsp.expect,
                               hsp.bits,
                               hsp.score
                               )
                    hits.append(new_hit)
            if hits:
                best_hits[record.query] = sorted(hits, key=lambda x: x[2])[0]
    handle.close()
    print("\ndone!", flush=True)
    return best_hits


def parse_list_tuples(list_of_tuples):
    result = []
    tuples = list_of_tuples.strip('][')
    number_of_tuples = tuples.count("(")
    items = [x.replace("(", "") for x in tuples.split(")") if x]
    for i in items:
        tuple_items = [x.strip() for x in i.split(",") if x]
        tuple_items_list = []
        for t in tuple_items:
            try:
                val = int(t)
                tuple_items_list.append(val)
            except ValueError:
                val = t.strip("'").strip('"')
                tuple_items_list.append(val)
        result.append(tuple(tuple_items_list))
    if len(result) == number_of_tuples:
        return result
    else:
        raise Exception("Could not parse results")


def read_cluster_file(cluster_file):
    parsed_data = {}
    with open(cluster_file, encoding='utf-8') as f:
        lines = f.read()
    parts = lines.split("¡¡¡¡¡")
    for part in parts:
        if part:
            header, content = part.split("!!!!!")
            if header == "CLUSTER_NAME":
                parsed_data["CLUSTER_NAME"] = content.strip()
            elif header == "REFSEQ_ID":
                parsed_data["REFSEQ_ID"] = content.strip()
            elif header == "REFSEQ_SEQ":
                parsed_data["REFSEQ_SEQ"] = content.strip()
            elif "MASKED_SEQ" in header:
                parsed_data["MASKED_SEQ"] = content.strip()
            elif "ALIGNMENT" in header:
                aln_temp = StringIO(content)
                aln = AlignIO.read(aln_temp, "fasta")
                parsed_data["ALIGNMENT"] = aln
            elif header == "POSITION_DATA":
                df = pd.read_csv(StringIO(content), sep=";")
                for ix in df.index:
                    distr = df["distribution"].iloc[ix]
                    distr = ast.literal_eval(distr)
                    df["distribution"].iloc[ix] = distr
                parsed_data["POSITION_DATA"] = df
    return parsed_data


def get_tier(epitope_values,
             rsa_thres=50,
             b_thres=1,
             bepi_thres=0.5,
             hydro_thres=0.5
             ):

    rsa_vals = np.array([x[6] for x in epitope_values])
    b_vals = np.array([x[7] for x in epitope_values])
    bepi_vals = np.array([x[4] for x in epitope_values])
    hydro_vals = np.array([x[5] for x in epitope_values if not np.isnan(x[5])])

    if rsa_vals.mean() >= rsa_thres:
        rsa_flag = True
    else:
        rsa_flag = False

    if b_vals.mean() >= b_thres:
        bval_flag = True
    else:
        bval_flag = False

    if bepi_vals.mean() >= bepi_thres:
        bepi_flag = True
    else:
        bepi_flag = False

    if hydro_vals.mean() <= hydro_thres:
        hydro_flag = True
    else:
        hydro_flag = False

    # Tier 1
    if (rsa_flag
            and bval_flag
            and bepi_flag
            and hydro_flag):
        return 1

    # Tier 2
    if (rsa_flag
            and bval_flag
            and bepi_flag):
        return 2

    # Tier 3
    if (rsa_flag
            and bval_flag
            and hydro_flag):
        return 3

    # Tier 4
    if (rsa_flag
            and bval_flag):
        return 4

    # Tier 5
    if (rsa_flag
            and bepi_flag
            and hydro_flag):
        return 5

    # Tier 6
    if (rsa_flag
            and bepi_flag):
        return 6

    # Tier 7
    if (bepi_flag
            and hydro_flag):
        return 7

    else:
        return None


def parse_epitopes(positions_df,
                   blast_dict,
                   rsa_thres=50,
                   b_thres=1,
                   bepi_thres=0.5,
                   hydro_thres=0.45,
                   entropy_thres=0,
                   epi_length=8,
                   gapped=False
                   ):
    records = positions_df.to_records(index=False)
    residues_list = [tuple(x) for x in records]

    current_region = []
    gapless_regions = []

    # If we set gapped to True, we will separate the sequence into regions when
    # it finds that the consensus is a gap. This can potentially break up
    # epitopes with only one inserted position over many. Best to be only used
    # in strict cases such as entropy 0.
    if gapped:
        for x in residues_list:
            if not x[1] == "-":
                current_region.append(x)
            else:
                gapless_regions.append(current_region)
                current_region = []
        gapless_regions.append(current_region)
    else:
        residues_list = [x for x in residues_list if not x[1] == "-"]
        gapless_regions.append(residues_list)

    # We split the sequence according to the entropy, and keep only those
    # regions that have the minimum length
    conserved_regions = []
    current_split = []
    for region in gapless_regions:
        if region:
            for position in region:
                entropy = position[3]
                if entropy <= entropy_thres:
                    current_split.append(position)
                else:
                    if len(current_split) >= epi_length:
                        conserved_regions.append(current_split)
                        current_split = []
                    else:
                        current_split = []
            if len(current_split) >= epi_length:
                conserved_regions.append(current_split)

    # We have to calculate the mean values of our epitope. Otherwise, it might
    # break by just one residue with a diff of 0.1 to the threshold. We will
    # parse length by length the sequence, and calculate the mean. If the
    # following contiguous region is an epitope too, its last position will be
    # appended to the previous epitope, thus making it bigger (instead of
    # creating two overlapping epitopes)
    epitopes = []
    current_region = []
    for conserved_region in conserved_regions:
        for i in range(len(conserved_region)):
            high_blast_flag = False
            region = conserved_region[i:i+epi_length]

            if len(region) < epi_length:
                if len(current_region) >= epi_length:
                    epitopes.append(current_region)
                    current_region = []
                else:
                    current_region = []
                break

            try:
                region_seq = "".join(x[1] for x in region)
                blast_hits = blast_dict[region_seq]
                for hit in blast_hits:
                    if hit[1] > 75:
                        high_blast_flag = True
                        if len(current_region) >= epi_length:
                            epitopes.append(current_region)
                            current_region = []
                        else:
                            current_region = []
                        break
                if high_blast_flag:
                    continue
            except Exception as e:
                print("kmer not found in BLAST results")
                print(e)

            epitope_tier = get_tier(region,
                                    rsa_thres,
                                    b_thres,
                                    bepi_thres,
                                    hydro_thres,
                                    )

            if epitope_tier:
                if not current_region:
                    for pos in region:
                        current_region.append([epitope_tier, pos])
                else:
                    start_pos = region[0]
                    for current_pos in current_region:
                        if start_pos == current_pos[1]:
                            start_index = current_region.index(current_pos)
                    for counter, pos in enumerate(region):
                        current_index = start_index + counter
                        if current_index < len(current_region):
                            current_pos = current_region[current_index]
                            if not current_pos[1] == pos:
                                raise Exception("Merged epitopes do not match")
                            # If the new tier is better than the last one, we
                            # will update it. This way we can end up with mixed
                            # tier epitopes, with some parts of the epitope
                            # "better" than others.
                            if epitope_tier < current_pos[0]:
                                current_region[current_index][0] = epitope_tier
                        else:
                            current_region.append([epitope_tier, pos])
            else:
                if len(current_region) >= epi_length:
                    epitopes.append(current_region)
                    current_region = []
                else:
                    current_region = []

    # We change the structure, and append the tier at the end of each position.
    tier_epitopes = []
    current_epitope = []
    for epitope in epitopes:
        for tier, position in epitope:
            new_position = list(position)
            new_position.append(tier)
            current_epitope.append(new_position)
        tier_epitopes.append(current_epitope)
        current_epitope = []

    return tier_epitopes


def get_protein_names(*args):
    return_names = {k: "UNK" for k in args}
    identificators = []
    for arg in args:
        protid = arg.split("_")[1]
        if "pir" in protid:
            protid = protid.split("pir||")[1]
        protid = "_".join(protid.split("|"))
        identificators.append(protid)
    with Entrez.efetch(db="protein",
                       id=identificators,
                       rettype="gp",
                       retmode="text"
                       ) as handle:
        records = list(SeqIO.parse(handle, "gb"))

        for record in records:
            protein_id = record.id.split(".")[0].split("_")[0]
            match = ""
            for arg in args:
                if protein_id in arg:
                    match = arg
                    break
            if match:
                protein_name = record.description
                if "RecName: Full=" not in protein_name:
                    return_names[match] = protein_name
                else:
                    splitted = protein_name.split(";")[0]
                    new_name = splitted.split("RecName: Full=")[1]
                    return_names[match] = new_name
    return return_names


def family_keywords(string):
    family_kwds = {"NP": ["natriuretic"],
                   "KUN": ["kunitz"],
                   "SNACLEC": ["c-type", "lectin", "snaclec"],
                   "SVMP": ["SVMP", "reprolysin", "metallopeptidase",
                            "metalloprotease", "metalloproteinase", "adam",
                            "m12b"
                            ],
                   "3FTX": ["three", "finger", "three-finger",
                            "3ftx", "snake toxin", "tolip"],
                   "PLA2": ["phospholipase", "pla2"],
                   "DIS": ["disintegrin"],
                   "CRISP": ["secretory", "crisp", "cysteine-rich"],
                   "LAAO": ["oxidase", "amine", "laao"],
                   "SVSP": ["SVSP", "snake venom serine protease", "serine",
                            "proteases"],
                   }
    families = set()
    for fam, keywords in family_kwds.items():
        for keyword in keywords:
            if keyword in string.lower():
                families.add(fam)
        if families:
            break
    if families:
        return list(families)
    else:
        return None


def get_protein_families(*args):
    fam_dict = {k: "OTHER" for k in args}
    id_dict = {}
    for arg in args:
        prot_id = arg.split("_")[1]
        if "pir||" in prot_id:
            prot_id = prot_id.replace("pir||", "")
        if "|" in prot_id:
            prot_id = "_".join(prot_id.split("|"))
        id_dict[prot_id] = arg
    with Entrez.efetch(db="protein",
                       id=id_dict.keys(),
                       rettype="gp",
                       retmode="text"
                       ) as handle:
        records = list(SeqIO.parse(handle, "gb"))

    for record in records:
        new_id = record.id.split(".")[0]

        try:
            refseq = id_dict[new_id]
        except KeyError:
            raise Exception("ID not parsed correctly")

        family_list = []

        # First, check name
        name = record.description.replace("RecName: Full=", "")
        name_families = family_keywords(name)
        if name_families:
            family_list.extend(name_families)
            fam_dict[refseq] = ("/".join(family_list), name)
            continue

        # If no results, then we check features
        features = record.features
        for feature in features:
            # First check Protein feature
            if feature.type.lower() == "protein":
                quals = feature.qualifiers
                items = []
                for k, v in quals.items():
                    for x in v:
                        items.append(x)
                for item in items:
                    item_families = family_keywords(item)
                    if item_families:
                        family_list.extend(item_families)
                if family_list:
                    break
            if feature.type.lower() == "region":
                quals = feature.qualifiers
                items = []
                for k, v in quals.items():
                    for x in v:
                        items.append(x)
                for item in items:
                    item_families = family_keywords(item)
                    if item_families:
                        family_list.extend(item_families)
        family_set = set(family_list)
        if not family_set:
            fam_dict[refseq] = ("OTHER", name)
        else:
            fam_dict[refseq] = ("/".join(family_set), name)
    return fam_dict


def get_tier_epitope(epitope, tier, epi_len=8):
    tier_epis = []
    current_epi = []
    for pos in epitope:
        if pos[-1] <= tier:
            current_epi.append(pos)
        else:
            if len(current_epi) >= epi_len:
                tier_epis.append(current_epi)
                current_epi = []
            else:
                current_epi = []
    if len(current_epi) >= epi_len:
        tier_epis.append(current_epi)

    if not tier_epis:
        return None

    final_epis = []
    for epi in tier_epis:
        epitope_seq = "".join(x[1] for x in epi)
        epitope_quality = "".join(str(x[-1]) for x in epi)
        depth = [x[2] for x in epi]
        min_depth = min(depth)
        max_depth = max(depth)
        epi_data = (epitope_seq,
                    (min_depth, max_depth),
                    epitope_quality
                    )
        final_epis.append(epi_data)
    return final_epis


def fastaToDict(fasta_file):
    """Simply generates a dictionary from a fasta file

    Args:
        fasta_file (file in fasta format)

    Returns:
        dictionary: {record1_id: record1_seq, record2_id: record2_seq, ...}
    """
    with open(fasta_file, "r") as f:
        fasta_dict = {record.id: str(record.seq)
                      for record in SeqIO.parse(f, "fasta")
                      }
    return fasta_dict


def transform_epi(epi):
    new_epi = []
    new_epi.extend(epi[:2])
    epi_cover = f"1-{len(epi[7])}"

    new_epi_cluster = defaultdict(list, {epi_cover: [epi[2]]})
    new_epi.append(new_epi_cluster)

    new_epi_spp = defaultdict(list, {epi_cover: epi[3].split("; ")})
    new_epi.append(new_epi_spp)

    epi_protein = epi[4].split("_")[1]
    protein_name = epi[5]
    new_epi_protein = defaultdict(list,
                                  {epi_cover: [(epi_protein, protein_name)]})
    new_epi.append(new_epi_protein)

    new_epi_family = [epi[6]]
    new_epi.append(new_epi_family)

    new_epi.extend(epi[7:])

    new_epi.append("1"*len(epi[7]))

    return new_epi


def merge_entire_epitopes(big_epi, small_epi):
    final_epi = [x for x in big_epi]

    # First, let's find the alignment start and end. Since the small epitope is
    # entirely inside the big one, the alignment is simple and we can brute
    # force it by iterating through the sequence
    big_epi_seq = final_epi[6]
    small_epi_seq = small_epi[6]
    for pos, aa in enumerate(big_epi_seq):
        if not small_epi_seq[0] == aa:
            continue
        else:
            equal_seq = big_epi_seq[pos:pos+len(small_epi_seq)]
            if small_epi_seq == equal_seq:
                merge_start = pos
                merge_end = pos + len(small_epi_seq)
                break
    small_epi_region = f"{merge_start + 1}-{merge_end}"

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

    # Here we specificy which regions cover which species
    big_epi_spp_dict = final_epi[3]
    small_epi_spp_dict = small_epi[3]
    big_epi_all_spp = big_epi_spp_dict[f"1-{len(big_epi_seq)}"]
    unique_spp = [s[0] for s in small_epi_spp_dict.values()
                  if s[0] not in big_epi_all_spp
                  ]
    if unique_spp:
        big_epi_spp_dict[small_epi_region].extend(unique_spp)
    final_epi[3] = big_epi_spp_dict

    # Same thing with protein origin
    big_epi_prot_dict = final_epi[4]
    small_epi_prot_dict = small_epi[4]
    big_epi_all_prot = big_epi_prot_dict[f"1-{len(big_epi_seq)}"]
    unique_prots = [p for p in small_epi_prot_dict.values()
                    if p not in big_epi_all_prot
                    ]
    if unique_prots:
        big_epi_prot_dict[small_epi_region].extend(unique_prots)
    final_epi[4] = big_epi_prot_dict

    # Protein families
    small_epi_families = small_epi[5]
    final_epi[5].extend(small_epi_families)

    # Cluster
    big_epi_cluster_dict = final_epi[2]
    small_epi_cluster_dict = small_epi[2]
    big_epi_all_clust = big_epi_cluster_dict[f"1-{len(big_epi_seq)}"]
    unique_clust = [c[0] for c in small_epi_cluster_dict.values()
                    if c[0] not in big_epi_all_clust
                    ]
    if unique_clust:
        big_epi_cluster_dict[small_epi_region].extend(unique_clust)
    final_epi[2] = big_epi_cluster_dict

    # Depth
    big_epi_min, big_epi_max = eval(str(final_epi[7]))
    small_epi_min, small_epi_max = eval(str(small_epi[7]))
    new_min = max(big_epi_min, small_epi_min)
    new_max = max(big_epi_max, small_epi_max)
    final_epi[7] = str((new_min, new_max))

    return final_epi


def merge_epitopes(epitope_list):
    records = [list(x) for x in epitope_list]
    records.sort(key=lambda x: len(x[7]), reverse=True)

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


def sort_key(item):
    result = []
    k, v = item
    parts = k.split("-")
    result.append(int(parts[0]))
    result.append(1 - int(parts[1]))
    return tuple(result)


# %%
if __name__ == '__main__':
    parser = init_argparser()
    args = parser.parse_args()
    Entrez.email = args.email
    clusters_data_dir = Path(args.cluster_dir)
    blast_data_dir = Path(args.blast_dir)

    # %%
    epitopes_dir = mkdir(Path("./RESULTS/09_EPITOPES"))

    # Parse the xml blast results
    blast_xml_list = list(blast_data_dir.glob("./*.xml"))
    blast_main_dict = defaultdict(list)
    for blast_xml in blast_xml_list:
        blast_dict = blast_parser(blast_xml)
        for kmer, hit in blast_dict.items():
            blast_main_dict[kmer].append(hit)

    main_dict = {}
    protein_ids = set()
    groups_set = set()
    cutoffs_set = set()

    cluster_file_list = list(clusters_data_dir.glob("**/*.cluster"))
    for cluster in tqdm(cluster_file_list):
        print(f"\n Working with {cluster.stem}")
        cutoff, cluster_name = cluster.stem.split("_")
        group = cluster.parent.parent.stem

        groups_set.add(group)
        cutoffs_set.add(cutoff)

        data = read_cluster_file(cluster)
        positions_df = data["POSITION_DATA"]

        refseq_id = data["REFSEQ_ID"]
        protein_ids.add(refseq_id)

        alignment = data["ALIGNMENT"]
        spp_set = set()
        for a in alignment:
            spp = a.id.rsplit("_", 1)[1]
            spp_set.add(spp)

        epitopes = parse_epitopes(positions_df,
                                  blast_main_dict,
                                  entropy_thres=0,
                                  rsa_thres=50,
                                  b_thres=1,
                                  bepi_thres=0.5,
                                  hydro_thres=0.45,
                                  epi_length=8,
                                  gapped=True
                                  )
        key = (group,
               cutoff,
               cluster_name,
               refseq_id,
               "; ".join(spp_set)
               )
        main_dict[key] = epitopes
    protein_names_dict = get_protein_names(*protein_ids)
    protein_families_dict = get_protein_families(*protein_ids)

    dfs_dict = {}
    for tier in range(1, 8):
        filtered_epis = []
        for key, epitopes in main_dict.items():
            group = key[0]
            cutoff = key[1]
            cluster_name = key[2]
            refseq_id = key[3]
            snake_spp = key[4]
            protein_name = protein_names_dict[refseq_id]
            protein_family = protein_families_dict[refseq_id][0]
            for epitope in epitopes:
                tier_epitopes = get_tier_epitope(epitope, tier, epi_len=8)
                if tier_epitopes:
                    for epi in tier_epitopes:
                        best_tier = min(epi[2])
                        epi_data = [group,
                                    cutoff,
                                    cluster_name,
                                    snake_spp,
                                    refseq_id,
                                    protein_name,
                                    protein_family,
                                    epi[0],
                                    epi[1],
                                    epi[2],
                                    best_tier
                                    ]
                        filtered_epis.append(epi_data)

        formated_records = []
        for epi in filtered_epis:
            new_record = []
            new_record.extend(epi[:4])
            new_record.append(epi[4].split("_")[1])
            new_record.extend(epi[5:8])
            new_record.append(epi[8][0])
            new_record.append(epi[8][1])
            new_record.extend(epi[9:])
            formated_records.append(new_record)
        df = pd.DataFrame(formated_records,
                          columns=["group",
                                   "cutoff",
                                   "cluster",
                                   "species",
                                   "protein",
                                   "protein_name",
                                   "protein_family",
                                   "epitope",
                                   "min_depth",
                                   "max_depth",
                                   "seq_tier",
                                   "best_tier",
                                   ]
                          )
        dfs_dict[tier] = df

    excel_path = epitopes_dir / "predicted_epitopes.xlsx"
    writer = pd.ExcelWriter(excel_path, engine='openpyxl')
    for tier, df in sorted(dfs_dict.items()):
        df.to_excel(writer, sheet_name=f"tier_{str(tier)}")
    writer.save()
    writer.close()
