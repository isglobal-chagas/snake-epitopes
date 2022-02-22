# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 17:18:19 2021

Create all possible kmers from the clusters refseqs (in this case 8-mers)
and blast them in the nr database for several organisms.

This BLAST search is run locally, so BLAST needs to be installed in the system,
as well as the "nr" database.

@author: arosl
"""

import subprocess
from pathlib import Path
import pandas as pd
from io import StringIO
from tqdm import tqdm
import multiprocessing
import ast
from Bio import AlignIO


def mkdir(path_obj):
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


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


def gen_split_overlap(seq, size):
    overlaps = set()
    for i in range(0, len(seq) - 1):
        comb = seq[i:i + size]
        if len(comb) == size:
            overlaps.add(comb)
    return list(overlaps)


def epitope_kmers(positions_df,
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

    combinations = []
    for conserved_region in conserved_regions:
        sequence = "".join(x[1] for x in conserved_region)
        possible_kmers = gen_split_overlap(sequence, epi_length)
        combinations.extend(possible_kmers)
    return combinations


def blastp_short_cmd(input_fasta, taxid, database="nr"):
    input_fasta = Path(input_fasta)
    output_file = input_fasta.parent / f"BLASTP_{input_fasta.stem}_{taxid}.xml"
    argums = ['blastp',
              '-task blastp-short',
              f'-query "{input_fasta}"',
              f'-db {database}',
              f'-taxids {taxid}',
              f'-out {output_file}',
              '-word_size 2',
              '-matrix PAM30',
              '-comp_based_stats 0',
              '-gapopen 9',
              '-gapextend 1',
              '-evalue 10000',
              '-outfmt 5',
              '-max_target_seqs 5',
              ]
    blast_cline = " ".join(argums)
    return blast_cline


def execute_cmd(cmd):
    subprocess.run(cmd, shell=True)


# %%
if __name__ == "__main__":
    clusters_root_dir = Path("./RESULTS/07_FEATURES/CLUSTER_DATA")
    kmers_dir = mkdir(Path("./RESULTS/08_BLAST"))
    cluster_file_list = list(clusters_root_dir.glob("**/*.cluster"))
    possible_kmers = []
    for cluster in tqdm(cluster_file_list):
        data = read_cluster_file(cluster)
        positions_df = data["POSITION_DATA"]
        kmers = epitope_kmers(positions_df,
                              entropy_thres=0,
                              epi_length=8,
                              gapped=True,
                              )
        possible_kmers.extend(kmers)
    possible_kmers = set(possible_kmers)

    kmers_fasta = kmers_dir / "kmers.fasta"
    with open(kmers_fasta, "w+") as f:
        for kmer in possible_kmers:
            f.write(f">{kmer}\n{kmer}\n")

    taxids = [9606,  # Human
              9986,  # Rabbit
              9796,  # Horse
              9793,  # Donkey
              9838,  # Dromedary
              9844,  # Llama
              9940,  # Sheep
              10090,  # Mouse
              9031,  # Chicken
              ]
    cmds = []
    for taxid in taxids:
        cmd = blastp_short_cmd(kmers_fasta, taxid)
        cmds.append(cmd)

    processes = []
    for c in cmds:
        p = multiprocessing.Process(target=execute_cmd, args=(c,))
        p.start()
        processes.append(p)
