# -*- coding: utf-8 -*-
"""
Created on Wed Oct 6 2021

First part involves launching CD-HIT and generating clusters from the seqs.
It will do it by hierarchical clustering using different identity thresholds
and finally merging the result. Needs the clstr_rev.pl script which can be
found in the CD-HIT site.

Second part is filtering the resulting clusters

@author: Albert Ros-Lucas
"""
from pathlib import Path
import subprocess
from collections import defaultdict
from Bio import SeqIO
from Bio import Entrez
import pandas as pd
import argparse
import numpy as np


def init_argparser():
    input_parser = argparse.ArgumentParser()
    input_parser.add_argument('-c',
                              '--clstr_rev',
                              required=True
                              )
    # Entrez tools require email
    input_parser.add_argument('-e',
                              '--email',
                              required=True
                              )
    input_parser.add_argument('-fd',
                              '--fasta_dir',
                              required=True
                              )
    return input_parser


def mkdir(path_str):
    path_obj = Path(path_str)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def cdhit_cline(input_file,
                output_file,
                identity=0.9,
                length_cutoff=0,
                word_size=5
                ):
    arguments = ["cd-hit",
                 f'-i "{input_file}"',  # input file in fasta
                 f'-o "{output_file}"',  # output file
                 f"-c {identity}",  # sequence identity threshold
                 f"-s {length_cutoff}",  # length difference cutoff
                 "-d 0",  # full description until first space
                 "-sc 1",  # sort clusters by size
                 "-M 0",  # unlimitted memory
                 "-T 0",  # all CPUs
                 "-g 1",  # precise mode
                 f"-n {word_size}"  # word length
                 ]
    cline = " ".join(arguments)
    return cline


def hierarchical_clustering_clines(input_fasta,
                                   output_dir,
                                   clstr_rev_script,
                                   *args
                                   ):
    """
    This will produce all commands needed for a hierarchical clustering
    using the *args given, which need to be fractions of 1.

    More info on hierarchical clustering here:
    https://github.com/weizhongli/cdhit/wiki/3.-User's-Guide#hierarchical-clustering

    Args:
        input_fasta (Path/string): Fasta with sequences
        output_dir (Path/string): Dir where to save the output
        clstr_rev_script (Path/string): Path to the clstr_rev.pl script

    Returns:
        List: One list with all commands needed for hierarchical clustering
        and another one with all the intermediate files that need to be deleted
    """

    fasta_name = Path(input_fasta).stem.replace(" ", "")
    output_dir = Path(output_dir)
    cmds_list = []
    files_to_delete = []
    cdhit_clstrs = []
    current_input = input_fasta

    # Generate a list with all the commands for the clustering
    for arg in args:
        if not type(arg) == float:
            raise Exception("Arguments provided need to be fractions of 1!")

        # String name for the cutoff, for file
        str_cutoff = str(arg).replace('.', '')
        if len(str_cutoff) == 2:
            str_cutoff = str_cutoff + "0"
        output_name = f"{fasta_name}_{str_cutoff}"

        current_output = input_fasta.parent / output_name
        current_output_clstr = input_fasta.parent / f"{output_name}.clstr"
        cdhit_clstrs.append((current_output_clstr, str_cutoff))
        cline = cdhit_cline(current_input,
                            output_file=current_output,
                            identity=arg,
                            word_size=2
                            )
        if not current_input == input_fasta:
            files_to_delete.append(current_input)
        cmds_list.append(cline)
        current_input = current_output
    files_to_delete.append(current_input)

    # First file does not need hierarchic clustering.
    first_file_output = output_dir / cdhit_clstrs[0][0].name
    cmds_list.append(f"cp {cdhit_clstrs[0][0]} {first_file_output}")
    files_to_delete.append(cdhit_clstrs[0][0])

    # Now merge the rest
    current_input = cdhit_clstrs[0][0]
    for clstr_file, cutoff in cdhit_clstrs[1:]:
        files_to_delete.append(clstr_file)
        cluster_file_1 = current_input
        cluster_file_2 = clstr_file
        merged_output_name = output_dir / f"{fasta_name}_{cutoff}.clstr"

        merged_args = [clstr_rev_script,
                       str(cluster_file_1),
                       str(cluster_file_2),
                       ">",
                       str(merged_output_name)
                       ]
        merged_cline = " ".join(merged_args)
        cmds_list.append(merged_cline)
        current_input = merged_output_name
    return cmds_list, files_to_delete


def fasta_to_dict(fasta_file):
    with open(fasta_file, "r") as f:
        fasta_dict = {record.id: str(record.seq)
                      for record in SeqIO.parse(f, "fasta")
                      }
    return fasta_dict


def parse_cluster(clstr_file, raw_fasta):
    """
    Parse the output of CD-HIT together with the input fasta
    to return a dictionary with cluster id and sequences in it

    Args:
        clstr_file (Path/string): Path to cluster output
        raw_fasta (Path/string): Path to fasta used in clustering

    Returns:
        Dict: Dictionary with clusters and sequences in them
    """

    clstr_file = Path(clstr_file)
    input_dict = fasta_to_dict(raw_fasta)
    returndict = defaultdict(list)

    rawdict = {}
    with open(clstr_file) as f:
        for line in f:
            if line.startswith(">"):
                clusterID = line.strip()[1:]
                rawdict[clusterID] = []
            else:
                rawdict[clusterID].append(line.strip())

    for cluster, values in rawdict.items():
        for protein in values:
            fullprot = protein[protein.find(">") + 1: protein.find("...")]
            prot_seq = input_dict[fullprot]
            protein_origin, protein_id = fullprot.split("_", 1)
            protein_id, species = protein_id.rsplit("_", 1)
            returndict[cluster].append((protein_origin,
                                        protein_id,
                                        species,
                                        prot_seq
                                        )
                                       )
    print(f"Total number of clusters: {str(len(returndict))}")
    return returndict


def get_clstr_stats(parsed_clstr):
    total_clusters = len(parsed_clstr)
    depths = np.array([len(x) for x in parsed_clstr.values()])
    mean = round(depths.mean(), 2)
    dev = round(depths.std(), 2)
    median = np.median(depths)
    min_depth = depths.min()
    max_depth = depths.max()
    description = {"total_clusters": total_clusters,
                   "mean": mean,
                   "median": median,
                   "standard_deviation": dev,
                   "min_depth": min_depth,
                   "max_depth": max_depth
                   }
    return description


def get_protein_names(*args):
    identificators = [arg.rsplit("_", 1)[0] for arg in args]
    with Entrez.efetch(db="protein",
                       id=identificators,
                       rettype="gp",
                       retmode="text"
                       ) as handle:
        records = list(SeqIO.parse(handle, "gb"))
        return_names = {}
        for counter, record in enumerate(records):
            protein_name = record.description
            if not record.id.split(".")[0] in args[counter]:
                print("Careful! IDs do not entirely match.")
                print("Just be sure that the keys somewhat match:")
                print(f"Original key: {args[counter]}")
                print(f"Parsed key: {record.id}\n\n")
            if "RecName: Full=" not in protein_name:
                return_names[args[counter]] = protein_name
            else:
                splitted = protein_name.split(";")[0]
                new_name = splitted.split("RecName: Full=")[1]
                return_names[args[counter]] = new_name
    return return_names


def filter_clusters(parsed_clstr, protein_names):
    accepted_dict = {}
    discarded_dict = {}
    for cluster, protein_list in list(parsed_clstr.items()):
        indexes = set()
        for protein_data in protein_list:
            indexes.add(protein_data[0])  # Prefix that determines the origin
        """
        First filter: UniProt proteins are reviewed, and thus are more
        trustable. Some UniProt proteins won't cluster, but we still keep
        them. Clusters with just one member from other databases will be
        discarded.
        """
        if len(protein_list) == 1:
            if "02" not in indexes:
                discarded_dict[cluster] = protein_list
                continue

        """
        Second filter: clusters without UniProt reviews members will be
        discarded at first. They might still make it if they pass a second
        repesca of keywords.
        """
        if "02" not in indexes:
            discarded_dict[cluster] = protein_list
            continue

        """
        Otherwise, if a cluster has >1 member and contains a UniProt protein,
        we will keep it as good.
        """
        accepted_dict[cluster] = protein_list

    # We will go through each discarded protein to see if it matches any
    # keyword related to venom
    accept_list = ["venom",
                   "metalloprotease",
                   "metalloproteinase",
                   "svmp",
                   "toxin",
                   "hyaluronidase",
                   "protease",
                   "jerdostatin",
                   "disintegrin",
                   "kunitz",
                   "3ftx",
                   "phospholipase",
                   "pla",
                   "c-type lectin",
                   "vascular endothelial growth factor",
                   "vegf",
                   "cysteine-rich",
                   "crisp",
                   "oxidase",
                   "snaclec"
                   ]
    reject_list = ["periplakin",
                   "cytochrome",
                   "nadh",
                   "neurotrophin",
                   "prolactin",
                   "recombination",
                   "oocyte",
                   "cmos",
                   "opsin",
                   "ubinuclein",
                   "homeobox"
                   ]
    for cluster, protein_list in list(discarded_dict.items()):
        accepted = []
        rejected = []

        if not len(protein_list) >= 2:
            continue

        for protein in protein_list:
            protein_id = protein[1]
            description = protein_names[protein_id].lower()
            if any(x in description for x in reject_list):
                rejected.append(protein)
                continue
            elif any(x in description for x in accept_list):
                accepted.append(protein)
                continue
            else:
                rejected.append(protein)

        if len(rejected) == 0 and len(accepted) >= 1:  # All accepted
            accepted_dict[cluster] = protein_list
            del discarded_dict[cluster]
            continue
        elif len(rejected) >= 1 and len(accepted) == 0:  # All rejected
            continue

        else:  # Some in accepted, some in rejected
            print(f"WARNING for {cluster}")
            print("Some proteins matched keyword and others don't.")
            print("It will be accepted as a cluster, but please revise it.\n")
            accepted_dict[cluster] = protein_list
            del discarded_dict[cluster]
    print(f"Filtered clusters: {len(accepted_dict)}")
    return accepted_dict, discarded_dict


def build_fastas(clstr_dictionary, output_dir):
    for cluster, protein_list in clstr_dictionary.items():
        output_fasta = Path(output_dir) / f"{cluster.replace(' ', '')}.fasta"
        with open(output_fasta, "w+") as f:
            for protein_data in protein_list:
                full_id = "_".join(protein_data[:-1])
                seq = protein_data[-1]
                f.write(f">{full_id}\n")
                f.write(f"{seq}\n")


def clusters_summary(cluster_dict, protein_names):
    cluster_list = []
    for cluster, protein_list in cluster_dict.items():
        for protein_data in protein_list:
            name = protein_names[protein_data[1]]
            full_id = "_".join(protein_data[:-1])
            cluster_list.append((cluster, full_id, name))
    df = pd.DataFrame(cluster_list, columns=["cluster",
                                             "protein_id",
                                             "protein_name"
                                             ]
                      )
    return df


def main(fasta_file):
    fasta_file = Path(fasta_file)
    root_output_dir = mkdir(Path("./RESULTS/02_CLUSTERS/") / fasta_file.stem)
    clusters_output_dir = mkdir(root_output_dir / "CLUSTERS")
    fastas_output_dir = mkdir(root_output_dir / "FASTAS")

    # Will make a hierarchical clustering from 95% to 50%.
    cutoffs = list(reversed(np.arange(0.50, 1.00, 0.05).round(2)))
    cutoffs = [float(x) for x in cutoffs]
    cmds, delete_list = hierarchical_clustering_clines(fasta_file,
                                                       clusters_output_dir,
                                                       clstr_rev_script,
                                                       *cutoffs
                                                       )
    for cmd in cmds:
        subprocess.run(cmd, shell=True)
    for f in delete_list:
        f.unlink()

    # Get protein names
    ids = [record.id.split("_")[1]
           for record in SeqIO.parse(fasta_file, "fasta")
           ]
    protein_names = get_protein_names(*ids)

    cluster_stats = {}
    for clstr in clusters_output_dir.glob("./*.clstr"):
        print(f"-------------{'-'*len(clstr.stem)}")
        print(f"-------------{'-'*len(clstr.stem)}")
        print(f"Working with {clstr.stem}\n")
        cluster_cutoff = clstr.stem.rsplit("_", 1)[1]
        cutoff_output_dir = mkdir(fastas_output_dir / cluster_cutoff)

        parsed_clstr = parse_cluster(clstr, fasta_file)
        accepted, rejected = filter_clusters(parsed_clstr, protein_names)
        cutoff_stats = get_clstr_stats(accepted)
        cluster_stats[cluster_cutoff] = cutoff_stats

        build_fastas(accepted, cutoff_output_dir)

        # Will create csv for all accepted and rejected proteins
        # Useful to check if we have discarded something by mistake
        accepted_summary = clusters_summary(accepted, protein_names)
        accepted_summary.to_csv(clusters_output_dir /
                                f"{cluster_cutoff}_accepted.csv")
        rejected_summary = clusters_summary(rejected, protein_names)
        rejected_summary.to_csv(clusters_output_dir /
                                f"{cluster_cutoff}_rejected.csv")
        print("\ndone!\n")

    stats_list = []
    for cutoff, stats in cluster_stats.items():
        cutoff_l = [cutoff]
        cutoff_l.append(stats["total_clusters"])
        cutoff_l.append(stats["min_depth"])
        cutoff_l.append(stats["max_depth"])
        cutoff_l.append(stats["mean"])
        cutoff_l.append(stats["median"])
        cutoff_l.append(stats["standard_deviation"])
        stats_list.append(tuple(cutoff_l))
    df = pd.DataFrame(stats_list, columns=["cutoff",
                                           "total_clusters",
                                           "min_depth",
                                           "max_depth",
                                           "mean",
                                           "median",
                                           "standard_deviation"
                                           ]
                      )
    df.to_csv(root_output_dir / "cdhit_stats.csv")


# %%
if __name__ == "__main__":
    parser = init_argparser()
    args = parser.parse_args()
    Entrez.email = args.email
    clstr_rev_script = args.clstr_rev
    fasta_files = Path(args.fasta_dir).glob("./*.fasta")
    for fasta_file in fasta_files:
        main(fasta_file)
