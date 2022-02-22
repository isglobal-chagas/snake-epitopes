# -*- coding: utf-8 -*-
"""
Created on Tue May 25 2021

Will launch a BLASTP against the PDB database of the cluster refseqs

@author: arosl
"""

from pathlib import Path
from Bio import AlignIO
from Bio import SeqIO
from tqdm import tqdm
import pandas as pd
from Bio.Blast import NCBIXML
from Bio.Blast import NCBIWWW
import atomium
from collections import defaultdict
from multiprocessing.pool import ThreadPool
import argparse
import ast
from io import StringIO
import time


def init_argparser():
    input_parser = argparse.ArgumentParser()
    input_parser.add_argument('-d',
                              '--dir',
                              required=True
                              )
    input_parser.add_argument('-f',
                              '--fasta',
                              required=True
                              )
    return input_parser


def mkdir(path_obj):
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def fasta_to_dict(fasta_file):
    with open(fasta_file, "r") as f:
        fasta_dict = {record.id: str(record.seq)
                      for record in SeqIO.parse(f, "fasta")
                      }
    return fasta_dict


def ungappedlen(sequence):
    return int(len(sequence) - sequence.count("-"))


def parse_pdb_blast(xml):
    best_hits = {}
    handle = open(xml)
    records = NCBIXML.parse(handle)
    for record in records:
        query_accession = record.query
        hits = []
        if record.alignments:
            for alignment in record.alignments:
                pdb_id, chain = alignment.hit_id.split("|")[-2:]
                pdb_length = alignment.length
                for hsp in alignment.hsps:
                    query_coverage = hsp.query_end - hsp.query_start + 1
                    cover_identity = round(hsp.identities /
                                           query_coverage, 2)
                    total_identity = round(hsp.identities /
                                           record.query_length, 2)
                    query_covered = round(query_coverage /
                                          record.query_length, 2)
                    hits.append((pdb_id,
                                 chain,
                                 query_covered,
                                 cover_identity,
                                 total_identity,
                                 pdb_length,
                                 hsp.expect,
                                 ))
            if hits:
                sorted_hits = sorted(hits,
                                     key=lambda x: (x[4], x[3], 1 - x[6]),
                                     reverse=True
                                     )
                best_hits[query_accession] = sorted_hits[0]
    handle.close()
    return best_hits


def get_pdb_data(pdb_record):
    pdb_id, pdb_chain = pdb_record
    pdb_record = atomium.fetch(pdb_id)
    pdb_res = pdb_record.resolution
    if not pdb_res:
        pdb_res = 999
    pdb_length = len(pdb_record.model.chain(pdb_chain).sequence)
    return {(pdb_id, pdb_chain): (pdb_res, pdb_length)}


def choose_best_pdb(pdb_records, refseq_length):
    pdb_results = ThreadPool(len(pdb_records)).imap_unordered(get_pdb_data,
                                                              pdb_records
                                                              )
    data_dict = {}
    for result_dict in pdb_results:
        if result_dict:
            for k, v in result_dict.items():
                data_dict[k] = v
    pdb_data_list = []
    for record in pdb_records:
        pdb_id, pdb_chain = record
        pdb_data = data_dict[record]
        pdb_res = pdb_data[0]
        pdb_length = pdb_data[1]
        coverage = round(pdb_length / refseq_length, 2)
        if pdb_res:
            pdb_data_list.append((pdb_id, pdb_chain, pdb_res, coverage))
    return_list = sorted(pdb_data_list,
                         key=lambda x: (1/x[3], x[2], x[0], x[1])
                         )
    best = return_list[0]
    if best[2] == 999:
        new_best = (best[0],
                    best[1],
                    "NA",
                    best[3]
                    )
        return new_best
    else:
        return best


def read_cluster_file(cluster_file):
    parsed_data = {}
    with open(cluster_file) as f:
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


def aln_pdb_records(aln):
    pdb_records = []
    for record in aln:
        prefix = record.id.split("_")[0]
        if prefix == "01":
            pdb_id, chain = record.id.split("_")[1].split("|")
            pdb_records.append((pdb_id, chain))
    return pdb_records


def web_blastp(input_fasta, database="nr", entrez_query="(none)"):
    print("BLASTING NCBI...")
    input_fasta = Path(input_fasta)
    fasta = open(input_fasta)
    try:
        result_handle = NCBIWWW.qblast("blastp",
                                       database,
                                       fasta.read(),
                                       entrez_query=entrez_query
                                       )
    except Exception as e:
        print(e)
        print("Something happened, waiting and retrying...")
        time.sleep(60)
        result_handle = NCBIWWW.qblast("blastp",
                                       database,
                                       fasta.read(),
                                       entrez_query=entrez_query
                                       )
    output = input_fasta.parent / f"{input_fasta.stem}_{database}_BLAST.xml"
    with open(output, "w+") as f:
        blast_results = result_handle.read()
        f.write(blast_results)
    fasta.close()
    return output


def main():
    pass


# %%
if __name__ == "__main__":
    parser = init_argparser()
    args = parser.parse_args()
    raw_fasta = Path(args.fasta)
    root_consensus_dir = Path(args.dir)
    root_pdb_dir = mkdir(Path("./RESULTS/05_PDBS"))

    # We need the raw sequences for the BLAST, and to get the length of the
    # reference sequences.
    raw_seqs_dict = fasta_to_dict(raw_fasta)
    lenghts_dict = {x: len(y) for x, y in raw_seqs_dict.items()}

    rows = []
    for group in [x for x in root_consensus_dir.glob("./*") if x.is_dir()]:
        cluster_files = list(group.glob("**/*.cluster"))
        for cluster_file in tqdm(cluster_files):
            cluster_data = read_cluster_file(cluster_file)
            cutoff, cluster_name = cluster_data["CLUSTER_NAME"].split("_")
            refseq_len = ungappedlen(cluster_data["REFSEQ_SEQ"])
            alignment = cluster_data["ALIGNMENT"]
            pdb_records = aln_pdb_records(alignment)
            pdb_ref = ""
            if cluster_data["REFSEQ_ID"].startswith("01_"):
                pdb_id = cluster_data["REFSEQ_ID"].split("_")[1]
                pdb_ref = pdb_id.replace("|", "_")
            elif pdb_records:
                best_pdb = choose_best_pdb(pdb_records, refseq_len)
                pdb_ref = f"{best_pdb[0]}_{best_pdb[1]}"
            row = [group.name,
                   cutoff,
                   cluster_name,
                   cluster_file,
                   cluster_data["REFSEQ_ID"],
                   refseq_len,
                   pdb_ref
                   ]
            rows.append(tuple(row))
    df = pd.DataFrame(rows, columns=["group_name",
                                     "cdhit_cutoff",
                                     "cluster_name",
                                     "cluster_path",
                                     "refseq_id",
                                     "refseq_length",
                                     "pdb_ref"
                                     ]
                      )

    # Get all the reference sequences
    all_refseqs = set(df["refseq_id"])
    with open(root_pdb_dir / "reference_sequences.fasta", "w+") as f:
        for refseq in all_refseqs:
            f.write(f">{refseq}\n")
            f.write(f"{raw_seqs_dict[refseq]}\n")

    blast_results = web_blastp(root_pdb_dir / "reference_sequences.fasta",
                               database="pdb",
                               entrez_query="txid8570[ORGN]")
    blast_dict = parse_pdb_blast(blast_results)
    df["BLAST_RAW"] = df["refseq_id"].map(blast_dict)
    df[["pdb_id",
        "pdb_chain",
        "pdb_query_covered",
        "pdb_cover_identity",
        "pdb_total_identity",
        "pdb_length",
        "pdb_evalue"
        ]] = pd.DataFrame(df['BLAST_RAW'].tolist(), index=df.index)
    df = df.drop(columns=["BLAST_RAW"])
    df.to_csv(root_pdb_dir / "PDB_data.csv")

    # Finally, we will fetch all PDBs and save them
    pdbs_to_download = defaultdict(set)
    for x in df["pdb_ref"]:
        if x:
            pdbid, chain = x.split("_")
            pdbs_to_download[pdbid].add(chain)
    for ix in df.index:

        if df["pdb_total_identity"].iloc[ix] >= 0.5:
            pdbid = df["pdb_id"].iloc[ix]
            chain = df["pdb_chain"].iloc[ix]
            pdbs_to_download[pdbid].add(chain)

    pdb_download_folder = mkdir(root_pdb_dir / "PDBs")
    for pdb_id, chains in tqdm(pdbs_to_download.items()):
        for c in chains:
            pdb = atomium.fetch(f"{pdb_id}.pdb")
            model = pdb.model.chain(c)
            residues = []
            counter = 1
            for residue in model.residues():
                heteroatom = False
                for atom in residue.atoms():
                    if atom._is_hetatm:
                        heteroatom = True
                if not heteroatom:
                    new_resi = residue.copy(id=f"{c}.{counter}")
                    residues.append(new_resi)
                    counter += 1
            new_chain = atomium.structures.Chain(*residues, id=c)
            savepath = str(pdb_download_folder / f"{pdb_id}_{c}.pdb")
            new_chain.save(savepath)
