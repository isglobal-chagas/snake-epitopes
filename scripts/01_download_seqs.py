# -*- coding: utf-8 -*-
"""
Created on Wed Oct 6 2021

Reads an existing csv file containing information on medically relevant
snakes. It needs to have a column with the taxids, preferably with "taxid"
as its header.

It then downloads all their available sequences from UniProt (reviewed
venom sequences) and NCBI, and generates a fasta file.

@author: Albert Ros-Lucas
"""
import argparse
from pathlib import Path
import pandas as pd
from Bio import SeqIO
import requests
from tqdm import tqdm
from io import StringIO
from Bio import Entrez
import time


def init_argparser():
    input_parser = argparse.ArgumentParser()
    input_parser.add_argument('-c',
                              '--csv',
                              required=True
                              )
    # Entrez tools require email
    input_parser.add_argument('-e',
                              '--email',
                              required=True
                              )
    return input_parser


def mkdir(path_str):
    path_obj = Path(path_str)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def fetch_Uniprot_seqs(taxid):
    base_url = "http://www.uniprot.org/uniprot/"
    # We want reviewed venom proteins, as these are the best annotated
    # This is part of Animal Toxin Annotation Project
    query_str = ('(keyword:toxin OR annotation:(type:"tissue specificity" '
                 f'venom)) taxonomy:"{taxid}" AND reviewed:yes'
                 )
    payload = {"query": query_str,
               "format": "fasta",
               "include": "yes"
               }
    result = requests.get(base_url, params=payload)
    if result.ok:
        return result.text
    else:
        print(f"Could not download seqs from UniProt for taxid {taxid}")
        return None


def fetch_NCBI_seqs(taxid):
    # Entrez first finds all the accessions for the species
    search_handle = Entrez.esearch(db="protein",
                                   term=f"txid{taxid}[Organism:exp]",
                                   retmax=100000
                                   )
    parsed = Entrez.read(search_handle)
    id_list = [str(x) for x in parsed["IdList"]]
    if id_list:
        # If accessions are found, we download all of them as fasta.
        fetch_handle = Entrez.efetch(db="protein",
                                     id=",".join(id_list),
                                     rettype="fasta",
                                     retmode="text",
                                     retmax=100000
                                     )
        return fetch_handle.read()
    else:
        print(f"Could not download seqs from NCBI for taxid {taxid}")
        return None


def fetch_all_seqs(taxid):
    taxid_seqs = {}
    try:
        uniprot_seqs = fetch_Uniprot_seqs(taxid)
        ncbi_seqs = fetch_NCBI_seqs(taxid)
        taxid_seqs[taxid] = (uniprot_seqs, ncbi_seqs)
        return taxid_seqs
    except Exception as e:
        print(f"Could not download sequences for taxid {taxid}")
        print(e)
        print("Retrying...")
        try:
            uniprot_seqs = fetch_Uniprot_seqs(taxid)
            ncbi_seqs = fetch_NCBI_seqs(taxid)
            taxid_seqs[taxid] = (uniprot_seqs, ncbi_seqs)
            print("ok")
            return taxid_seqs
        except Exception as e:
            print(f"Could not download sequences for taxid {taxid}")
            print(e)


def download_sequences(taxid_list):
    full_taxids_seqs = {}
    for taxid in tqdm(taxid_list):
        seqs = fetch_all_seqs(taxid)
        for k, v in seqs.items():
            full_taxids_seqs[k] = v
    return full_taxids_seqs


def parse_sequences(taxid, taxid_tuple):
    return_dict = {}
    uniprot_seqs = taxid_tuple[0]
    ncbi_seqs = taxid_tuple[1]

    # Uses StringIO to simulate a fasta file
    uniprot_temp = StringIO(uniprot_seqs)
    uniprot_records = list(SeqIO.parse(uniprot_temp, "fasta"))
    for record in uniprot_records:
        uniprot_accession = record.id.split("|")[1]
        new_id = f">02_{uniprot_accession}_{taxid}"
        return_dict[new_id] = record.seq

    ncbi_temp = StringIO(ncbi_seqs)
    ncbi_records = list(SeqIO.parse(ncbi_temp, "fasta"))
    for record in ncbi_records:
        if record.id.startswith("sp"):  # sequences from UniProt
            uniprot_accession = record.id.split("|")[1].split(".")[0]
            dup_flag = False
            for k in return_dict:
                if uniprot_accession in k:
                    dup_flag = True  # we already have this sequence
            if dup_flag:
                continue
            else:
                # this sequence comes from UniProt, but it's probably not
                # reviewed, (maybe not a venom protein)
                new_id = f">03_{uniprot_accession}_{taxid}"
                return_dict[new_id] = record.seq

        elif record.id.startswith("pdb"):  # sequences from PDB
            pdb_accession = record.id[4:]
            new_id = f">01_{pdb_accession}_{taxid}"
            return_dict[new_id] = record.seq

        elif record.id.startswith("prf"):
            prf_accession = record.id.split("||")[1]
            new_id = f">03_{prf_accession}_{taxid}"
            return_dict[new_id] = record.seq

        else:
            new_id = f">03_{record.id.split('.')[0]}_{taxid}"
            return_dict[new_id] = record.seq

    return return_dict


# %%

if __name__ == "__main__":
    parser = init_argparser()
    args = parser.parse_args()
    Entrez.email = args.email
    taxids_csv = Path(args.csv)
    output_dir = mkdir("./RESULTS/01_SEQS")

    # Read the csv and select the column with the taxids
    df = pd.read_csv(taxids_csv)
    taxid_col = ""
    for col in df:
        if "taxid" in col:
            taxid_col = col
    if not taxid_col:
        print("Taxid column could not be found. Please, specify one:\n")
        for col in df:
            print(col)
        taxid_col = input("Write column name: ")
    if not taxid_col:
        raise Exception("Taxid column was not selected.")
    if taxid_col not in df:
        raise Exception("Taxid column selected incorrectly.")

    # We get all taxids, download the raw sequences and tag them
    all_taxids = set(df[taxid_col])

    # Also we will get all taxids from only category 1 snakes
    cat1_taxids = set()
    for ix in df.index:
        if df["category"].iloc[ix] == 1:
            cat1_taxids.add(df[taxid_col].iloc[ix])

    print("Starting sequence download\n")
    time.sleep(1)
    taxid_seqs = download_sequences(all_taxids)

    # All categories
    output_seqs = output_dir / f"{taxids_csv.stem}_all.fasta"
    with open(output_seqs, "w+") as output:
        for taxid, seq_tuple in taxid_seqs.items():
            sequences = parse_sequences(taxid, seq_tuple)
            for prot_id, seq in sequences.items():
                output.write(f"{prot_id}\n{seq}\n")
    # Only cat 1
    output_seqs = output_dir / f"{taxids_csv.stem}_cat1.fasta"
    with open(output_seqs, "w+") as output:
        for taxid, seq_tuple in taxid_seqs.items():
            if taxid in cat1_taxids:
                sequences = parse_sequences(taxid, seq_tuple)
                for prot_id, seq in sequences.items():
                    output.write(f"{prot_id}\n{seq}\n")
    print("\ndone!")
