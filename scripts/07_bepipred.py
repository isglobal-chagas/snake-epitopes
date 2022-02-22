# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 10:35:25 2021

Runs Bepipred2 against the refseqs of the clusters

Bepipred2 needs specific packages in order to work. It's highly reccommended
to use an environment just for this. Some of them are better installed via pip.

For this script, this was used to create the environment:
conda create -p ~/envs/bcell_predictions
source activate ~/envs/bcell_predictions
conda install python=3.6 pip=20.3.3 tqdm
pip install scipy==1.2.3 --no-cache-dir
pip install numpy==1.16.6 --no-cache-dir
pip install matplotlib==2.0.0 --no-cache-dir
pip install scikit-learn==0.17 --no-cache-dir

@author: arosl
"""
from pathlib import Path
import subprocess
import argparse


def init_argparser():
    input_parser = argparse.ArgumentParser()
    input_parser.add_argument('-f',
                              '--fasta',
                              required=True
                              )
    input_parser.add_argument('-b',
                              '--bcell',
                              required=True
                              )
    return input_parser


def mkdir(path_obj):
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def read_fasta(path):
    records = {}
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                idnt = line.strip()[1:]
            else:
                seq = line.strip()
                records[idnt] = seq
    return records


def run_bepipred(bcell_script, seqs_file, output_dir):
    output = Path(output_dir / "bepipred_results.txt")
    args = ['python',
            str(bcell_script),
            '-m Bepipred2',
            '-f',
            f'"{str(seqs_file)}"',
            '>',
            f'"{str(output)}"'
            ]
    script = " ".join(args)
    subprocess.run(script, shell=True)
    return output


# %%
if __name__ == "__main__":
    parser = init_argparser()
    args = parser.parse_args()
    bcell_script = args.bcell
    refseqs_fasta = Path(args.fasta)
    refseq_records = read_fasta(refseqs_fasta)
    output_bepipred = mkdir(Path("./RESULTS/07_FEATURES/BEPIPRED"))
    input_bepipred = output_bepipred / "bepipred_input.fasta"
    with open(input_bepipred, "w+") as f:
        for seq_id, seq in refseq_records.items():
            # Bepipred does not accept X residues.
            # They will be aligned later
            correct_seq = str(seq).replace("X", "")
            f.write(f">{seq_id}\n")
            f.write(f"{correct_seq}\n")
    bepipred_output = run_bepipred(bcell_script,
                                   input_bepipred,
                                   output_bepipred
                                   )
