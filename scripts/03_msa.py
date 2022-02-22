# -*- coding: utf-8 -*-
"""
Created on Wed Oct 6 2021

Aligns all fastas inside a directory using MUSCLE


@author: Albert Ros-Lucas
"""
from Bio import AlignIO
from Bio.Align.Applications import MuscleCommandline
from Bio.Application import ApplicationError
from pathlib import Path
from tqdm import tqdm
import multiprocessing
import itertools
import shutil
import argparse


def init_argparser():
    input_parser = argparse.ArgumentParser()
    input_parser.add_argument('-m',
                              '--muscle',
                              required=True
                              )
    input_parser.add_argument('-d',
                              '--dir',
                              required=True
                              )
    return input_parser


def mkdir(path_str):
    path_obj = Path(path_str)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def run_muscle(infile, outdir):
    global muscle_exe
    outfile = Path(outdir) / f"MSA_{Path(infile).name}"
    muscle_cline = MuscleCommandline(muscle_exe,
                                     input=str(infile),
                                     out=str(outfile),
                                     )
    try:
        muscle_cline()
    except ApplicationError as e:
        print(f"{infile} was not aligned. Is it a correct FASTA file?")
        print(e)
        pass


def align_dir(fastas_dir, output_dir):
    fastas_dir = Path(fastas_dir)
    outpath = Path(output_dir) / "UNSORTED"
    outpath.mkdir(parents=True, exist_ok=True)
    fastas_list = [str(x) for x in fastas_dir.glob("**/*")
                   if x.suffix == ".fasta"
                   ]
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(run_muscle,
                     zip(fastas_list, itertools.repeat(outpath))
                     )
    return outpath


def keyMSA(item):
    # We need a custom function to sort the MSA by priority
    # and length of alignment
    returnList = []
    # (1 - x) because descending order
    returnList.append(1 - int(item.id.split("_")[0]))
    returnList.append(len(item) - item.seq.count("-"))
    return tuple(returnList)


def sort_msa(msafile):
    alignment = AlignIO.read(msafile, "fasta")
    alignment.sort(key=keyMSA, reverse=True)
    return alignment


def sort_msa_dir(unsorted_folder):
    unsorted_folder = Path(unsorted_folder)
    musclefiles = [x for x in unsorted_folder.glob("**/*") if x.is_file()]
    sorted_folder = Path(unsorted_folder).parent
    for alignment in tqdm(musclefiles):
        sorted_aln = sort_msa(alignment)
        out_aln_file = sorted_folder / f"{Path(alignment).stem}_sorted.aln"
        with out_aln_file.open("w+") as outfile:
            AlignIO.write(sorted_aln, outfile, "fasta")
    shutil.rmtree(unsorted_folder)


def main(fastas_root):
    aln_root = mkdir(Path("./RESULTS/03_MSA") / fastas_root.parent.stem)
    print("Working with {fastas_root.parent.stem}")
    for fastas_dir in [x for x in fastas_root.glob('**/*') if x.is_dir()]:
        group_dir = aln_root / fastas_dir.name
        unsorted_dir = align_dir(fastas_dir, group_dir)
        sort_msa_dir(unsorted_dir)


# %%
if __name__ == "__main__":
    parser = init_argparser()
    args = parser.parse_args()
    muscle_exe = args.muscle
    main_root = Path(args.dir)
    group_roots = [x for x in main_root.glob("./*")]
    fastas_roots = [y
                    for x in group_roots
                    for y in x.glob("./FASTAS")
                    ]
    for fastas_root in fastas_roots:
        main(fastas_root)
