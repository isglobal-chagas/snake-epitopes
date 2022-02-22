# -*- coding: utf-8 -*-
"""
Created on Wed May 26 2021

Generates protein models with MODELLER using the PDB templates and
the cluster refseqs. It will take some time, depending on the
number of models to generate.

The parameters used in MODELLER are very thorough.

For more info:
https://salilab.org/modeller/manual/node19.html

@author: arosl
"""

from pathlib import Path
import pandas as pd
from Bio import SeqIO
import modeller as modeller
from modeller.automodel import DOPEHRLoopModel, AutoModel, assess
from modeller.scripts import complete_pdb
from io import StringIO
from tqdm import tqdm
import shutil
import argparse


def mkdir(path_obj):
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def init_argparser():
    input_parser = argparse.ArgumentParser()
    input_parser.add_argument('-d',
                              '--dir',
                              required=True
                              )
    input_parser.add_argument('-c',
                              '--csv',
                              required=True
                              )
    input_parser.add_argument('-f',
                              '--fasta',
                              required=True
                              )
    return input_parser


def fasta_to_dict(fasta_file):
    with open(fasta_file, "r") as f:
        fasta_dict = {record.id: str(record.seq)
                      for record in SeqIO.parse(f, "fasta")
                      }
    return fasta_dict


def pir_record(fasta_id, fasta_seq):
    lines = []
    lines.append(f">P1;{fasta_id}")
    lines.append(f"sequence:{fasta_id}::::::::")
    lines.append(str(fasta_seq) + "*\n")
    pir = "\n".join(lines)
    return pir


def get_pir_id(pir):
    if type(pir) == StringIO:
        firstline = pir.readline().strip()
        pir_id = firstline.split(";", 1)[1]
        return str(pir_id)
    elif type(pir) == str:
        firstline = pir.splitlines()[0]
        pir_id = firstline.split(";", 1)[1]
        return pir_id
    else:
        with open(pir) as p:
            firstline = p.readline().strip()
        pir_id = firstline.split(";", 1)[1]
        return str(pir_id)


def make_alignment(pdb_file, pdb_chain, pir_seq, savedir):
    seq_id = get_pir_id(pir_seq)
    pdb_file = Path(pdb_file)
    savedir = Path(savedir)

    global env
    env.libs.topology.read(file='$(LIB)/top_heav.lib')  # read topology
    env.libs.parameters.read(file='$(LIB)/par.lib')  # read parameters
    env.io.hetatm = False
    fake_pir = StringIO(pir_seq)

    aln = modeller.Alignment(env)
    mdl = complete_pdb(env,
                       str(pdb_file),
                       transfer_res_num=True,
                       model_segment=(f'FIRST:{pdb_chain}',
                                      f'LAST:{pdb_chain}'
                                      )
                       )
    aln.append_model(mdl,
                     align_codes=pdb_file.stem,
                     )
    aln.append(file=fake_pir,
               )
    aln.salign()

    # This cuts the overhang
    aln.edit(edit_align_codes=seq_id,
             base_align_codes=pdb_file.stem,
             min_base_entries=1,
             overhang=3
             )

    align_stem = f"{savedir.name}_{pdb_file.stem}"
    aln.write(file=str(savedir / f"{align_stem}.ali"),
              alignment_format='PIR'
              )
    aln.write(file=str(savedir / f"{align_stem}.pap"),
              alignment_format='PAP',
              alignment_features='INDICES HELIX BETA'
              )

    env.libs.topology.clear()
    env.libs.parameters.clear()

    return savedir / f"{align_stem}.ali"


def get_ali_ids(ali_file):
    with open(ali_file) as a:
        first_flag = True
        for line in a:
            if line.startswith(">"):
                if first_flag:
                    pdb_id = line.strip().split(";", 1)[1]
                    first_flag = False
                else:
                    target_id = line.strip().split(";", 1)[1]
                    break
    return pdb_id, target_id


def model_with_loops(alignment_file):
    global env

    env.libs.topology.clear()
    env.libs.parameters.clear()

    pdb_id, target_id = get_ali_ids(alignment_file)
    a = DOPEHRLoopModel(env,
                        alnfile=str(alignment_file),
                        knowns=pdb_id,
                        sequence=target_id,
                        assess_methods=(assess.DOPEHR,
                                        assess.normalized_dope,
                                        assess.GA341
                                        ),
                        loop_assess_methods=(assess.DOPEHR,
                                             assess.normalized_dope,
                                             assess.GA341
                                             )
                        )
    a.starting_model = 1
    a.ending_model = 5

    # Give less weight to all soft-sphere restraints:
    env.schedule_scale = modeller.physical.Values(default=1.0, soft_sphere=0.7)

    # Very thorough VTFM optimization:
    a.library_schedule = modeller.automodel.autosched.slow
    a.max_var_iterations = 300

    # Thorough MD optimization:
    a.md_level = modeller.automodel.refine.slow

    # Repeat the whole cycle 2 times and do not stop unless obj.func. > 1E6
    a.repeat_optimization = 2
    a.max_molpdf = 1e6

    a.loop.starting_model = 1
    a.loop.ending_model = 4
    a.loop.md_level = modeller.automodel.refine.slow

    a.make()

    return a


def model_without_loops(alignment_file):
    global env

    env.libs.topology.clear()
    env.libs.parameters.clear()

    pdb_id, target_id = get_ali_ids(alignment_file)
    a = AutoModel(env,
                  alnfile=str(alignment_file),
                  knowns=pdb_id,
                  sequence=target_id,
                  assess_methods=(assess.DOPEHR,
                                  assess.normalized_dope,
                                  assess.GA341
                                  ),
                  )
    a.starting_model = 1
    a.ending_model = 5

    env.schedule_scale = modeller.physical.Values(default=1.0, soft_sphere=0.7)

    # Very thorough VTFM optimization:
    a.library_schedule = modeller.automodel.autosched.slow
    a.max_var_iterations = 300

    # Thorough MD optimization:
    a.md_level = modeller.automodel.refine.slow

    # Repeat the whole cycle 2 times and do not stop unless obj.func. > 1E6
    a.repeat_optimization = 2
    a.max_molpdf = 1e6

    a.make()

    return a


def build_models(alignment_file):
    pdb_id, target_id = get_ali_ids(alignment_file)
    try:
        a = model_with_loops(alignment_file)
        # For each refined loop model, we will select the best one
        best_models = []
        for i in range(a.starting_model, a.ending_model):
            loop_models = []
            for x in a.loop.outputs:
                if x['failure'] is None:
                    if x["GA341 score"][0] > 0.9:
                        if x["num"] == i:
                            loop_models.append(x)
            key = 'DOPE-HR score'
            loop_models.sort(key=lambda a: a[key])
            if loop_models:
                top_model = loop_models[0]
                best_models.append(top_model)
        if best_models:
            best_models.sort(key=lambda a: a["Normalized DOPE score"])
            top_model = best_models[0]["name"]

            for x in Path().cwd().glob(f"{target_id}*"):
                if top_model not in str(x):
                    Path(x).unlink()
                else:
                    return_file = Path(x)
            return return_file
        else:
            print(f"\n\n\n\nWARNING, NO GOOD MODELS FOR {target_id}\n\n\n\n")
            for x in Path().cwd().glob(f"{target_id}*"):
                Path(x).unlink()
            return None

    except modeller.ModellerError:
        a = model_without_loops(alignment_file)

        for i in range(a.starting_model, a.ending_model):
            models = []
            for x in a.outputs:
                if x['failure'] is None:
                    if x["GA341 score"][0] > 0.9:
                        models.append(x)
            key = 'DOPE-HR score'
            models.sort(key=lambda a: a[key])
            if models:
                top_model = models[0]["name"]
                for x in Path().cwd().glob(f"{target_id}*"):
                    if top_model not in str(x):
                        Path(x).unlink()
                    else:
                        return_file = Path(x)
                return return_file
            else:
                print(f"\n\n\nWARNING, NO GOOD MODELS FOR {target_id}\n\n\n")
                for x in Path().cwd().glob(f"{target_id}*"):
                    Path(x).unlink()
                return None


# %%

if __name__ == "__main__":
    parser = init_argparser()
    args = parser.parse_args()
    master_csv = Path(args.csv)
    raw_fasta = Path(args.fasta)
    pdb_dir = Path(args.dir)
    root_models_dir = mkdir(Path("./RESULTS/06_MODELS"))
    seq_dict = fasta_to_dict(raw_fasta)

    df = pd.read_csv(master_csv, index_col=0)
    seqs_to_model = set()
    for ix in df.index:
        if 0.9 > df["pdb_total_identity"].iloc[ix] >= 0.5:
            seq_id = df["refseq_id"].iloc[ix]
            seq = seq_dict[seq_id].replace("X", "")
            pdb = df["pdb_id"].iloc[ix]
            pdb_chain = df["pdb_chain"].iloc[ix]
            pdb_id = f"{pdb}_{pdb_chain}"
            seqs_to_model.add((seq_id, seq, pdb_id))
    counter = 1
    total = len(seqs_to_model)
    for record in tqdm(seqs_to_model):
        env = modeller.Environ()
        env.io.atom_files_directory = [str(pdb_dir)]
        refseq_id = record[0]
        refseq_seq = record[1]
        pdb_file = pdb_dir / f"{record[2]}.pdb"
        pdb_id, pdb_chain = record[2].split("_")
        pir_seq = pir_record(record[0], record[1])
        model_dir = mkdir(root_models_dir / refseq_id)
        alignment = make_alignment(pdb_file, pdb_chain, pir_seq, model_dir)
        output_file = model_dir / f"{alignment.stem}.pdb"

        # Check if file exists (in case script was interrupted)
        if output_file.is_file():
            print("Skipped sequence, model already exists")
            counter += 1
            continue
        else:
            best_model = build_models(alignment)
            if best_model:
                best_model.replace(model_dir / f"{alignment.stem}.pdb")
            else:
                shutil.rmtree(model_dir)
            print(f"\n\n\n\nFINISHED MODEL {counter}/{total}\n\n\n\n")
            counter += 1
