# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 11:07:23 2021

Describe all the features needed for epitope prediction

It uses NACCESS to predict relative solvent accessibility.
This program is kindly shared by Simon Hubbard
More info:
http://www.bioinf.manchester.ac.uk/naccess/nacdownload.html

@author: arosl
"""

import pandas as pd
import numpy as np
import subprocess
from tqdm import tqdm
import argparse
from io import StringIO
from pathlib import Path
from Bio import SeqIO, AlignIO
from Bio.Align.Applications import MuscleCommandline
import atomium
import tempfile
import os
import ast


def mkdir(path_obj):
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def init_argparser():
    input_parser = argparse.ArgumentParser()
    input_parser.add_argument('-c',
                              '--csv',
                              required=True
                              )
    input_parser.add_argument('-f',
                              '--fasta',
                              required=True
                              )
    input_parser.add_argument('-n',
                              '--naccess',
                              required=True
                              )
    input_parser.add_argument('-br',
                              '--bepipred_results',
                              required=True
                              )
    input_parser.add_argument('-md',
                              '--models_dir',
                              required=True
                              )
    input_parser.add_argument('-pd',
                              '--pdbs_dir',
                              required=True
                              )
    input_parser.add_argument('-m',
                              '--muscle',
                              required=True
                              )
    return input_parser


def parse_bepipred(results_file):
    bepipred_dict = {}
    with open(results_file, "r") as f:
        for line in f:
            if line.startswith("input"):
                seq_flag = False
                current_id = line.strip().split("input: ")[1]
                bepipred_dict[current_id] = {}
            elif line.startswith("Position"):
                seq_flag = True
            else:
                if seq_flag:
                    data = line.strip().split()[:-1]
                    bepipred_dict[current_id][int(data[0])] = tuple(data[1:])
    return bepipred_dict


def fasta_to_dict(fasta_file):
    with open(fasta_file, "r") as f:
        fasta_dict = {record.id: str(record.seq)
                      for record in SeqIO.parse(f, "fasta")
                      }
    return fasta_dict


def hydrophobicity(peptide):
    # Kyte & Doolittle hydrophobicity scale
    hydrophobicity_scale = {'A': 1.8,
                            'R': -4.5,
                            'N': -3.5,
                            'D': -3.5,
                            'C': 2.5,
                            'Q': -3.5,
                            'E': -3.5,
                            'G': -0.4,
                            'H': -3.2,
                            'I': 4.5,
                            'L': 3.8,
                            'K': -3.9,
                            'M': 1.9,
                            'F': 2.8,
                            'P': -1.6,
                            'S': -0.8,
                            'T': -0.7,
                            'W': -0.9,
                            'Y': -1.3,
                            'V': 4.2,
                            'X': 0
                            }
    min_val = min(hydrophobicity_scale.values())
    max_val = max(hydrophobicity_scale.values())

    hydrophobicity_norm = {k: (v - min_val) / (max_val - min_val)
                           for k, v in hydrophobicity_scale.items()
                           }
    try:
        tot_hydroph = [hydrophobicity_norm[aa] for aa in peptide]
        hydroph = round(sum(tot_hydroph) / len(peptide), 3)
        return hydroph
    except KeyError:
        print("Incorrect residue for hydrophobicity scale")
        return None


def protein_hydrophobicity(protein_seq, window=7):
    window_edge = int((window - 1)/2)
    hydroph_dict = {}
    first_position = window_edge
    last_position = int(len(protein_seq) - window_edge - 1)
    for counter, aa in enumerate(protein_seq):
        if first_position <= counter <= last_position:
            first_pept_pos = counter - window_edge
            last_pept_pos = counter + window_edge + 1
            peptide = protein_seq[first_pept_pos:last_pept_pos]
            hydroph_dict[counter] = (aa, hydrophobicity(peptide))
        else:
            hydroph_dict[counter] = (aa, np.nan)
    return hydroph_dict


def run_naccess(pdb_file, naccess_exe):
    pdb_name = Path(pdb_file).stem
    cmd = f"{naccess_exe} {str(pdb_file)}"
    subprocess.run(cmd, shell=True)
    output_files = [x
                    for x in Path.cwd().glob(f"./{pdb_name}*")
                    if x.suffix in [".rsa", ".asa", ".log"]
                    ]
    return output_files


def parse_rsa(rsa_file):
    d_aa = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
            'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
            'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
            'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'
            }
    rsa_values = {}
    with open(rsa_file, "r") as f:
        for line in f:
            if line.startswith("RES"):
                values = line.split()
                try:
                    aa3 = values[1]
                    residue = d_aa[aa3]
                    rsa = float(line[23:29])
                except KeyError:
                    print(f"Careful, unkwnon aminoacid {values[1]}")
                    residue = "X"
                    rsa = np.nan
                chain = line[8]
                position = str(int(line[9:13]))
                rsa_values[(chain, position)] = (residue, rsa)
    return rsa_values


def b_values(pdb_file):
    b_values_dict = {}
    pdb = atomium.open(pdb_file)

    residues = pdb.model.residues()
    bvalues = [a.bvalue for r in residues for a in r.atoms() if a.name == "CA"]
    bvalue_array = np.array(bvalues)
    b_mean = bvalue_array.mean()
    b_stdv = bvalue_array.std(dtype=np.float64)

    for r in residues:
        for a in r.atoms():
            if a.name == "CA":
                if not b_stdv == 0:
                    norm_bval = round((a.bvalue - b_mean) / b_stdv, 2)
                    b_values_dict[tuple(r.id.split("."))] = (r.code, norm_bval)
                else:
                    b_values_dict[tuple(r.id.split("."))] = (r.code, np.nan)
    return b_values_dict


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


def align_values_to_seq(seq, feature_values):
    parsed_values = []
    feature_vals_counter = 0
    for i in range(len(seq)):
        aligned_seq_aa = seq[i]
        if aligned_seq_aa in ["-", "X"]:
            position_vals = [i + 1, aligned_seq_aa, np.nan]
            parsed_values.append(tuple(position_vals))
        else:
            if feature_vals_counter < len(feature_values):
                current_values = feature_values[feature_vals_counter]
                feature_aa = current_values[1]
                if aligned_seq_aa == feature_aa:
                    position_vals = [i + 1, aligned_seq_aa, current_values[2]]
                    parsed_values.append(tuple(position_vals))
                    feature_vals_counter += 1
                else:
                    print(aligned_seq_aa)
                    print(feature_aa)
                    raise Exception("Residues don't match")
            else:
                raise Exception("Residues without values were found")
    return parsed_values


def align_features(muscle_exe, base_alignment, bepi_dict, hydro_dict,
                   rsa_dict=None, bvals_dict=None):
    new_fastas = {}

    bepi_seq = [v[0] for k, v in sorted(bepi_dict.items())]
    new_fastas["BEPI_SEQ"] = ''.join(bepi_seq)
    bepi_values = [(k + 1, v[0], v[1])
                   for k, v in sorted(bepi_dict.items())
                   ]

    hydro_seq = [v[0] for k, v in sorted(hydro_dict.items())]
    new_fastas["HYDRO_SEQ"] = ''.join(hydro_seq)
    hydro_values = [(k + 1, v[0], v[1])
                    for k, v in sorted(hydro_dict.items())
                    ]

    if rsa_dict is not None:
        rsa_seq = []
        rsa_values = []
        for k, v in sorted(rsa_dict.items(), key=lambda x: int(x[0][1])):
            rsa_seq.append(v[0])
            rsa_values.append((k[1], v[0], v[1]))
        new_fastas["RSA_SEQ"] = ''.join(rsa_seq)

    if bvals_dict is not None:
        bvals_seq = []
        bvals_values = []
        for k, v in sorted(bvals_dict.items(), key=lambda x: int(x[0][1])):
            bvals_seq.append(v[0])
            bvals_values.append((k[1], v[0], v[1]))
        new_fastas["BVALS_SEQ"] = ''.join(bvals_seq)

    aln_temporary_file = tempfile.mkstemp(text=True)
    AlignIO.write(base_alignment, aln_temporary_file[1], "fasta")
    for feature, seq in new_fastas.items():
        seq_temporary_file = tempfile.mkstemp(text=True)
        with open(seq_temporary_file[1], "w+") as f:
            f.write(f">{feature}\n{seq}")
        muscle_cline = MuscleCommandline(muscle_exe,
                                         profile=True,
                                         in1=aln_temporary_file[1],
                                         in2=seq_temporary_file[1]
                                         )
        stdout, stderr = muscle_cline()
        temp_output = StringIO(stdout)
        new_alignment = AlignIO.read(temp_output, "fasta")
        AlignIO.write(new_alignment, aln_temporary_file[1], "fasta")
        os.close(seq_temporary_file[0])
        os.remove(seq_temporary_file[1])
    os.close(aln_temporary_file[0])
    os.remove(aln_temporary_file[1])

    aligned_seqs = {}
    for feature, seq in new_fastas.items():
        for s in new_alignment:
            if feature == s.id:
                aligned_seq = s.seq
        aligned_seqs[feature] = str(aligned_seq)

    aligned_values = {}
    aligned_bepi = align_values_to_seq(aligned_seqs["BEPI_SEQ"],
                                       bepi_values)
    aligned_values["BEPI"] = aligned_bepi

    aligned_hydro = align_values_to_seq(aligned_seqs["HYDRO_SEQ"],
                                        hydro_values)
    aligned_values["HYDRO"] = aligned_hydro

    if "RSA_SEQ" in aligned_seqs:
        aligned_rsa = align_values_to_seq(aligned_seqs["RSA_SEQ"],
                                          rsa_values)
        aligned_values["RSA"] = aligned_rsa
    else:
        aligned_values["RSA"] = None

    if "BVALS_SEQ" in aligned_seqs:
        aligned_bvals = align_values_to_seq(aligned_seqs["BVALS_SEQ"],
                                            bvals_values)
        aligned_values["BVALS"] = aligned_bvals
    else:
        aligned_values["BVALS"] = None

    return new_alignment, aligned_values


def align_values_to_df(positions_df, aln_vals):
    rows = positions_df.to_records(index=False)
    position_values = [tuple(x) for x in list(rows)]
    parsed_seq = []
    aln_counter = 0
    for i in range(len(position_values)):
        current_pos_vals = position_values[i]
        original_aa = current_pos_vals[1]

        aln_pos_vals = aln_vals[aln_counter]
        aln_aa = aln_pos_vals[1]

        while not original_aa == aln_aa:
            aln_counter += 1
            aln_pos_vals = aln_vals[aln_counter]
            aln_aa = aln_pos_vals[1]

        features_vals = aln_pos_vals[2:]
        new_position = [x for x in current_pos_vals]
        new_position.extend(features_vals)
        aln_counter += 1
        parsed_seq.append(new_position)

    column_names = [x for x in positions_df]
    column_names.extend(["bepipred",
                         "hydrophobicity",
                         "rsa",
                         "bval"
                         ]
                        )
    column_order = ["position",
                    "residue",
                    "depth",
                    "entropy",
                    "bepipred",
                    "hydrophobicity",
                    "rsa",
                    "bval",
                    "distribution"
                    ]
    df = pd.DataFrame(parsed_seq, columns=column_names)
    df = df[column_order]
    return df


def generate_cluster_file(cluster_data, position_df, output_file):
    with open(output_file, "w+") as f:
        f.write("¡¡¡¡¡CLUSTER_NAME!!!!!\n")
        f.write(f"{cluster_data['CLUSTER_NAME']}\n")

        f.write("¡¡¡¡¡REFSEQ_ID!!!!!\n")
        f.write(f"{cluster_data['REFSEQ_ID']}\n")

        f.write("¡¡¡¡¡REFSEQ_SEQ!!!!!\n")
        f.write(f"{cluster_data['REFSEQ_SEQ']}\n")

        f.write("¡¡¡¡¡MASKED_SEQ!!!!!\n")
        f.write(f"{cluster_data['MASKED_SEQ']}\n")

        alignment = cluster_data["ALIGNMENT"]
        f.write(f"¡¡¡¡¡ALIGNMENT_{len(alignment)}!!!!!\n")
        for a in alignment:
            f.write(f">{a.id}\n{str(a.seq)}\n")

        f.write("¡¡¡¡¡POSITION_DATA!!!!!\n")
        position_data_str = position_df.to_csv(index=False,
                                               line_terminator='\n',
                                               sep=";"
                                               )
        f.write(position_data_str)
        f.write("¡¡¡¡¡END!!!!!")


# %%
if __name__ == "__main__":
    parser = init_argparser()
    args = parser.parse_args()
    master_csv = Path(args.csv)
    refseqs_fasta = Path(args.fasta)
    refseq_records = list(SeqIO.parse(refseqs_fasta, "fasta"))
    refseqs_dict = fasta_to_dict(refseqs_fasta)
    bepipred_output = Path(args.bepipred_results)
    naccess_script = Path(args.naccess)
    models_dir = Path(args.models_dir)
    pdbs_dir = Path(args.pdbs_dir)
    muscle_exe = args.muscle
    df = pd.read_csv(master_csv, index_col=0)
    root_output = mkdir(Path("./RESULTS/07_FEATURES"))

    # We had to run bepipred separately due to package incompatibility
    bepipred_results = parse_bepipred(bepipred_output)

    # Calculating hydrophobicity
    hydrophobicity_results = {}
    for record in refseq_records:
        corrected_seq = str(record.seq).replace("X", "")
        hydroph = protein_hydrophobicity(corrected_seq)
        hydrophobicity_results[record.id] = hydroph

    # List with all protein structures, one for native pdb and the other
    # includes models
    all_structures = []
    native_pdbs = []
    for f in pdbs_dir.glob("./*.pdb"):
        protein_id = f.stem.replace("_", "|")
        all_structures.append((protein_id, f))
        native_pdbs.append((protein_id, f))
    for f in models_dir.rglob("./*.pdb"):
        protein_id = f.parts[2].split("_")[1]
        all_structures.append((protein_id, f))

    # Run NACCESS on structures
    rsa_results = {}
    output_dir = mkdir(root_output / "RSA")
    for pdb in all_structures:
        protein_id = pdb[0]
        pdb_path = pdb[1]
        output_files = run_naccess(pdb_path, naccess_script)
        for f in output_files:
            if f.suffix == ".rsa":
                f.replace(output_dir / f.name)
                rsa_file = output_dir / f.name
            else:
                f.unlink()
        rsa_dict = parse_rsa(rsa_file)
        rsa_results[protein_id] = rsa_dict

    # Calculate flexibility only in native PDBs
    bvals_results = {}
    for pdb in native_pdbs:
        protein_id = pdb[0]
        pdb_file = pdb[1]
        bvals_dict = b_values(str(pdb_file))
        bvals_results[protein_id] = bvals_dict

    # Now put it all together, and add the data to each cluster file
    output_dir = mkdir(root_output / "CLUSTER_DATA")
    for ix in tqdm(df.index):
        cluster_path = Path(df["cluster_path"].iloc[ix])
        group_path = cluster_path.parent.parts[2:]
        current_output_dir = mkdir(output_dir / "/".join(group_path))
        cluster_data = read_cluster_file(cluster_path)
        pdb_reference = df["pdb_ref"].iloc[ix]
        positions_df = cluster_data["POSITION_DATA"]
        refseq_id = cluster_data["REFSEQ_ID"]
        alignment = cluster_data["ALIGNMENT"]
        protein_id = refseq_id.split("_")[1]
        cluster_name = cluster_data['CLUSTER_NAME']

        output_file = current_output_dir / f"{cluster_name}.cluster"

        # First we check PDB BLAST.
        pdb_identity = df["pdb_total_identity"].iloc[ix]

        print(f"Working on {cluster_path}...")

        # FIRST CASE: PDB IDENTITY IS GREATER THAN 90%
        # In this case, we will parse the RSA and B-values from the template
        # PDB without modelling. This way, we are confident that the changes
        # will be minimal, and we can keep the B-values from the original PDB.
        if pdb_identity >= 0.9:
            pdb_id = df['pdb_id'].iloc[ix]
            pdb_chain = df['pdb_chain'].iloc[ix]
            full_pdb = f"{pdb_id}|{pdb_chain}"

            bvals_dict = bvals_results[full_pdb]
            rsa_dict = rsa_results[full_pdb]
            bepi_dict = bepipred_results[refseq_id]
            hydro_dict = hydrophobicity_results[refseq_id]

            new_aln, aln_values = align_features(muscle_exe,
                                                 alignment,
                                                 bepi_dict,
                                                 hydro_dict,
                                                 rsa_dict,
                                                 bvals_dict
                                                 )

            for record in new_aln:
                if record.id == refseq_id:
                    aligned_refseq = str(record.seq)
            aligned_values_list = []
            for counter, position in enumerate(aligned_refseq):
                new_pos = [counter + 1, position]

                aln_bepi = aln_values["BEPI"][counter]
                new_pos.append(aln_bepi[2])

                aln_hydro = aln_values["HYDRO"][counter]
                new_pos.append(aln_hydro[2])

                if aln_values["RSA"] is not None:
                    aln_rsa = aln_values["RSA"][counter]
                    new_pos.append(aln_rsa[2])
                else:
                    new_pos.append(np.nan)

                if aln_values["BVALS"] is not None:
                    aln_bvals = aln_values["BVALS"][counter]
                    new_pos.append(aln_bvals[2])
                else:
                    new_pos.append(np.nan)

                aligned_values_list.append(new_pos)

            parsed_df = align_values_to_df(positions_df,
                                           aligned_values_list
                                           )
            generate_cluster_file(cluster_data, parsed_df, output_file)
            continue

        # SECOND CASE: PDB IDENTITY BETWEEN 90 AND 50%
        # In this case, we can only parse the RSA from the model
        elif 0.9 > pdb_identity >= 0.5:
            bepi_dict = bepipred_results[refseq_id]
            hydro_dict = hydrophobicity_results[refseq_id]
            bvals_dict = None

            if protein_id in rsa_results:
                rsa_dict = rsa_results[protein_id]
            else:
                print(f"{protein_id} does not have a good model")
                print("Could not obtain RSA results")
                rsa_dict = {("A", k): (v[0], np.nan)
                            for k, v in sorted(bepi_dict.items())
                            }
            new_aln, aln_values = align_features(muscle_exe,
                                                 alignment,
                                                 bepi_dict,
                                                 hydro_dict,
                                                 rsa_dict,
                                                 bvals_dict=None
                                                 )

            for record in new_aln:
                if record.id == refseq_id:
                    aligned_refseq = str(record.seq)
            aligned_values_list = []
            for counter, position in enumerate(aligned_refseq):
                new_pos = [counter + 1, position]

                aln_bepi = aln_values["BEPI"][counter]
                new_pos.append(aln_bepi[2])

                aln_hydro = aln_values["HYDRO"][counter]
                new_pos.append(aln_hydro[2])

                if aln_values["RSA"] is not None:
                    aln_rsa = aln_values["RSA"][counter]
                    new_pos.append(aln_rsa[2])
                else:
                    new_pos.append(np.nan)

                new_pos.append(np.nan)  # B-values

                aligned_values_list.append(new_pos)

            parsed_df = align_values_to_df(positions_df,
                                           aligned_values_list
                                           )
            generate_cluster_file(cluster_data, parsed_df, output_file)
            continue

        # THIRD CASE: PDB IDENTITY BELOW 50%
        # In this case, we can only obtain sequence based
        else:
            bepi_dict = bepipred_results[refseq_id]
            hydro_dict = hydrophobicity_results[refseq_id]
            rsa_dict = None
            bvals_dict = None
            new_aln, aln_values = align_features(muscle_exe,
                                                 alignment,
                                                 bepi_dict,
                                                 hydro_dict,
                                                 rsa_dict=None,
                                                 bvals_dict=None
                                                 )
            for record in new_aln:
                if record.id == refseq_id:
                    aligned_refseq = str(record.seq)
            aligned_values_list = []
            for counter, position in enumerate(aligned_refseq):
                new_pos = [counter + 1, position]

                aln_bepi = aln_values["BEPI"][counter]
                new_pos.append(aln_bepi[2])

                aln_hydro = aln_values["HYDRO"][counter]
                new_pos.append(aln_hydro[2])

                new_pos.append(np.nan)  # RSA
                new_pos.append(np.nan)  # B-values

                aligned_values_list.append(new_pos)

            parsed_df = align_values_to_df(positions_df,
                                           aligned_values_list
                                           )
            generate_cluster_file(cluster_data, parsed_df, output_file)
            continue
