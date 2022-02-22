# -*- coding: utf-8 -*-
"""
Created on Thu May 20 2021

Calculate the entropy of each position of the alignments,
and save the results in a file text

@author: arosl
"""
import atomium
from pathlib import Path
from Bio import SeqIO
from Bio import AlignIO
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
import pandas as pd
import scipy.stats
from collections import Counter
import argparse
from io import StringIO
import ast
from collections import defaultdict
from tqdm import tqdm


def init_argparser():
    input_parser = argparse.ArgumentParser()
    input_parser.add_argument('-d',
                              '--dir',
                              required=True
                              )
    return input_parser


def mkdir(path_obj):
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def shannon_entropy(aa_list):
    refined_list = []
    for aa in aa_list:
        if aa not in ["X"]:
            refined_list.append(aa)
    data_series = pd.Series(refined_list)
    counts = data_series.value_counts()
    shannon_entropy = scipy.stats.entropy(counts, base=2)
    return round(shannon_entropy, 2)


def get_coverage(ref_seq, aln_seq):
    ref_seq_ungapped_len = len(ref_seq) - ref_seq.count("-")
    aln_seq_ungapped_len = len(aln_seq) - aln_seq.count("-")
    return round(aln_seq_ungapped_len / ref_seq_ungapped_len, 4)


def alignment_start(aln_seq):
    for count, aa in enumerate(aln_seq):
        if aa == "-":
            continue
        else:
            start_aln = count
            break
    return start_aln


def alignment_end(aln_seq):
    for count, aa in enumerate(reversed(aln_seq)):
        if aa == "-":
            continue
        else:
            end_aln = len(aln_seq) - count - 1
            break
    return end_aln


def repair_aln(aln):
    # Some sequences contain X in residues. This will create artifical gaps
    # in the alignment, so we will fix it.
    new_seqs_dict = {record.id: [] for record in aln}
    for position in range(aln.get_alignment_length()):
        x_record_flag = False
        residues = []
        for record in aln:
            residue = record.seq[position]
            if residue == "X":
                x_record_flag = True
                residues.append(residue)
            else:
                residues.append(residue)
        if not x_record_flag:  # normal case
            for record in aln:
                residue = record.seq[position]
                new_seqs_dict[record.id].append(residue)
        else:  # there is a record with an X
            wrong_flag = True
            for residue in residues:
                #  Check all residues are either X or "-", which would mean
                #  that this is an artificial gap due to the Xs.
                if residue not in ["-", "X"]:
                    wrong_flag = False
            if wrong_flag:  # if all residues are X or -, we will omit them.
                continue
            else:
                for record in aln:
                    residue = record.seq[position]
                    new_seqs_dict[record.id].append(residue)
    new_aln_list = []
    for alnid, alnseq_list in sorted(new_seqs_dict.items()):
        alnseq = "".join(alnseq_list)
        new_aln_list.append(SeqRecord(Seq(alnseq), id=alnid))
    new_aln = MultipleSeqAlignment(new_aln_list)
    return new_aln


def parse_aln(aln):
    positions_list = []
    for position in range(0, aln.get_alignment_length()):
        residues = []
        for record in aln:
            residue = record.seq[position]

            # We get the first position and the last position with actual
            # residues, not gaps. This way we can identify protein fragments.
            # Otherwise, we will return positions with gaps that should not
            # be there, and the entropy calculations will be wrong.
            aln_start = alignment_start(record.seq)
            aln_end = alignment_end(record.seq)

            # Normal case
            if residue not in ["-", "X"]:
                residues.append(residue)

            # In case residue is a gap
            elif residue == "-":
                # Gap at the beggining suggests partial seq
                if position < aln_start:
                    continue  # Skip residue

                # Gap at the end suggests partial seq
                elif position > aln_end:
                    continue  # Skip residue
                else:
                    residues.append(residue)  # Gap is added as normal

            # If residue is badly annotated, we will ignore it.
            elif residue == "X":
                continue

        positions_list.append(residues)
    return positions_list


def pdb_res(pdb_id):
    try:
        pdb = atomium.fetch(pdb_id)
        if pdb.resolution:
            return pdb.resolution
        else:
            return 999
    except Exception as e:
        print("Could not fetch PDB file")
        print(e)
        print("Retrying...")
        try:
            pdb = atomium.fetch(pdb_id)
            if pdb.resolution:
                return pdb.resolution
            else:
                return 999
        except Exception as e2:
            print("Could not fetch PDB file")
            print(e2)
            return 999


def merge_fastas(files_list, merged_file):
    with open(merged_file, "w+") as output:
        for f in files_list:
            parsed_data = read_cluster_file(f)
            record_id = parsed_data["refseqid"]
            masked_seq = parsed_data["maskedseq"]
            output.write(f">{record_id}\n{masked_seq}\n")
    return merged_file


def fasta_to_dict(fasta_file):
    with open(fasta_file, "r") as f:
        fasta_dict = {record.id: str(record.seq)
                      for record in SeqIO.parse(f, "fasta")
                      }
    return fasta_dict


def get_entropy_data(alignment):
    aa_distr_dict = {}
    positions_lists = parse_aln(alignment)
    for counter, position_list in enumerate(positions_lists):
        position_entropy = shannon_entropy(position_list)
        aa_distr = Counter(position_list)
        aa_distr_dict[counter + 1] = (len(position_list),
                                      position_entropy,
                                      aa_distr
                                      )
    return aa_distr_dict


def get_consensus(entropy_dict, alignment, shannon_mask=0):
    # Get the consensus sequence from the most common residue in each position
    consensus_list = []
    for i in range(len(entropy_dict)):
        data = entropy_dict[i + 1]
        distr = data[2]
        most_common_aa = distr.most_common()[0][0]
        consensus_list.append(most_common_aa)
    consensus_seq = "".join(consensus_list)

    # Get the real sequence closest to the consensus
    ref_seq = alignment[0]
    ref_seq_origin = int(ref_seq.id[:2])
    highest_score = 0
    for record in alignment:
        record_origin = int(record.id[:2])

        if "X" in record.seq:
            pairwise_score = 0
        else:
            # We get the pairwise score for the alignment
            pairwise_score = pairwise2.align.globalxx(record.seq,
                                                      consensus_seq,
                                                      score_only=True
                                                      )
        # Higher score means better alignment, and closer sequence
        if pairwise_score > highest_score:
            highest_score = pairwise_score
            ref_seq = record

        elif pairwise_score == highest_score:

            # Priorize origin of proteins
            if record_origin < ref_seq_origin:
                ref_seq = record

            # In case origin is the same
            elif record_origin == ref_seq_origin:

                # If both come from PDB, keep highest resolution
                if record.id == 1:
                    record_pdb_id = record.id.split("_")[1].split("|")[0]
                    refseq_pdb_id = ref_seq.id.split("_")[1].split("|")[0]
                    pdb_res_record = pdb_res(record_pdb_id)
                    pdb_res_refseq = pdb_res(refseq_pdb_id)
                    if pdb_res_record < pdb_res_refseq:
                        ref_seq = record
                    elif pdb_res_record == pdb_res_refseq:
                        ref_seq = sorted([record, ref_seq],
                                         key=lambda x: x.id
                                         )[0]
                    else:
                        continue

                # Any other origin
                else:
                    ref_seq = sorted([record, ref_seq], key=lambda x: x.id)[0]

    # Mask the ref seq according to entropy
    ref_seq_aa = list(ref_seq.seq)
    masked_list = []
    updated_entropy_dict = {}
    for i in range(len(ref_seq_aa)):
        current_residue = ref_seq_aa[i]
        data = entropy_dict[i + 1]
        entropy = data[1]
        updated_entropy_dict[i + 1] = (current_residue,
                                       data[1],
                                       data[0],
                                       [(x, y) for x, y in data[2].items()]
                                       )
        if entropy > shannon_mask:
            masked_list.append("*")
        else:
            if consensus_seq[i] == "-":  # If the consensus is a gap, it means
                continue                 # that this gap should not exist
            elif ref_seq_aa[i] == "-":    # Similarly, if the refseq is a gap,
                continue                 # we will ignore it
            else:
                masked_list.append(ref_seq_aa[i])
    masked_ref_seq = "".join(masked_list)
    return ref_seq, masked_ref_seq, updated_entropy_dict


def generate_cluster_file(aln_file, output_dir, shannon_mask=0):
    aln_file = Path(aln_file)
    output_dir = Path(output_dir)
    print(f"Generating entropy file for {aln_file.name}...", end="")
    cluster_number = aln_file.stem.split("_")[1].replace(" ", "")
    group_name = aln_file.parent.name.replace(" ", "_")
    output_file_name = f"{group_name}_{cluster_number}.cluster"
    output_file = output_dir / output_file_name

    alignment = AlignIO.read(aln_file, "fasta")
    alignment = repair_aln(alignment)
    entropy_data = get_entropy_data(alignment)
    refseq, masked_refseq, entropy_data = get_consensus(entropy_data,
                                                        alignment,
                                                        shannon_mask
                                                        )

    if not len(entropy_data) == len(refseq.seq):
        raise Exception("Entropy data does not match")

    with open(output_file, "w+") as f:
        f.write("¡¡¡¡¡CLUSTER_NAME!!!!!\n")
        f.write(f"{group_name}_{cluster_number}\n")

        f.write("¡¡¡¡¡REFSEQ_ID!!!!!\n")
        f.write(refseq.id)
        f.write("\n")

        f.write("¡¡¡¡¡REFSEQ_SEQ!!!!!\n")
        f.write(str(refseq.seq))
        f.write("\n")

        f.write(f"¡¡¡¡¡MASKED_SEQ_{shannon_mask}!!!!!\n")
        f.write(masked_refseq)
        f.write("\n")

        f.write(f"¡¡¡¡¡ALIGNMENT_{len(alignment)}!!!!!\n")
        for a in alignment:
            f.write(f">{a.id}\n{str(a.seq)}\n")

        f.write("¡¡¡¡¡POSITION_DATA!!!!!\n")
        f.write("position;residue;entropy;depth;distribution\n")
        for position, data in entropy_data.items():
            f.write(f"{position};{';'.join([str(x) for x in data])}\n")

        f.write("¡¡¡¡¡END!!!!!")

    # Make sure that it can be parsed correctly
    data = read_cluster_file(output_file)

    print("done!")
    return output_file, data


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


def main(group_dir):
    group_name = group_dir.name
    root_output_dir = mkdir(Path("./RESULTS/04_ENTROPY") / group_name)
    main_cluster_dict = defaultdict(dict)
    msa_files = [x for x in group_dir.glob("**/*") if x.is_file()]
    for msa_file in tqdm(msa_files):
        cutoff = msa_file.parent.name
        output_dir = mkdir(root_output_dir / msa_file.parent.name)
        output_file, parsed_data = generate_cluster_file(msa_file,
                                                         output_dir,
                                                         shannon_mask=0
                                                         )
        masked_seq = parsed_data["MASKED_SEQ"]
        cluster_name = parsed_data["CLUSTER_NAME"]
        main_cluster_dict[cutoff][cluster_name] = masked_seq
    rows = []
    for cutoff, cluster_dict in main_cluster_dict.items():
        row = [group_name, cutoff, 0, 0]
        for cluster, seq in cluster_dict.items():
            asterisk_count = seq.count("*")
            full_count = len(seq)
            conserved = full_count - asterisk_count
            row[2] += full_count
            row[3] += conserved
        rows.append(row)
    return rows


# %%
if __name__ == "__main__":
    parser = init_argparser()
    args = parser.parse_args()
    root_msa_dir = Path(args.dir)
    all_rows = []
    for group_dir in [x for x in root_msa_dir.glob("./*") if x.is_dir()]:
        group_rows = main(group_dir)
        all_rows.extend(group_rows)
    df = pd.DataFrame(all_rows, columns=["group",
                                         "cutoff",
                                         "positions",
                                         "conserved"
                                         ]
                      )
    df.to_csv(Path("./RESULTS/04_ENTROPY") / "entropy_stats.csv")
