
# Penguin Cytb Translation & Analysis

# Memis Bilgici  
# BCB 546X Python Assignment 
# Date: 2025-06-05
#

# 1. Load penguin cytochrome-b sequences (FASTA) and adult body mass data (CSV).
# 2. Translate DNA→protein (two methods).
# 3. Compute molecular weight and GC-content.
# 4. Plot results and answer key questions (8a, 8b).
# 5. Bonus: PCA, heatmap, annotated plots.

## 1. Imports & Setup  
# We need Biopython for sequence parsing & analysis, pandas for data handling,
# numpy for numeric operations, matplotlib for plotting, and scikit-learn for PCA.


!pip install biopython scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Data import CodonTable
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.decomposition import PCA

#
## 2. Function Definitions  
# Below we fully document each function with docstrings and inline comments.

 
def get_sequences_from_file(fasta_fn):
    """
    Read a FASTA file of cytochrome-b sequences and return a dict mapping
    species name ("Genus species") → Bio.Seq.Seq object.
    
    Args:
        fasta_fn (str): path to the FASTA file.
    
    Returns:
        dict[str, Seq]: keys are species names, values are Seq objects.
    """
    sequence_data_dict = {}  # Initialize empty dict
    for record in SeqIO.parse(fasta_fn, "fasta"):
        # record.description looks like: ">accession Genus species ..."
        parts = record.description.split()
        species_name = f"{parts[1]} {parts[2]}"  # e.g. "Aptenodytes forsteri"
        sequence_data_dict[species_name] = record.seq
    return sequence_data_dict


def translate_manual(dna_seq):
    """
    Translate a DNA sequence to amino acids by manually looping through codons,
    using the Vertebrate Mitochondrial code, and dropping a terminal stop if present.
    
    Args:
        dna_seq (Seq or str): nucleotide sequence.
    
    Returns:
        str: translated amino acid string.
    """
    mito = CodonTable.unambiguous_dna_by_name["Vertebrate Mitochondrial"]
    seq_str = str(dna_seq).upper()
    aa_list = []
    # Loop over full codons only
    for i in range(0, (len(seq_str)//3)*3, 3):
        codon = seq_str[i:i+3]
        # If this codon is a stop at the very end, break
        if i+3 == len(seq_str) and codon in mito.stop_codons:
            break
        # Map to amino acid, default 'X' if unknown
        aa_list.append(mito.forward_table.get(codon, 'X'))
    return "".join(aa_list)


def translate_biopython(dna_seq):
    """
    Translate using Biopython's built-in translate with Vertebrate Mito code.
    Stops at the first stop codon encountered.
    
    Args:
        dna_seq (Seq or str): nucleotide sequence.
    
    Returns:
        str: translated amino acid string.
    """
    return str(Seq(str(dna_seq)).translate(
        table="Vertebrate Mitochondrial", to_stop=True))


def compute_molecular_weight(aa_seq):
    """
    Compute molecular weight of a protein sequence using Bio.SeqUtils.ProtParam.
    
    Args:
        aa_seq (str): amino acid sequence (no '*' characters).
    
    Returns:
        float: molecular weight in Daltons.
    """
    analysed = ProteinAnalysis(aa_seq)
    return analysed.molecular_weight()


def compute_gc_content(dna_seq):
    """
    Compute GC-content (fraction of G + C) of a DNA sequence.
    
    Args:
        dna_seq (Seq or str): nucleotide sequence.
    
    Returns:
        float: GC-content (0–1).
    """
    s = str(dna_seq).upper()
    return (s.count('G') + s.count('C')) / len(s)

## 3. Load Data  
# - cytb_seqs: dictionary of species → sequence  
# - penguins_df: DataFrame with columns `species` and `mass` (kg)  


cytb_seqs = get_sequences_from_file("penguins_cytb.fasta")
penguins_df = pd.read_csv("penguins_mass.csv")

# Initialize new columns for later
penguins_df["molecular_weight"] = np.nan
penguins_df["gc_content"] = np.nan


## 4. Compute Molecular Weight & GC-content  
# Loop through each row, translate, then fill in the new columns.


for idx, row in penguins_df.iterrows():
    sp = row["species"]
    seq = cytb_seqs.get(sp)
    if seq is None:
        continue  # skip if no sequence
    aa = translate_biopython(seq)            # choose either method
    penguins_df.at[idx, "molecular_weight"] = compute_molecular_weight(aa)
    penguins_df.at[idx, "gc_content"]       = compute_gc_content(seq)

 
## 5. Plot: Adult Body Mass per Species  
# Task #8:  
# - 8a: What is the smallest penguin species?  
Southern rockhopper penguin (_Eudyptes chrysocome_), mean adult mass = 2.80 kg.
# - 8b: What is the geographical range of this species?
Eudyptes chrysocome breeds on sub-Antarctic islands and the northern Antarctic Peninsula, including South Georgia, the Falkland Islands, South Sandwich Islands, South Orkney and South Shetland Islands, Bouvet, Prince Edward, Marion, Crozet, Kerguelen, and Heard & McDonald Islands; during the non-breeding season, it disperses widely throughout the Southern Ocean.
#%% 
plt.figure(figsize=(10,5))
mass_sorted = penguins_df.set_index("species")["mass"].sort_values()
mass_sorted.plot(kind="bar")
plt.ylabel("Mass (kg)")
plt.title("Penguin Adult Body Mass by Species")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


8a. Smallest Penguin Species  
From the bar chart above, the smallest species is:

> Southern rockhopper penguin (_Eudyptes chrysocome_), mean adult mass = 2.80 kg.

```python
# Code that prints it:
smallest_idx     = penguins_df["mass"].idxmin()
smallest_species = penguins_df.loc[smallest_idx, "species"]
smallest_mass    = penguins_df.loc[smallest_idx, "mass"]
print(smallest_species, smallest_mass)

penguins_df.to_csv("penguins_mass_cytb.csv", index=False)
print("Saved penguins_mass_cytb.csv")



# Bonus Questions

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# (A) Prepare data matrix
#   Using only the three numeric columns: gc_content, molecular_weight, mass
X = penguins_df[['gc_content', 'molecular_weight', 'mass']].dropna()
species = penguins_df.loc[X.index, 'species']

# (1) PCA on [GC, MW, mass]
pca = PCA(n_components=2)
pcs = pca.fit_transform(X)

plt.figure(figsize=(6,6))
for sp in pcs:
    pass  # placeholder
# Plot points colored by genus
genus_colors = {
    'Aptenodytes': 'C0',
    'Eudyptes':    'C1',
    'Eudyptula':   'C2',
    'Pygoscelis':  'C3',
    'Spheniscus':  'C4'
}
for i, sp in enumerate(species):
    genus = sp.split()[0]
    plt.scatter(pcs[i,0], pcs[i,1], color=genus_colors.get(genus,'k'), label=genus if i==0 else "")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
plt.title("PCA of GC, MW & Mass")
plt.grid(True)
# avoid duplicate legend entries
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), title="Genus")
plt.tight_layout()
plt.show()

# (2) Correlation heatmap
corr = X.corr()
plt.figure(figsize=(4,4))
plt.imshow(corr, vmin=-1, vmax=1)
plt.colorbar(label="Pearson r")
plt.xticks(np.arange(3), corr.columns, rotation=45, ha='right')
plt.yticks(np.arange(3), corr.columns)
plt.title("Correlation Matrix")
# annotate values
for i in range(3):
    for j in range(3):
        plt.text(j, i, f"{corr.iloc[i,j]:.2f}", ha='center', va='center', color='w')
plt.tight_layout()
plt.show()

# (3) Annotated scatter: GC vs MW
plt.figure(figsize=(6,6))
plt.scatter(penguins_df['gc_content'], penguins_df['molecular_weight'])
for _, row in penguins_df.iterrows():
    plt.annotate(row['species'].split()[1], 
                 (row['gc_content'], row['molecular_weight']),
                 textcoords="offset points", xytext=(3,3), fontsize=8)
plt.xlabel("GC Content")
plt.ylabel("Molecular Weight (Da)")
plt.title("Annotated GC vs MW")
plt.tight_layout()
plt.show()

# (4) GC Content vs Body Mass
plt.figure(figsize=(6,6))
plt.scatter(penguins_df['gc_content'], penguins_df['mass'])
for _, row in penguins_df.iterrows():
    plt.annotate(row['species'].split()[1],
                 (row['gc_content'], row['mass']),
                 textcoords="offset points", xytext=(3,3), fontsize=8)
plt.xlabel("GC Content")
plt.ylabel("Mass (kg)")
plt.title("GC Content vs Adult Body Mass")
plt.tight_layout()
plt.show()
