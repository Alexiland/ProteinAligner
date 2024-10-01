# ProteinAligner
ProteinAligner is a multimodal protein representation framework that unifies structural, sequential, and textual data into a single representation space, allowing for diverse applications with minimal training by using pre-trained modality encoders.

See [our manuscript]() for details.

# Installation
To install, run the following bash script.
```bash
conda create -n proteinaligner python=3.10
conda activate proteinaligner
pip install -r requirements.txt
```
Note that, for simplicity, we have contained the ESM inside the pretrain folder.

# Model Architecture and Pretraining
ProteinAligner is an innovative multimodal protein representation framework that integrates structural, sequential, and textual data into a unified embedding space. It employs three distinct encoding pathways for the amino acid sequence, the 3D structure, and the textual description of a protein, using the sequence as the anchor modality for alignment. The sequence pathway processes the one-dimensional amino acid sequence, generating representations that capture the molecular characteristics of each amino acid using the ESM-2 protein language model. The structure pathway, utilizing the ESM-IF1 model, processes the three-dimensional structure, capturing the protein's molecular interactions and dynamics. The textual pathway employs a vanilla 8-layer Transformer encoder to process textual descriptions derived from experimentally verified publications, creating unique representations for each protein. To ensure compatibility across modalities, each encoded input is projected to the same dimension.

The framework is trained on a large-scale dataset of 150,000 (structure, sequence, description) triples, sourced from the UniProtKB/Swiss-Prot and RCSB PDB databases. The dataset was assembled by mapping PDB IDs to UniProt IDs to retrieve triples for the same protein. During the pretraining stage, ProteinAligner aims to minimize the contrastive loss between sequence-structure and sequence-text pairs, aligning the embeddings of structures and textual descriptions with sequence embeddings to form a unified representation. This pretraining stage leverages the sequence as the anchor modality to facilitate this alignment process.

The training of ProteinAligner follows a two-stage pretrain-finetune pipeline. Initially, the pretraining stage involves computing the contrastive loss on sequence-paired data, specifically focusing on sequence-structure and sequence-text pairs. This contrastive learning approach ensures that the embeddings of protein structures and textual descriptions align with the sequence embeddings of the same protein. Following this, the fine-tuning stage integrates the pretrained encoder weights with task-specific layers, enabling the application of ProteinAligner to a variety of domain-specific tasks. This structured approach allows ProteinAligner to leverage the strengths of each modality, creating a robust and versatile framework for protein representation.

We detail model, training procedures, and data access in [our manuscript](). 

## Running the pretrain code
You can run the pretrain code with the following bash script.

```bash
cd pretrain
torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:23308 --nnodes=1 --rdzv_id 234 --nproc-per-node=8 train_joint_encoder.py \
 --output_dir /PATH/TO/SAVE \
 --log_dir /PATH/TO/SAVE \
 --world_size 8
```

# Downstream
You can check [downstream](ProteinAligner_downstream) see how to apply pretrained ProteinAligner to downstream tasks.

