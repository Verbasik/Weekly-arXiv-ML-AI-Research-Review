# Evo-2: A Genome Generation Model That Knows the Entire Tree of Life

## **The Story of Evo 2**

Let me briefly introduce the Arc Institute.

![Pictures_1.jpg](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-10/assets/Pictures_1.jpg)

The Arc Institute is an independent, non-profit research institute located in California. Its primary mission is to accelerate scientific progress and investigate the fundamental causes of complex diseases. The institute employs an innovative research model that grants scientists complete freedom to pursue directions driven by curiosity, while simultaneously fostering deep interdisciplinary collaboration.

![Pictures_2.png](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-10/assets/Pictures_2.png)

The Arc Institute and its two scientists may not be widely known, but you’ve likely heard the romantic story that became popular in academic circles in 2022. A billionaire fiancé, a private research scientist from a prestigious university, made her a generous donation of $500 million. This enabled her to avoid the hassle of grant applications and hire 150 scientists to focus entirely on research.

This engineer is named Sylvana Conermann. She holds a Ph.D. in neuroscience from the Massachusetts Institute of Technology (MIT) and previously worked in the lab of renowned CRISPR specialist Feng Zhang.

Patrick Collison is the billionaire fiancé (co-founder of the Arc Institute) and one of the world’s youngest self-made billionaires. At age 20, he dropped out of MIT and co-founded the technology company Stripe, one of whose earliest investors was Elon Musk. Today, Stripe employs thousands of people worldwide.

![Pictures_3.jpg](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-10/assets/Pictures_3.jpg)

Patrick Su, another co-founder of the Arc Institute, earned a B.S. in molecular and cellular biology from the University of California, Berkeley in 2010. He then continued his studies at Harvard University, where he obtained a master’s degree in biology and a Ph.D. in biochemistry, completing his doctorate in just one year. As Feng Zhang’s first graduate student, he made significant contributions to early research and development of the CRISPR-Cas9 technology.

![Pictures_4.jpg](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-10/assets/Pictures_4.jpg)

In December 2021, Patrick Su, along with his friends Patrick Collison and Sylvana Conermann, founded the Arc Institute. This well-funded institution, which values scientific freedom, has created something akin to Evo 2.

By the way, Evo 2 is an improved version of Evo 1. The training data for Evo 1 contained only single-cell genome data. The results of Evo 1, published in 2024, can be found in the cover article of Science.

## **Introduction and Motivation**

Progress in DNA sequencing and editing methods over the past decades has transformed genomic data analysis into one of the fundamental tools of modern biology. However, for comprehensive analysis of DNA sequences—including predicting functional effects of mutations and rational design of novel biological systems—it is necessary to develop highly efficient methods for machine representation of data. In this work, we present **Evo 2**, a large-scale language model trained on a corpus of 9 trillion tokens of genomic sequences spanning all domains of life (bacteria, archaea, eukaryotes, bacteriophages, and others).

The conceptual foundation of this research lies in applying the principles of autoregressive language models (analogous to natural language processing methods) to the symbols of the nucleotide alphabet: adenine (A), cytosine (C), guanine (G), and thymine (T). The results demonstrate that Evo 2, with its optimized scaling architecture and representative dataset, can uncover fundamental statistical patterns in DNA sequences, enabling efficient resolution of the following tasks:

- Prediction of functional significance of genetic variants (Variant Effect Prediction, VEP)
- Generation of realistic genomic sequences scalable to the full genome level
- Optimization of epigenomic patterns, including modeling of chromatin accessibility loci

## **Data and Training Process**

- **Dataset (OpenGenome2):** The foundational corpus for training the model is the open repository **OpenGenome2**, integrating diverse types of DNA sequences (bacterial and eukaryotic genomes, metagenomic data, organellar sequences, messenger RNAs, non-coding RNAs, and other genomic elements). The total corpus volume exceeds 9 trillion nucleotides.
  
- **Parametric Scaling:** Two versions of the Evo 2 model were developed, differentiated by parameter count: **7 billion** and **40 billion** respectively. Architectural optimization enabled processing of contextual windows up to **1 million base pairs (bp)**.

- **Stratified Training Methodology:**
  1. **Pretraining:** An initial phase with a limited contextual window (8–16 thousand bp) to identify local genomic features, including coding regions, regulatory elements, and functional motifs.
  2. **Midtraining:** Incremental expansion of the contextual window to 1 million bp, enabling identification of long-range dependencies and macrostructural genomic elements such as bacterial operon organization or complex intron-exon architectures in eukaryotic genes.

- **StripedHyena 2 Architecture:** Unlike traditional Transformer-based models, the Evo 2 architecture is built upon an optimized hybrid convolutional-attention mechanism, **StripedHyena 2**, demonstrating enhanced computational efficiency at scale.

- **Representativeness Correction Algorithms (Repeat Down Weighting):** A system of differential weighting of repetitive genomic elements (tandem repeats, duplications) was implemented, alongside integration of phylogenetic markers to ensure taxonomic stratification of sequences during model training.

### **Architectural Features of Evo 2, Training Methodology, and Dataset Structure**

This study introduces an innovative model, Evo 2, enabling high-efficiency modeling of genomic sequences across all taxonomic domains. A schematic representation, illustrated in **Figure 1**, demonstrates the integrated architecture of the model, stratified training stages, and the structure of the datasets used.

![Table_2](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-10/assets/Figure_2.png)

> Apologies for the image quality—this is the best available, even in the original paper.

As shown in **Figure 1A**, the Evo 2 model is conceptualized as a comprehensive system for interpreting the “nucleotide language,” applicable across scales from the molecular to the organismal level. The model’s architecture is optimized to identify structural features of coding and non-coding regions, model protein-nucleic acid interactions, and analyze high-level genomic and epigenomic patterns.

**Figure 1B** presents a projection of genomic sequences from the training corpus into a two-dimensional space using the UMAP algorithm. Discrete clusters of points correspond to different domains and taxonomic groups of organisms, color-coded according to their phylogenetic classification. This visualization clearly demonstrates the heterogeneity of the training corpus and justifies the need for a universal model capable of effective generalization across all domains of life.

**Figures 1C–D** illustrate the two-phase training strategy:
1. The pretraining phase, focused on local sequence patterns and major functional elements.
2. The midtraining phase, characterized by incremental expansion of the contextual window to 1 million bp to identify global genomic patterns.

During training, a suite of specialized methodologies was implemented, including differential weighting of repetitive elements and optimized batch data distribution, aimed at enhancing training quality when working with long sequences and ensuring representative coverage of diverse genomic functional domains.

**Figure 1E** shows the statistical distribution of tokens used at each training stage for the Evo 2 models with 40 and 7 billion parameters, respectively. It is evident that the volume of data dominates during midtraining with the expanded contextual window, ensuring optimal approximation to real genomic scales.

**Figure 1F** presents a schematic visualization of the StripedHyena 2 architecture implemented in Evo 2, featuring three differentiated block types (SE, MR, and LI), structured to maximize efficiency of convolutional and attention operations on large input sequences. **Figure 1G** demonstrates a comparative performance analysis of StripedHyena 2 against its predecessor (StripedHyena 1) and the classical Transformer when trained on 1024 GPUs, with clear superiority of the new architecture in computational efficiency.

**Figure 1H** displays validation results under varying context length and scale (parameter count) conditions, illustrating a positive correlation between expanded contextual window, increased model parameter volume, and improved final metrics (including reduced perplexity). Finally, **Figure 1I** contains a methodological description of the modified “needle in a haystack” task—a specialized test evaluating Evo 2’s ability to extract relevant information from a context up to 1 million base pairs long. Results confirm the model’s effective capacity for information retention in extensive contexts, a critical parameter when working with whole-genome sequences.

Thus, the schematic representation in **Figure 1** integrates the key aspects of Evo 2’s architecture, data structure, and training methodology, demonstrating the model’s universal applicability across all levels of biological organization—from individual genes to whole-genome sequences.

## **Variant Effect Prediction (VEP)**

![Table_3](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-10/assets/Figure_3.png)

One of the fundamental tasks in modern genomics is predicting the functional consequences (pathogenic or neutral) of genetic variations. Evo 2 implements a zero-shot approach based on quantitative assessment of the likelihood ratio between mutant and reference sequences (Figure 2A).

### **Analysis of Pathogenic Variants in ClinVar**

The model demonstrates superior accuracy in identifying pathogenic mutations, including non-coding variants and structural rearrangements (insertions/deletions), outperforming existing tools in zero-shot mode. For specialized analytical tasks, the authors also developed compact classifiers based on Evo 2 vector representations, achieving state-of-the-art results in predicting variants in the BRCA1 and BRCA2 genes associated with cancer.

As shown in Figure 2B, the model effectively identifies functionally significant sites across genomes of diverse organisms. Analysis of sequence probability changes upon introducing mutations along gene start sites across diverse model organisms revealed that the model correctly predicts reduced probability of mutations in critical elements: start codons of protein-coding genes, the first two bases of each codon in coding regions, and ribosome binding sites in the 5' untranslated region (5'UTR). This indicates the model’s ability to identify functionally conserved genomic elements without explicit annotation.

### **Universality of Predictions Across Domains of Life**

Figures 2C–D show stratified analysis of mutation probability across functional genomic elements in prokaryotic (2C) and eukaryotic (2D) sequences using the 7-billion-parameter version of Evo 2. Results demonstrate consistency with fundamental biological principles: genomic regions under strong evolutionary pressure exhibit heightened sensitivity to mutations in model predictions. Median probability change from wild-type to mutant sequence is visualized with differentiation by taxonomic domain (for prokaryotes) or kingdom (for eukaryotes), confirming the model’s universal applicability across diverse phylogenetic lineages.

### **Splice Variant Analysis (SpliceVarDB)**

The Evo 2 model effectively distinguishes variants disrupting splicing in exonic and intronic regions. Notably, unlike traditional variant effect prediction methods, Evo 2 does not require multiple sequence alignments, yet achieves high predictive accuracy.

To validate the predictive efficacy of Evo 2, the authors conducted a comprehensive Spearman correlation analysis between zero-shot sequence probability predictions and experimental data obtained via Deep Mutational Scanning (DMS) across a wide range of proteins and RNAs (Figure 2E). This methodology allows experimental determination of functional effects of thousands of variants simultaneously, providing robust empirical validation of model accuracy.

Additionally, the authors implemented an innovative exon classifier with single-nucleotide resolution based on Evo 2 vector embeddings (Figure 2F). Comparative analysis of classifier performance trained on embeddings from various models (Evo 2, Nucleotide Transformer, and Evo 1) on a dataset of eight phylogenetically distant species demonstrated Evo 2’s superiority in ROC area under the curve (AUROC) for identifying exonic nucleotides (Figure 2G). Visualization of the classifier’s predictions for the human STOML2 locus (Figure 2H) clearly demonstrates the accuracy of exon-intron boundary detection, where the vertical axis represents the classifier’s quantitative score and the horizontal axis represents genomic position.

### **Analysis of Non-Coding Regions and Predictive Genetics**

In addition to single-nucleotide variants (SNVs) in protein-coding sequences, the model demonstrates high efficacy in predicting pathogenicity of variants in non-coding and splicing-regulatory elements, as well as other functional genomic components.

Particular interest lies in Evo 2’s application to predictive analysis of gene essentiality. Figure 2I presents results using the mutational probability of premature stop codon insertions as a genetic perturbation to predict essentiality/non-essentiality of genes in bacteria and bacteriophages. Model predictions show high concordance with experimental studies of gene essentiality, confirming Evo 2’s practical applicability in functional genomics and synthetic biology.

A similar methodological approach was applied to predict the functional significance of human long non-coding RNAs (lncRNAs) (Figure 2J). The model effectively distinguishes essential (N = 46) from non-essential (N = 5,417) lncRNAs based on assessment of their random rearrangement (sequence scrambling) probability, confirmed by experimental cell essentiality screens across all tested cell lines.

Thus, integrative analysis of Evo 2’s predictive efficacy across diverse experimental paradigms confirms its universal applicability for predicting functional effects of genetic variations across all taxonomic domains, without requiring additional training or parameter tuning for specific biological contexts.

## **Genome Sequence Generation**

Unlike narrow-purpose analytical tools, Evo 2 is a **generative** model operating on an autoregressive principle for nucleotide sequence formation. Experimental validation of the model included generation of several types of genomic sequences:

- **Human mitochondrial genomes:** Generation of full-length sequences of approximately 16 thousand base pairs, with accurate reconstruction of functional elements including transfer RNAs (tRNAs), ribosomal RNAs (rRNAs), and conserved protein-coding genes. Generated sequences show high conservation of genomic organization (synteny) alongside variability at the amino acid sequence level.
  
- **Genome of the model microorganism Mycoplasma genitalium:** Thanks to its extended contextual window (up to 1 million bp), the Evo 2 model can extrapolate and reconstruct long sequences (~580 thousand bp). Generated genomic sequences preserve structural homology with known protein domains (confirmed by Pfam database analysis) while diverging at the primary structure level, opening potential avenues for identifying novel functional protein variants.

- **Chromosomes of yeast Saccharomyces cerevisiae:** Successful generation of long genomic fragments on the order of hundreds of thousands of base pairs, with accurate reconstruction of functional elements including tRNAs, intron-exon structures, and regulatory promoter regions.

## **Epigenomic Pattern Optimization**

A category of more complex analytical tasks involves generating sequences matching specified criteria for epigenomic characteristics, particularly local chromatin accessibility. To implement this approach, the authors developed an inference-time search methodology:

- **Integration of Predictive Models (Enformer, Borzoi):** These auxiliary models quantitatively predict the degree of chromatin accessibility (open conformation) for regulatory factor binding in specific cell types based on nucleotide sequence.
  
- **Beam Search Algorithm:** Generation proceeds incrementally in 128-bp fragments, with Evo 2 generating multiple candidate sequence variants, while Enformer/Borzoi evaluate corresponding epigenomic profiles. During selective filtering, only sequences demonstrating maximal convergence with a target reference pattern are retained, enabling stepwise construction of a target sequence of desired length.

- **Experimental Demonstration:** Researchers, modeling differential patterns of open and compacted chromatin regions, demonstrated the ability to encode informational patterns analogous to Morse code (a system of dots and dashes) as discrete epigenomic peaks.

Significantly, the quality of generated patterns shows positive correlation with increased computational resources (expansion of the candidate sequence pool at each generation step). This phenomenon illustrates the principle of inference-time scaling applied to biological models.

## **Interpretation of Internal Representations (Sparse Autoencoders)**

Given the increasing complexity and scalability of language models, the authors focused on mechanistic interpretability. For this purpose, **Sparse Autoencoders (SAEs)** were developed and trained based on analysis of internal activations of Evo 2, aiming to identify latent features demonstrating direct correlation with functional biological elements:

- **Regulatory elements of gene expression:** Sparse autoencoder filters effectively identify specific sequence motifs, including core promoter elements (TATA box) and transcription factor binding sites.
  
- **Protein structural elements:** Activation patterns reveal characteristic features corresponding to major secondary protein structures (α-helices and β-sheets).

- **Prophage integration sites and mobile genetic elements:** The model demonstrates the ability to automatically annotate phage insertions, CRISPR spacer sites, and other mobile elements without explicit training on labeled data.

- **Gene structural organization:** Certain latent features correlate with exon-intron architecture, enabling identification of structural components even in complex genes of higher eukaryotes, including humans.

Implementation of these methodological approaches provides deeper insight into the biological concepts identified by Evo 2 during training and establishes a foundation for high-accuracy annotation of uncharacterized genomes, including paleogenomic data (e.g., the woolly mammoth genome).

## **Conclusion and Future Directions**

This study positions the **Evo 2** model as a universal platform for comprehensive analysis and rational design of genomic sequences. Key findings can be summarized as follows:

- **Taxonomic Universality:** The model demonstrates effective generalization across all three domains of life, extending language modeling methodology to structurally and functionally complex eukaryotic genomes.
  
- **Zero-Shot Variant Effect Prediction (Zero-shot VEP):** Evo 2 outperforms existing algorithms in predicting functional consequences of non-coding and structural variants, functioning as an alignment-free method.

- **Whole-Genome Generation:** The authors have, for the first time, demonstrated an autoregressive approach to reconstructing mitochondrial, bacterial, and eukaryotic genomes with integration of fragments into long contexts on the order of hundreds of thousands of base pairs.

- **Epigenomic Landscape Design:** Integration of auxiliary predictive models enables dynamic optimization of generated sequences according to specified epigenomic patterns.

- **Accessibility and Reproducibility:** The dataset repository (OpenGenome2), source code, and parametric models (7 and 40 billion parameters) are openly available, positioning Evo 2 as one of the most extensive open projects at the intersection of artificial intelligence and molecular biology.

Thus, the **Evo 2** model establishes a new methodological standard in fundamental biological modeling, demonstrating the potential of large-scale language models for efficient analysis and modeling of nucleotide sequences with high granularity—from mutation analysis to genome architecture—at the level of entire chromosomes. Future directions include optimization of inference-time steering methodologies, integration of spatial 3D chromatin structures, and synergy with experimental genome editing techniques, creating a foundation for translating the paradigm of generative biology into practical applications—from molecular diagnostics to synthetic biology.