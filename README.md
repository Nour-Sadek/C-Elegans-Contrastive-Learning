# Inferring Intrerpretable, TF motif-based representations of Caenorhabditis Elegans promoter regions using contrastive learning with orthology as the learning signal

## Table of Contents

1. Homologous species selection and determination of orthologous genes
2. Data collection of upstream non-coding promoter regions of Caenorhabditis Elegans genes and its orthologs
3. Architecture of the motif-based encoder
4. Training of the encoder to infer PWMs of the motifs through reverse homology (contrastive learning) that uses evolution 
(orthologous sequences) as the training signal
5. Encoding of the promoter sequences after training by averaging the representation over all available orthologs
6. Comparison of the learned PWMs to available consensus motif databases for transcription factors using TomTom
7. Go Enrichment Analysis
8. References

## 1. Homologous species selection and determination of orthologous genes


