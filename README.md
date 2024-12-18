# OPUS-GO

Accurate annotation of protein and RNA sequences is essential for understanding their structural and functional attributes. However, due to the relative ease of obtaining whole sequence-level annotations compared to residue-level annotations, existing biological language model (BLM)-based methods often prioritize enhancing sequence-level classification accuracy while neglecting residue-level interpretability. To address this, we introduce OPUS-GO, which exclusively utilizes sequence-level annotations. It not only provides the sequence-level annotations but also offers the rationale behind these predictions by pinpointing their corresponding most critical residues within the sequence. Our results show that, by leveraging features derived from BLMs and our modified Multiple Instance Learning (MIL) strategy, OPUS-GO exhibits superior sequence-level classification accuracy compared to baseline methods in most downstream tasks. Furthermore, OPUS-GO demonstrates robust interpretability by accurately identifying the residues associated with the corresponding labels. Additionally, the OPUS-GO framework can be seamlessly integrated into any language model, enhancing both accuracy and interpretability for their downstream tasks.

## Usage

### Dependency

```
Python 3.7
TensorFlow 2.4
Horovod
```

The pre-trained models of OPUS-GO for each downstream task are hosted on [Google Drive](xxx).

## Reference 
```bibtex
@article{xu2024opus3,
  title={OPUS-GO: An interpretable protein/RNA sequence annotation framework based on biological language model},
  author={Xu, Gang and Lv, Ying and Zhang, Ruoxi and Xia, Xinyuan and Wang, Qinghua and Ma, Jianpeng},
  journal={bioRixv},
  year={2024},
}
