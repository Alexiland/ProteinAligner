# Downstream tasks for ProteinAligner
The implementation of how to apply our pre-trained ProteinAligner for protein-related downstream tasks. In this repository, we offer an example of how to load our pre-trained ProteinAligner to calculate the corresponding embeddings for the input protein sequences.

## Requirements
The major dependencies used in this project are as following:
```
Python 3.8.16
fair-esm 2.0.0
keras 2.9.0
pandas 1.3.5
numpy 1.21.6
scikit-learn 1.0.2
tensorflow 2.9.2
torch 1.13.0+cu116
focal-loss
```
Install all the above packages by ```!pip install package_name```

## Embedding calculation
You can try the code on the provided examplary data. Here is the command: 
```bash
python esm_example.py
```
If everything works, you can find the saved embedding in your save folder, which is also set in the esm_example.py file.

## Downstream task fine-tuning
After getting the calculated embeddings, you can apply them for further applications, like classification, regression, etc. We tested ProteinBind in four downstream tasks: [Type I Anti-CRISPR activities detection](https://github.com/USask-BINFO/AcrTransAct), [Peptide bioactivities detection](https://github.com/dzjxzyd/UniDL4BioPep), [Pathogenicity prediction of missense variants](https://github.com/wlin16/VariPred), [Minimum inhibitory concentration value prediction](https://github.com/amirpandi/Deep_AMP), and [Thermostability prediction](https://github.com/VITA-Group/SMC-Bench). When applying to these tasks, we replaced the vanilla encoders by the encoders from pre-trained ProteinBind.

## License
This work is licensed under the MIT license. See the [LICENSE](LICENSE) for details.


## Acknowledgement
We appreciate the developers of [ESM](https://github.com/facebookresearch/esm), and we express our gratitude for these awesome projects.
