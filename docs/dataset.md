# Piano Fingering Dataset

Due to licensing restrictions, we are unable to redistribute the PIG dataset inside this repository. You will need to go to the [PIG website](https://beam.kisarazu.ac.jp/~saito/research/PianoFingeringDataset/) and download it by [registering for an account](https://beam.kisarazu.ac.jp/~saito/research/PianoFingeringDataset/register.php).

**Note: We are working with the authors of PIG to make the dataset available for download in a more convenient way.**

The download will contain a zip file called `PianoFingeringDataset_v1.2.zip`. Extract the folder `PianoFingeringDataset_v1.2` from the zip file, then run the following command, in the root directory of `robopianist`:

```bash
python scripts/pig_preprocess.py --dataset_dir /PATH/TO/PianoFingeringDataset_v1.2
```

This will create a directory called `pig_single_finger` in `robopianist/music/data`.

## Sanity Check

You can use the CLI to check that the dataset has been downloaded and processed correctly:

```bash
robopianist --check-pig-exists
# > PIG dataset is ready to use!
```
