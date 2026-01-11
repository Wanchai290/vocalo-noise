## Dataset requirements
- Librispeech dev-clean dataset : [download](https://openslr.trmal.net/resources/12/dev-clean.tar.gz)
- ESC-50 dataset : [download](https://github.com/karoldvl/ESC-50/archive/master.zip)

Once downloaded, please extract the datasets in the root directory of the project, and run the following
at the root directory. These commands will generate the dataset used in the project.
Ensure that you have the following directories at the root of the project :
```
.gitignore
(...)
ESC-50-master/
LibriSpeech/
```

```bash
python3 librispeech_data_fusion.py
python3 esc50_librispeech_generation.py
```