# Implementing ANNs with Tensorflow
## How do I setup teh Repository?
1. Clone the repository.
1. cd into the repository.
1. Tell git where to find the configuration information for our iPython Notebooks with this command: `git config --add include.path $(pwd)/.gitconfig` (The path needs to point to your root git repository where the .gitconfig is stored).

## How do I setup a new assignment notebook?
1. Create a new Jupyter Notebook.
1. Hit `Edit -> Edit Notebook Metadata`.
1. Add `"git": { "suppress_outputs": true },` as a top level element to the json metadata. This will be a notification to our git filter that we want to strip the metadata.

## How do I quickly generate HTML from ipynb?
Run `jupyter-nbconvert --to=html --execute <notebook>.ipynb`

## How do I train a network?
Run something like
```
python util.py -o RMSProp -l 0.001 -b 32 -e 20 -f grid_search.txt -m
ex04.ex03model_improved -t ex04.exercise5
```
where
* `-t`  is the module of the `train_model` function
* `-f`  is the file to log to
* `-m`  is the module whera a `Model` class (subclass of `model.BaseModel`)
    class definition is located

