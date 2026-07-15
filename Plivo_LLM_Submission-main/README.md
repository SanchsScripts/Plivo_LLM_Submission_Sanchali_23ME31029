# Execution Guide

Run the training and evaluation cycle using the following commands:
  python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt.pt
  python evaluate.py --checkpoint ckpt.pt --text_file ../data/dev_eval.txt

* Laptop CPUs typically complete the reference run within approximately 1.5 to 3 minutes.
* You must document each execution in the `RUNLOG.md` file.
* Developers can modify any code inside `train.py` and `model.py`.
* The maximum resource limits and the program interface for `evaluate.py` must remain completely unchanged.
* Prior to the expiration of the time window, verify that the delivery directory contains `ckpt.pt`, all source scripts, `RUNLOG.md`, `NOTES.md`, and `SUMMARY.html`.
* Consult the "Deliverables" portion of the assignment instructions to verify the list of required files.
