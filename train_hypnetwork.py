from easyeditor import EditTrainer, MENDTrainingHparams, ZsreDataset
import argparse

true_dir = "."
# true_dir = "/data/anonymous/simIE"

parser = argparse.ArgumentParser()
parser.add_argument('--editing_method', default="MEND", type=str)
parser.add_argument('--model_name', default="gpt2-xl", type=str)
args = parser.parse_args()

args.hparams_dir = f"hparams/TRAINING/{args.editing_method}/{args.model_name}.yaml"
if args.editing_method == 'MEND':
    training_hparams = MENDTrainingHparams.from_hparams(args.hparams_dir)
else:
    raise ValueError("Editing method not supported")

training_hparams.model_name = training_hparams.model_name.replace('.', true_dir, 1)
training_hparams.tokenizer_name = training_hparams.tokenizer_name.replace('.', true_dir, 1)
training_hparams.results_dir = f"{true_dir}/hypernetwork/{args.editing_method}"

train_ds = ZsreDataset(f"{true_dir}/data/ZsRE/zsre_mend_train.json", config=training_hparams)
eval_ds = ZsreDataset(f"{true_dir}/data/ZsRE/zsre_mend_eval.json", config=training_hparams)

trainer = EditTrainer(
    config=training_hparams,
    train_set=train_ds,
    val_set=eval_ds
)
trainer.run()