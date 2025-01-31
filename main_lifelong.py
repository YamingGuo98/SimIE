import os.path
import sys
import json
import argparse
import wandb

sys.path.append('..')
from easyeditor import (
    FTHyperParams,
    GraceHyperParams,
    MEMITHyperParams,
    ROMEHyperParams,
    MENDHyperParams,
    AlphaEditHyperParams,
    WISEHyperParams,
    BaseEditor,
)
from easyeditor.models.prune import PRUNE


true_dir = "."
# true_dir = "/data/anonymous/simIE"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', default="ROME", type=str)
    parser.add_argument('--model_name', default="llama-7b", type=str)
    parser.add_argument('--data_dir', default="./data", type=str)
    parser.add_argument('--data_type', default="ZsRE", type=str,
                        choices=['ZsRE', 'hallucination', 'counterfact'])
    parser.add_argument('--output_dir', default='./outputs', type=str)
    parser.add_argument('--ds_size', default=1000, type=int)
    parser.add_argument('--sequential_edit', default=True, action="store_true")
    parser.add_argument('--simIE', default=False, action="store_true")
    parser.add_argument('--lamHyper', default=1, type=float)
    parser.add_argument('--init_model', default=False, action="store_true")
    parser.add_argument('--solver', default='LU', type=str)
    parser.add_argument('--save_model', default=False, action="store_true")
    args = parser.parse_args()

    if args.editing_method == 'PRUNE':
        args.hparams_dir = f"hparams/ROME/{args.model_name}.yaml"
    else:
        args.hparams_dir = f"hparams/{args.editing_method}/{args.model_name}.yaml"
    args.data_dir = args.data_dir.replace('.', true_dir, 1)

    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'MEND':
        editing_hparams = MENDHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'AlphaEdit':
        editing_hparams = AlphaEditHyperParams
    elif args.editing_method == 'GRACE':
        editing_hparams = GraceHyperParams
    elif args.editing_method == 'WISE':
        editing_hparams = WISEHyperParams
    elif args.editing_method == 'PRUNE':
        editing_hparams = ROMEHyperParams
    else:
        raise NotImplementedError

    K = args.ds_size

    if args.data_type == 'ZsRE':
        edit_data = json.load(open(f'{args.data_dir}/{args.data_type}/zsre_mend_edit.json', 'r', encoding='utf-8'))[:K]
        loc_data = json.load(open(f'{args.data_dir}/{args.data_type}/zsre_mend_train.json', 'r', encoding='utf-8'))[:K]
        loc_prompts = [edit_data_['loc'] + ' ' + edit_data_['loc_ans'] for edit_data_ in loc_data]

        prompts = [edit_data_['src'] for edit_data_ in edit_data]
        subject = [edit_data_['subject'] for edit_data_ in edit_data]
        rephrase_prompts = [edit_data_['rephrase'] for edit_data_ in edit_data]
        target_new = [edit_data_['alt'] for edit_data_ in edit_data]
        locality_prompts = [edit_data_['loc'] for edit_data_ in edit_data]
        locality_ans = [edit_data_['loc_ans'] for edit_data_ in edit_data]
        locality_inputs = {
            'neighborhood':{
                'prompt': locality_prompts,
                'ground_truth': locality_ans
            },
        }
    elif args.data_type == 'counterfact':
        edit_data = json.load(open(f'{args.data_dir}/{args.data_type}/counterfact-edit.json', 'r', encoding='utf-8'))[:K]
        loc_data = json.load(open(f'{args.data_dir}/{args.data_type}/counterfact-train.json', 'r', encoding='utf-8'))[:K]
        loc_prompts = [edit_data_['locality_prompt'] + ' ' + edit_data_['locality_ground_truth'] for edit_data_ in loc_data]

        prompts = [edit_data_['prompt'] for edit_data_ in edit_data]
        subject = [edit_data_['subject'] for edit_data_ in edit_data]
        rephrase_prompts = [edit_data_['rephrase_prompt'] for edit_data_ in edit_data]
        target_new = [edit_data_['target_new'] for edit_data_ in edit_data]
        locality_prompts = [edit_data_['locality_prompt'] for edit_data_ in edit_data]
        locality_ans = [edit_data_['locality_ground_truth'] for edit_data_ in edit_data]
        locality_inputs = {
            'neighborhood': {
                'prompt': locality_prompts,
                'ground_truth': locality_ans
            },
        }
    elif args.data_type == 'hallucination':
        edit_data = json.load(open(f'{args.data_dir}/{args.data_type}/hallucination-edit.json', 'r', encoding='utf-8'))[:K]
        loc_data = json.load(open(f'{args.data_dir}/{args.data_type}/hallucination-train.json', 'r', encoding='utf-8'))[:K]
        loc_prompts = [edit_data_['locality_prompt'] + ' ' + edit_data_['locality_ground_truth'] for edit_data_ in loc_data]

        prompts = [edit_data_['prompt'] for edit_data_ in edit_data]
        subject = [edit_data_['subject'] for edit_data_ in edit_data]
        rephrase_prompts = None
        target_new = [edit_data_['target_new'] for edit_data_ in edit_data]
        locality_prompts = [edit_data_['locality_prompt'] for edit_data_ in edit_data]
        locality_ans = [edit_data_['locality_ground_truth'] for edit_data_ in edit_data]
        locality_inputs = {
            'neighborhood': {
                'prompt': locality_prompts,
                'ground_truth': locality_ans
            },
        }

    hparams = editing_hparams.from_hparams(f'{args.hparams_dir}')
    hparams.model_name = hparams.model_name.replace('.', true_dir, 1)
    if hasattr(hparams, 'stats_dir'):
        hparams.stats_dir = hparams.stats_dir.replace('.', true_dir, 1)
    if hasattr(hparams, 'P_loc'):
        hparams.P_loc = hparams.P_loc.replace('.', f'./data/stats/{hparams.model_name.split("/")[-1]}', 1)
        hparams.P_loc = hparams.P_loc.replace('.', true_dir, 1)
    if hasattr(hparams, 'archive'):
        hparams.archive = hparams.archive.replace('.', true_dir, 1)

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f'{hparams.model_name.split("/")[-1]}_{args.editing_method}_N={args.ds_size}_Sequential={args.sequential_edit}_{args.simIE}_{args.lamHyper}_{args.init_model}.json'
        )
    print("See results at: ", output_file)

    run_config = {
        'data_type': args.data_type,
        'ds_size': args.ds_size,
        'model_name': hparams.model_name.split("/")[-1],
        'editing_method': args.editing_method,
        'simIE': args.simIE,
        'lamHyper': args.lamHyper,
        'init_model': args.init_model,
        'solver': args.solver,
        'other': hparams.to_dict()
    }
    run = wandb.init(
        project="SimIE",
        entity="anonymous",
        config=run_config
    )

    editor = BaseEditor.from_hparams(hparams)
    if args.editing_method == 'PRUNE':
        reduce_name = 'log2' if "mistral" in args.model_name else 'log1_2'
        run.config.update({'reduce_name': reduce_name})
        editor.prune = PRUNE(reduce_name)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        target_new=target_new,
        loc_prompts=loc_prompts,
        subject=subject,
        locality_inputs=locality_inputs,
        sequential_edit=args.sequential_edit,
        eval_metric='ppl' if args.data_type == 'hallucination' else 'token em',
        simIE=args.simIE,
        lamHyper=args.lamHyper,
        init_model=args.init_model,
        solver=args.solver,
        run=run,
        save_model=args.save_model,
    )

    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    run.finish()




