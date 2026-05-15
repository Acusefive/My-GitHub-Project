param(
    [string]$Python = "python",
    [string]$OutputRoot = "ckpts_context_residual_gate",
    [int]$NumEpochs = 0
)

$ErrorActionPreference = "Stop"

$Common = @(
    "--split_mode", "new_concept",
    "--test_concept_ratio", "0.2",
    "--valid_ratio", "0.0",
    "--valid_concept_ratio", "0.0",
    "--fusion_mode", "residual_gate",
    "--ctx_logit_init", "-3.0",
    "--gate_bias_init", "-2.0"
)

$EpochArgs = @()
if ($NumEpochs -gt 0) {
    $EpochArgs = @("--num_epochs", "$NumEpochs")
}

& $Python train_context.py @Common @EpochArgs --model_name sakt --context_type none --ctx_logit_mode none --ckpt_root "$OutputRoot\sakt_baseline"
& $Python train_context.py @Common @EpochArgs --model_name sakt --context_type main --ctx_logit_mode scaled --ckpt_root "$OutputRoot\sakt_main_scaled"
& $Python train_context.py @Common @EpochArgs --model_name sakt --context_type all --ctx_logit_mode scaled --ckpt_root "$OutputRoot\sakt_all_scaled"
& $Python train_context.py @Common @EpochArgs --model_name sakt --context_type main --ctx_logit_mode none --ckpt_root "$OutputRoot\sakt_main_no_ctxlogit"

& $Python train_context.py @Common @EpochArgs --model_name saint --context_type none --ctx_logit_mode none --ckpt_root "$OutputRoot\saint_baseline"
& $Python train_context.py @Common @EpochArgs --model_name saint --context_type main --ctx_logit_mode scaled --ckpt_root "$OutputRoot\saint_main_scaled"
& $Python train_context.py @Common @EpochArgs --model_name saint --context_type all --ctx_logit_mode scaled --ckpt_root "$OutputRoot\saint_all_scaled"
& $Python train_context.py @Common @EpochArgs --model_name saint --context_type main --ctx_logit_mode none --ckpt_root "$OutputRoot\saint_main_no_ctxlogit"
