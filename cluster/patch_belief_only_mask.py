"""Belief-only treatment: an opt-in loss mask that zeroes the ACTION-frame loss on failed
rollout samples.  Everything else in the recipe is untouched.

Why: the default loss mask is all-ones and the official experiments never enable the
per-sample-type masking, so a failed counterfactual rollout currently trains the policy head
to imitate the failing action.  This flag lets the value/world-model frames learn from the
failure while the policy head never sees it as a target.  Idempotent. Backs up the file.
"""
import os
import pathlib, shutil, re
M = (pathlib.Path(os.environ["CUREWM_ROOT"])
     / "third_party/cosmos-policy/cosmos_policy/models/policy_text2world_model.py")
s = M.read_text()
if "mask_action_loss_on_failure_rollouts" in s:
    print("already patched"); raise SystemExit
shutil.copy(M, str(M) + ".bak_belief_only")

def once(old, new, tag):
    global s
    assert s.count(old) == 1, f"{tag}: expected exactly one anchor, found {s.count(old)}"
    s = s.replace(old, new)

# 1) config field, next to the existing masking switches
once("    mask_loss_for_action_future_state_prediction: bool = False\n",
     "    mask_loss_for_action_future_state_prediction: bool = False\n"
     "    # CureWM belief-only treatment: drop the ACTION-frame loss on FAILED rollout samples so the\n"
     "    # policy head never imitates a failing action; all other frames keep their loss.\n"
     "    mask_action_loss_on_failure_rollouts: bool = False\n", "config field")
# 2) thread the success mask into the loss function: call site + signature
once('            rollout_data_mask=data_batch["rollout_data_mask"],\n',
     '            rollout_data_mask=data_batch["rollout_data_mask"],\n'
     '            rollout_data_success_mask=data_batch["rollout_data_success_mask"],\n', "call site")
once("        rollout_data_mask: torch.Tensor,\n        world_model_sample_mask: torch.Tensor,\n",
     "        rollout_data_mask: torch.Tensor,\n        rollout_data_success_mask: torch.Tensor,\n        world_model_sample_mask: torch.Tensor,\n", "signature")
# 3) the mask itself, right after the all-ones initialisation (later blocks multiply into it, so zeros survive)
once("        final_mask_B_T = torch.ones((B, T), dtype=torch.long, device=sigma_B_T.device)  # All 1s mask initially\n",
     "        final_mask_B_T = torch.ones((B, T), dtype=torch.long, device=sigma_B_T.device)  # All 1s mask initially\n"
     "        if getattr(self.config, 'mask_action_loss_on_failure_rollouts', False):\n"
     "            # failed rollouts (ours and the official native failures): supervise belief and future state, never behaviour\n"
     "            fail_idx_B = ((rollout_data_mask == 1) & (rollout_data_success_mask == 0)).to(sigma_B_T.device)\n"
     "            if torch.any(fail_idx_B):\n"
     "                fail_batch_indices = torch.nonzero(fail_idx_B, as_tuple=False).squeeze(-1).to(torch.long)\n"
     "                final_mask_B_T[fail_batch_indices, action_indices[fail_batch_indices].to(torch.long)] = 0\n", "mask block")
M.write_text(s)
print("patched:", M.name, "| backup:", M.name + ".bak_belief_only")
