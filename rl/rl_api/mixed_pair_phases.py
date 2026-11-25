"""Score each reciprocal strength once and every directed phase independently."""
import math
import torch
from .reciprocal_pairs import pair_layout,check_shared_actions
from .exploration import shared_pair_parameters,mirror_pairs


def sample_mixed_pairs(policy,features,masks,*,mirrored=False,continuous_mean=False,
                       return_phase_parameters=False):
    batch,candidates,edges=masks.shape
    repeated=features[:,None].expand(-1,candidates,-1,-1).reshape(-1,features.shape[1],4)
    mean,log_std,_,pooled=policy.parameters_for_edges(repeated,
        topology_masks=masks.reshape(-1,edges),return_pooled=True)
    mean,log_std=mean.reshape(batch,candidates,2*edges),log_std.reshape(batch,candidates,2*edges)
    check_shared_actions(mean,masks,features.shape[1],share_phases=False)
    bm,bs=policy.budget_parameters(repeated,pooled_states=pooled)
    bm,bs=bm.reshape(batch,candidates),bs.reshape(batch,candidates)
    scale=policy.config.exploration_std_scale
    if scale==0. or continuous_mean:
        result=(mean.clone(),bm.clone(),features.new_zeros((batch,candidates)))
        return (*result,mean[...,edges:],log_std[...,edges:].exp()) if return_phase_parameters else result
    canonical,expansion,_=pair_layout(features.shape[1],features.device)
    pairs=len(canonical)
    columns=torch.cat((canonical,torch.arange(edges,2*edges,device=features.device)))
    mean,std=mean[...,columns],log_std.exp()[...,columns]
    slots=torch.arange(candidates,device=features.device)%policy.config.actions_per_topology
    wide=round(policy.config.actions_per_topology*policy.config.phase_exploration_fraction)
    extra=(slots<wide).to(std.dtype)[None,:,None]*policy.config.phase_exploration_start_std*scale/math.sqrt(2.)
    std=torch.cat((std[...,:pairs],torch.sqrt(std[...,pairs:].square()+extra.square())),dim=-1)
    if mirrored:
        if candidates%policy.config.actions_per_topology or wide%2 or not torch.equal(masks[:,::2],masks[:,1::2]):
            raise ValueError('Mixed sampling needs complete mirrored groups and matching masks')
        mean,std=shared_pair_parameters(mean),shared_pair_parameters(std)
        bm,bs=shared_pair_parameters(bm),shared_pair_parameters(bs)
    normal=torch.distributions.Normal(mean,std)
    actions=normal.sample()
    budget_normal=torch.distributions.Normal(bm,bs.exp())
    budget=budget_normal.sample()
    if mirrored:
        actions,budget=mirror_pairs(actions,mean),mirror_pairs(budget,bm)
    score_mask=torch.cat((masks[...,canonical],masks),dim=-1)
    score=(normal.log_prob(actions)*score_mask).sum(-1)+budget_normal.log_prob(budget)
    full=torch.cat((actions[...,:pairs][...,expansion],actions[...,pairs:]),dim=-1)
    result=(full,budget,score)
    return (*result,mean[...,pairs:],std[...,pairs:]) if return_phase_parameters else result


def mixed_pair_distributions(policy, features, masks, *, mirrored=True):
    """Reevaluate the sampler's independent coordinates, without drawing actions.

    Strengths use one canonical coordinate per pair; phases use every directed
    coordinate. Wide slots and mirrored parameter sharing match sampling.
    """
    batch, candidates, edges = masks.shape
    size = features.shape[1]
    repeated = features[:, None].expand(-1, candidates, -1, -1).reshape(-1, size, 4)
    mean, log_std, _, pooled = policy.parameters_for_edges(repeated,
        topology_masks=masks.reshape(-1, edges), return_pooled=True)
    mean, std = mean.reshape(batch, candidates, 2*edges), log_std.exp().reshape(batch, candidates, 2*edges)
    canonical, _, _ = pair_layout(size, features.device)
    columns = torch.cat((canonical, torch.arange(edges, 2*edges, device=features.device)))
    mean, std = mean[..., columns], std[..., columns]
    wide = round(policy.config.actions_per_topology*policy.config.phase_exploration_fraction)
    slots = torch.arange(candidates, device=features.device) % policy.config.actions_per_topology
    extra = (slots < wide).to(std.dtype)[None, :, None]
    extra = extra*policy.config.phase_exploration_start_std*policy.config.exploration_std_scale/math.sqrt(2.)
    pairs = len(canonical)
    std = torch.cat((std[..., :pairs], torch.sqrt(std[..., pairs:].square()+extra.square())), -1)
    bm, bs = policy.budget_parameters(repeated, pooled_states=pooled)
    bm, bs = bm.reshape(batch, candidates), bs.exp().reshape(batch, candidates)
    if mirrored:
        mean, std = shared_pair_parameters(mean), shared_pair_parameters(std)
        bm, bs = shared_pair_parameters(bm), shared_pair_parameters(bs)
    return torch.distributions.Normal(mean, std), torch.distributions.Normal(bm, bs), columns
