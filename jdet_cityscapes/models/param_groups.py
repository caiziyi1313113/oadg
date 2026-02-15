from jdet.utils.registry import MODELS


@MODELS.register_module()
def FDDParamGroups(named_params, model=None, fdd_key="fdd", fdd_grad_mult=1.0, fdd_weight_decay=None):
    base_params = []
    fdd_params = []
    for name, p in named_params:
        if fdd_key in name:
            fdd_params.append(p)
        else:
            base_params.append(p)

    groups = [{"params": base_params}]
    if fdd_params:
        g = {"params": fdd_params}
        if fdd_grad_mult != 1.0:
            g["grad_mutilpy"] = fdd_grad_mult
        if fdd_weight_decay is not None:
            g["weight_decay"] = fdd_weight_decay
        groups.append(g)
    return groups
