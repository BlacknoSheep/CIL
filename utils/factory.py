def get_model(model_name, args):
    name = model_name.lower()
    if name == "finetune":
        from models.finetune import Finetune

        return Finetune(args)
    elif name == "joint":
        from models.joint import Joint

        return Joint(args)
    elif name == "ncm":
        from models.ncm import NCM

        return NCM(args)
    elif name == "reprojection":
        from models.reprojection import Reprojection

        return Reprojection(args)
    elif name == "momentum":
        from models.momentum import Momentum

        return Momentum(args)
    elif name == "demo":
        from models.demo import Demo

        return Demo(args)
    else:
        assert 0, "Model {} not available".format(name)
