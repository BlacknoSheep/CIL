def get_model(model_name, args):
    name = model_name.lower()
    if name == "joint":
        from models.joint import Joint
        return Joint(args)
    elif name == "test":
        from models.test import Test
        return Test(args)
    elif name == "momentum":
        from models.momentum import Momentum
        return Momentum(args)
    elif name == "denoise":
        from models.denoise import Denoise
        return Denoise(args)
    else:
        assert 0, "Model {} not available".format(name)
