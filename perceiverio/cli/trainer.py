from pytorch_lightning.plugins import DDPPlugin


class DDPStaticGraphPlugin(DDPPlugin):
    def _setup_model(self, model):
        wrapped = super()._setup_model(model)
        wrapped._set_static_graph()
        return wrapped
