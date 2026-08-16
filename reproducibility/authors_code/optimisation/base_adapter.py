# adapters/base_adapter.py
class BaseModelAdapter:
    def __init__(self,  model_id, device='cuda', image_size=(336, 336), patch_size=(224, 224), patch_only=True, optimise_text=False):
        self.device = device
        self.model_id = model_id
        self.processor, self.model = self.load(model_id)
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_only = patch_only
        self.optimise_text = optimise_text
        #self.img_root = img_root

    def load(self, model_id):
        raise NotImplementedError

    def compute_loss(self, target, patch): 
        raise NotImplementedError

    def generate(self, inputs):
        raise NotImplementedError
