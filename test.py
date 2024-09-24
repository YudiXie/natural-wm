from models.deepdive.model_options import *
import numpy as np

options = get_model_options(model_source='taskonomy')
model_names = options.keys()
print(model_names)
exit(0)

for model_string in model_names:
    image_transforms = get_recommended_transforms(model_string, input_type='numpy')
    print(model_string, image_transforms)
    model = eval(options[model_string]['call'])
    
    image = np.zeros((224, 224, 3)).astype(np.uint8)
    image = image_transforms(image)
    image = torch.rand(1, 3, 224, 224)
    out = model(image)
    print(model_string, out.shape)
