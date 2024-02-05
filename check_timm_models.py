import timm

result = []                                                                 

model_names = ['seresnext50_32x4d']
model_names += timm.list_models('*convnext*')
model_names += timm.list_models('*swin*')
model_names += timm.list_models('*regnet*')
model_names += timm.list_models('eva*')
model_names += timm.list_models('*vit*')
model_names += timm.list_models('*nfnet*')

for m in model_names:                                                       
    model = timm.create_model(m, pretrained=False)                          
    n = sum([p.numel() for p in model.parameters()])                        
    result.append( (m,n) )                                                  
    print(n, m)  

for m,n in sorted(result):
    print(n, m)

