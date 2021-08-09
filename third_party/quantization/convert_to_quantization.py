
def convert2quantization(model, cfg):
    # quantization on
    quantization = cfg.MODEL.QUANTIZATION if hasattr(cfg.MODEL, 'QUANTIZATION') else None
    if quantization is not None:
        for item in quantization.scope:
            if len(item) == 0:
                continue
            attrs = item.split('.')
            cur = model
            find = True
            for attr in attrs:
                if hasattr(cur, attr):
                    cur = getattr(cur, attr)
                else:
                    find = False
            #print('find model to convert', find, item)
            if find:
                for m in cur.modules():
                    if hasattr(m, 'convert_to_quantization_version'):
                        m.convert_to_quantization_version(quantization)
                        #print('model to convert in block', item)
    # quantization off
