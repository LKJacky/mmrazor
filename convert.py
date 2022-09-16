# %%

import json
import copy
from mmengine import fileio
old = fileio.load('./testx.yml')
print(old)

new = fileio.load('./test.json')
print(new)
# %%
module2group = {}
newx = copy.deepcopy(new)['channl_group_cfg']['groups']
for group_key in newx:
    replaced = False
    for layer in newx[group_key]['channels']['output_related']:
        name = layer['name']
        for old_key in old:
            old_key: str
            if 'mutable_out_channels' in old_key:
                old_key_ = old_key.replace('.mutable_out_channels', '')

                if old_key_ == name:
                    newx[group_key]['choice'] = (
                        old[old_key]['current_choice'])
                    replaced = True
                    break
    if not replaced:
        print(group_key, 'not replaced')
    newx[group_key].pop('init_args')
    newx[group_key].pop('channels')

ret = copy.deepcopy(new)
ret['channl_group_cfg']['groups'] = newx
print(json.dumps(ret, indent=4))

# %%
