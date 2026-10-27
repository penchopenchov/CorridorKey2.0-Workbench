"""Compare every renderer branch with the pre-optimization Git snapshot."""
import json
from pathlib import Path
import sys
FOLDER=Path(__file__).resolve().parent
ROOT=FOLDER.parents[1]
sys.path[:0]=[str(ROOT),str(FOLDER)]
import numpy as np
import torch
from corridorkey import runtime as current
from micro_reference import load_reference, REVISION

reference=load_reference()
torch.manual_seed(41)
meta={'model_native_hw':(73,101),'shape_hw':(128,128),'output_hw':(73,101),
      'pad':(13,14,27,28),'linear_brightness':2.,'linear_contrast':1.2,
      'linear_contrast_pivot':.18}
logits=torch.randn(2,4,128,128,device='cuda',dtype=torch.bfloat16)*1.5
checker=torch.from_numpy(current.make_checker(73,101))
checker_linear=current.cu.srgb_to_linear(checker.cuda()).unsqueeze(0)
records=[]

def resolve(rendered):
    return {k:(v.resolve() if hasattr(v,'resolve') else v) for k,v in rendered.items()}

for strength in (0.,.5,1.):
    for cutout_linear in (False,True):
        for kind in ('video','exr'):
            meta.update(despill_strength=strength,cutout_linear=cutout_linear)
            name='_render_predictions_exr_torch' if kind=='exr' else '_render_predictions_torch'
            old=getattr(reference,name)
            new=getattr(current,name)
            for outputs in ({'alpha'}, {'fg'}, {'cutout'}, {'checker'}, {'alpha','fg','cutout','checker'}):
                expected=resolve(old(meta,outputs,checker,logits))
                for cached in (False,True):
                    actual=resolve(new(meta,outputs,checker,logits,
                                       checker_linear=checker_linear if cached else None))
                    # Float bit patterns also catch signed-zero changes in HDR EXR output.
                    for key in outputs:
                        a,b=actual[key],expected[key]
                        np.testing.assert_array_equal(a.view(np.uint8),b.view(np.uint8))
                    records.append({'strength':strength,'cutout_linear':cutout_linear,
                                    'format':kind,'outputs':sorted(outputs),'cached':cached,'bit_exact':True})
result={'reference_revision':REVISION,'cases':len(records),'all_bit_exact':True,'records':records}
(FOLDER/'micro-render-quality.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'cases':len(records),'all_bit_exact':True}),flush=True)
