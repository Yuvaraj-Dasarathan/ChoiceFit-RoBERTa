"""Inspect the saved checkpoint and verify dynamic-choice invariants."""
import json
from pathlib import Path
import torch
from choicefit_roberta import ChoiceModel

model = ChoiceModel('artifacts/choicefit-roberta')
base = {'billing':'Payments, charges and billing discrepancies',
        'HR':'Employee benefits and workplace questions'}
results=[]
for message in ['My card was charged twice.', 'How do I enroll in employee dental benefits?']:
    result=model.predict(message,base)
    reversed_result=model.predict(message,dict(reversed(list(base.items()))))
    renamed=model.predict(message,dict(zip(['option_a','option_b'],base.values())))
    assert abs(sum(result['probabilities'].values())-1) < 1e-6
    assert all(abs(value-reversed_result['probabilities'][key]) < 1e-5
               for key,value in result['probabilities'].items())
    assert all(abs(a-b) < 1e-5 for a,b in zip(result['probabilities'].values(),renamed['probabilities'].values()))
    results.append({'message':message,**result})
Path('artifacts/choicefit-roberta/inference_checks.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
