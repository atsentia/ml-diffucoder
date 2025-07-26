#!/usr/bin/env python3
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"

import pickle
from flax.core import frozen_dict

# Load params
with open('/Users/amund/ml-diffucoder/models/DiffuCoder-7B-JAX-original/params.pkl', 'rb') as f:
    params = pickle.load(f)

print(f"Type: {type(params)}")
print(f"Is FrozenDict: {isinstance(params, frozen_dict.FrozenDict)}")

if isinstance(params, (dict, frozen_dict.FrozenDict)):
    print(f"Keys: {list(params.keys())}")
    
    # Look for params key
    if 'params' in params:
        inner = params['params']
        print(f"\nInner type: {type(inner)}")
        if isinstance(inner, (dict, frozen_dict.FrozenDict)):
            print(f"Inner keys: {list(inner.keys())[:10]}")