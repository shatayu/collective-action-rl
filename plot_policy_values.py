# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

def get_policy_means(filename):
    f = open(filename, "r")
    result = [json.loads(line) for line in f]
    f.close()

    return result


test = get_policy_means("maddpgcompetitivegamestate_policymeans.txt")
print(test[0])

# Data
# df=pd.DataFrame({'x_values': range(1, 50001), 'y1_values': np.random.randn(10), 'y2_values': np.random.randn(10)+range(1,11), 'y3_values': np.random.randn(10)+range(11,21) })
