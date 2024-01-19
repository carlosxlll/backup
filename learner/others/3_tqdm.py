from tqdm import tqdm
import time
for i in tqdm(range(100)):
    time.sleep(0.01)

for i in tqdm(range(100), desc='Loading...', bar_format='{desc}: {percentage:3.0f}%|{bar}'):
    time.sleep(0.1)