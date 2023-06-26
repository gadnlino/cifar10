import requests
import os

files = {
    'data_batch_1': 'https://drive.google.com/uc?id=1J2czPbgvD8s0RBjXry60I7kslMgPqvNr&confirm=t',
    'data_batch_2': 'https://drive.google.com/uc?id=1b7KAUgO3gDm3c70NLwBSAfwsLhAtM5YN&confirm=t',
    'data_batch_3': 'https://drive.google.com/uc?id=1hOcLDHXKVINWXFG42-lVPLhpdnj2rfDz&confirm=t',
    'data_batch_4': 'https://drive.google.com/uc?id=1nFd9bI8i9MRbQNz3AAX2pOg6EpOybphQ&confirm=t',
    'data_batch_5': 'https://drive.google.com/uc?id=18jULpgGqMvf8OIHeT-q0HUEA9rAKt0CK&confirm=t',
    'test_batch': 'https://drive.google.com/uc?id=1YLjUfR3HAeoH4icL1BRyAtkeiyX2PvSm&confirm=t',
    'batches.meta': 'https://drive.google.com/uc?id=1Sz9FXGUqPv69MR6xuH_A59234M_z5ep1&confirm=t'
}

folder_path = os.path.join('.','files','dataset')
os.makedirs(folder_path, exist_ok=True)

for file_name, link in files.items():
    file_path = os.path.join(folder_path,file_name)
    if(not os.path.exists(file_path)):
        r = requests.get(link, allow_redirects=True)

        if(r.status_code >= 200 and r.status_code <= 299):
            with open(file_path, 'wb') as f:
                f.write(r.content)
                print(f'File {file_path} created')
        else:
            print(f'Failed when download files: {r.json()}')