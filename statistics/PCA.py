def PCA(_arg1, _arg2, *_argN):
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    cols = [_arg2] + list(_argN)
    df = pd.DataFrame(data=cols).transpose()

    scale = StandardScaler()
    dat = scale.fit_transform(df)

    pca = PCA()
    comps = pca.fit_transform(dat)  # (358,p)

    return comps[:, _arg1[0] - 1].tolist()


from tabpy.tabpy_tools.client import Client

client = Client('http://localhost:9004/')
client.deploy('PCA', PCA, 'This is using sklearn PCA',override=True)


# pca1 = comps[:,0]
# pca2 = comps[:,1]
# pca3 = comps[:,2]
# pca4 = comps[:,3]

# new_params = pca.components_
# (4,12). column마다 4개씩.
# pca1-4들의 의미를 알 수 있다.
# new_params[0,:]: Z1이 어떻게 구성돼있나. 중요.
# new_params[pca1, column_interest] 가 x 좌표
# new_params[pca2, column_interest] 가 y 좌표

a1 = [1] * 100
a2 = [2] * 100
a3 = [3] * 100
[a1]
list([a2, a3])
cols = [a1]+list([a2, a3])
cols
len([a1]+list([a2, a3]))

import pandas as pd
pd.DataFrame({'m': a1})
pd.DataFrame()
pd.DataFrame(data=cols)
pd.DataFrame(data=cols).transpose()

