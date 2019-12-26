from tabpy.tabpy_tools.client import Client

client = Client('http://localhost:9004/')


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
    comps = pca.fit_transform(dat)  # (358,4). 레코드별 pca1-4 값

    return comps[:, _arg1[0] - 1].tolist()


# pca1 = comps[:,0]
# pca2 = comps[:,1]
# pca3 = comps[:,2]
# pca4 = comps[:,3]

# new_params = pca.components_
# (4,12). column마다 4개씩.
# pca1-4들의 의미를 알 수 있다.
# new_params[0,:]: Z1이 어떻게 구성돼있나. 중요.
# new_params[:,0]: X1이 어떻게 구성돼있나. 이걸로 화살표 그림.
# new_params[pca1, column_interest] 가 x 좌표
# new_params[pca2, column_interest] 가 y 좌표


client.deploy('PCA', PCA, 'This is using sklearn PCA',override=True)
