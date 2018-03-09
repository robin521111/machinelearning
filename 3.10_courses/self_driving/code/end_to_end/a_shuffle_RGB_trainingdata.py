# (3) 数据打乱处理
# 利用train_test_split函数将样本数据进行打乱处理，增加数据的随机性。
# shuffle train data 
def shuffle_data(imgs_train_RGB, wheels_train_RGB):
    from sklearn.model_selection import train_test_split
    X_train_RGB, X_val_RGB, y_train_RGB, y_val_RGB = train_test_split(imgs_train_RGB, wheels_train_RGB, test_size=None, random_state=28)
    return X_train_RGB, y_train_RGB