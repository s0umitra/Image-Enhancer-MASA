from data_handler import DIV2K

path = "dataset/"
div2k_train = DIV2K(scale=4, subset='train', downgrade='bicubic')
div2k_valid = DIV2K(scale=4, subset='valid', downgrade='bicubic')

div2k_train.dataset(batch_size=16, random_transform=True)
div2k_valid.dataset(batch_size=16, random_transform=True, repeat_count=1)