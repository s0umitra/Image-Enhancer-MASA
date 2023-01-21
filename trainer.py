
from dataset.data_handler import DIV2K
from train.scripts import SrganGeneratorTrainer, SrganTrainer
from models.scripts import discriminator, generator


def dataset_loader():
    div2k_train = DIV2K(scale=4, subset='train', downgrade='bicubic')
    div2k_valid = DIV2K(scale=4, subset='valid', downgrade='bicubic')

    train_ds = div2k_train.dataset(batch_size=16, random_transform=True)
    valid_ds = div2k_valid.dataset(batch_size=16, random_transform=True, repeat_count=1)

    return train_ds, valid_ds

def preTrain(train_ds, valid_ds):
    
    pre_save_path = "train/models/preModel.h5"
    pre_trainer = SrganGeneratorTrainer(model=generator(), checkpoint_dir=f'.ckpt/pre_generator')

    pre_trainer.train(train_ds,
                    valid_ds.take(10),
                    steps=1000000, 
                    evaluate_every=1000, 
                    save_best_only=False)

    pre_trainer.model.save_weights(pre_save_path)

def ganTrain(train_ds, valid_ds):
    
    pre_save_path = "train/models/preModel.h5"
    gan_save_path = "train/models/ganModel.h5"
    dis_save_path = "train/models/disModel.h5"

    gan_generator = generator()
    gan_generator.load_weights(pre_save_path)

    gan_trainer = SrganTrainer(generator=gan_generator, discriminator=discriminator())
    gan_trainer.train(train_ds, steps=200000)

    gan_trainer.generator.save_weights(gan_save_path)
    gan_trainer.discriminator.save_weights(dis_save_path)

def trainModel():

    train_ds, valid_ds = dataset_loader()

    preTrain(train_ds, valid_ds)
    ganTrain(train_ds, valid_ds)

    
# if __name__ == "__main__" :
    # trainModel()
    # print("haha")