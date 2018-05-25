import scipy
from glob import glob
import numpy as np

dir_real = '/media/kylelindgren/Data/nyu_processed/'
dir_virt = '/media/kylelindgren/Data/princeton_processed/'

dir_real_test = '/media/kylelindgren/Data/nyu_test/'
dir_virt_test = '/media/kylelindgren/Data/princeton_test/'

class DataLoader():
    def __init__(self, img_res=(128, 128)):
        self.img_res = img_res

    def load_data(self, domain, batch_size=1, is_testing=False):
        if is_testing:
            if domain == "real":
                path = glob(dir_real_test + '*')
            elif domain == "virt":
                path = glob(dir_virt_test + '*')

        else:
            if domain == "real":
                dir_path = glob(dir_real + '*')
            elif domain == "virt":
                dir_path = glob(dir_virt + '*')
            else:
                print("unknown domain")
                return -1
            n_flds = int(len(dir_path))
            idx_fld = np.random.randint(low=0, high=n_flds, dtype='int')
            path = sorted(glob(dir_path[idx_fld] + '/*'))

        batch_images = np.random.choice(path, size=batch_size)

        imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)
            if not is_testing:
                img = scipy.misc.imresize(img, self.img_res)

                if np.random.random() > 0.5:
                    img = np.fliplr(img)
            else:
                img = scipy.misc.imresize(img, self.img_res)
            imgs.append(img)

        imgs = np.array(imgs)/127.5 - 1
        imgs = np.expand_dims(imgs, axis=3)

        return imgs

    # demo function, not used
    def load_img(self, path):
        img = self.imread(path)
        img = scipy.misc.imresize(img, self.img_res)
        img = img/127.5 - 1.
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        # return scipy.misc.imread(path, mode='RGB').astype(np.float)
        return scipy.misc.imread(path, mode='L').astype(np.float)
