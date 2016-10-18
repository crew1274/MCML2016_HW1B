from PIL import Image
import glob
import io
import os
import numpy as np
import sys

ATT_NFACES = 1
ATT_NIMAGES = 36
ATT_NIMAGES_PER_FACE = 36
ATT_DATA_DIR = 'att_faces'

ATT_NFACES = 1
ATT_NIMAGES = 1
ATT_NIMAGES_PER_FACE = 1
ATT_TEST_DIR = 'test_faces'

ATT_IMG_FORMAT = 'tif'
ATT_IMG_WIDTH = 128
ATT_IMG_HEIGHT = 128
OUTPUT_DIR = 'output'

def load_image(filename, verbose=False):
    img = Image.open(filename)
    result = np.array(img, dtype=np.uint8)

    if verbose:
        print('Loaded image file %s' % filename)

    return result

def load_images(path=ATT_DATA_DIR, fmt=ATT_IMG_FORMAT):
    pattern = os.path.join(path, '*/*.' + fmt)
    filenames = np.array(sorted(glob.glob(pattern)))
    data = np.array([load_image(f).flatten() for f in filenames])
    return data

def load_test_images(path=ATT_TEST_DIR, fmt=ATT_IMG_FORMAT):
    pattern = os.path.join(path, '*/*.' + fmt)
    filenames = np.array(sorted(glob.glob(pattern)))
    data = np.array([load_image(f).flatten() for f in filenames])
    return data

def save_image(filename, data,dirname=OUTPUT_DIR, imgdim=(1, 1), imgsize=(ATT_IMG_WIDTH, ATT_IMG_HEIGHT),
               order='row', verbose=True):
    pathname = os.path.join(dirname, filename)
    ncols, nrows = imgdim
    w, h = imgsize
    width = w * ncols
    height = h * nrows
    img = Image.new('L', (width, height))
    idx = 0

    if (data.ndim > 1):
        if (order == 'row'):
            for x in range(0, width, w):
                for y in range(0, height, h):
                    pixels = normalize(data[idx])
                    seg = Image.new('L', imgsize)
                    seg.putdata(pixels)
                    img.paste(seg, (x, y))
                    idx += 1
        elif (order == 'column'):
            for y in range(0, height, h):
                for x in range(0, width, w):
                    pixels = normalize(data[idx])
                    seg = Image.new('L', imgsize)
                    seg.putdata(pixels)
                    img.paste(seg, (x, y))
                    idx += 1
    else:
        pixels = normalize(data)
        img.putdata(pixels)

    img.save(pathname)

    if verbose:
        print('Saved image file %s' % pathname)


def center(data):
    mu = np.mean(data, axis=0)
    return (data - mu), mu


def normalize(eigvec):
    minval, maxval = np.min(eigvec), np.max(eigvec)
    scale = 255.0 / (maxval - minval)
    result = (eigvec - minval) * scale
    return result.astype(np.uint8)


def project(basis, img, mu):
    return np.dot(img - mu, basis.T)


def reconstruct(basis, weights, mu):
    return np.dot(weights, basis) + mu


def distribute(basis, data, mu):
    return np.array([project(basis, d, mu) for d in data])


def pca(data, k=0):
    U, mu = center(data)
    cov = np.dot(U, U.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = (eigvals > 0)
    eigvecs = np.dot(U.T, eigvecs).T[::-1][idx]
    D = np.sqrt(eigvals[idx])[::-1]
    k = eigvecs.shape[0] if (k <= 0) else k

    for i in range(eigvecs.shape[1]):
        eigvecs[:, i] /= D

    return D[:k], eigvecs[:k], mu


def euclidean(proj1, proj2):
    return np.sqrt(np.sum(np.power(proj1 - proj2, 2)))


def mahalanobis(proj1, proj2, covinv):
    diff = proj1 - proj2
    return np.sqrt(np.dot(diff, np.dot(covinv, diff.T)))


def predict(*args, metric=euclidean):
    mindist, pred = F32_MAX, -1
    weights = args[0]

    for i in range(weights.shape[0]):
        dist = metric(weights[i], *args[1:])
        if (dist < mindist):
            mindist, pred = dist, i

    return pred


def split(data, nclasses=ATT_NIMAGES_PER_FACE, ratio=3/4):
    classes = np.arange(data.shape[0]) // nclasses
    ntrain = np.rint(ratio * nimgs)
    idx = np.random.permutation(nimgs)
    trainset = data[idx < ntrain]
    traincls = classes[idx < ntrain]
    testset = data[idx >= ntrain]
    testcls = classes[idx >= ntrain]
    return (trainset, traincls, testset, testcls)


def classify(trainset, traincls, testset, testcls,
             metric=euclidean, ndim=0):
    preds, score = None, 0.0
    D, eigvecs, mu = pca(trainset, ndim)
    trainweights = distribute(eigvecs, trainset, mu)
    testweights = distribute(eigvecs, testset, mu)
    ntest = testset.shape[0]

    if (metric == euclidean):
        preds = np.array([predict(trainweights,
                                  testweights[i])
                          for i in range(ntest)])
    elif (metric == mahalanobis):
        covinv = np.linalg.inv(np.diag(D))
        preds = np.array([predict(trainweights,
                                  testweights[i],
                                  covinv,
                                  metric=mahalanobis)
                          for i in range(ntest)])

    score = (testcls == traincls[preds]).sum() / ntest

    return (preds, score)


if __name__ == '__main__':
    if not os.path.exists(ATT_DATA_DIR):
        print('Can not retrieving floder att_faces')
        sys.exit(0)

    print('Loading image database...')
    data = load_images()
    nimgs = data.shape[0]

    print('\nDeriving PCA and reconstructing faces...')
    D, eigvecs, mu = pca(data)


    
    save_image('mean_image.png', mu)
    save_image('eigenfaces.png', eigvecs[:5],imgdim=(5, 1), order='column')

    subjects = load_test_images()
    save_image('test_mean_image.png', (subjects - mu))

    recfaces = np.array([reconstruct(
                                     eigvecs, project(eigvecs,subjects[i],mu),  mu)
                                     for i in range(subjects.shape[0]
                                    )])

    save_image('project_test_faces.png', np.vstack((subjects, recfaces)), imgdim=(1, 2))



    recfaces = np.array([reconstruct(eigvecs,project(eigvecs, subjects, mu),mu)
                         for i in range(subjects.shape[0])])

    save_image('reconstructed_test_faces.png', np.vstack((subjects, recfaces)), imgdim=(1, 2))
    

    subject = load_test_images()
    minvecs, maxvecs = 8, nimgs+8
    steps = np.arange(minvecs, maxvecs, step=minvecs)
    recfaces = np.array([reconstruct(eigvecs[:s],project(eigvecs[:s], subject, mu),mu)
                         for s in steps])

    save_image('partial_reconstruction.png', recfaces,imgdim=(8, 3), order='column')